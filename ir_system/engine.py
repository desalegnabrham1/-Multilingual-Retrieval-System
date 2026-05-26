from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from difflib import SequenceMatcher
import json
import math
import re

from .text_processing import detect_language, expand_query_tokens, normalize_text, tokenize


@dataclass(frozen=True)
class Document:
    doc_id: str
    title: str
    language: str
    text: str
    source: str = ""


class IRSystem:
    def __init__(self, documents: list[Document]) -> None:
        self.documents = documents
        self.doc_by_id = {document.doc_id: document for document in documents}
        self.document_tokens: dict[str, list[str]] = {}
        self.document_frequencies: dict[str, Counter[str]] = {}
        self.inverted_index: dict[str, dict[str, int]] = defaultdict(dict)
        self.document_lengths: dict[str, int] = {}
        self.document_count = len(documents)
        self.average_document_length = 0.0
        self._build_index()

    @classmethod
    def from_json(cls, file_path: str | Path) -> "IRSystem":
        with Path(file_path).open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        documents = [
            Document(
                doc_id=item["id"],
                title=item["title"],
                language=item["language"],
                text=item["text"],
                source=item.get("source", ""),
            )
            for item in payload["documents"]
        ]
        return cls(documents)

    def _build_index(self) -> None:
        total_length = 0
        for document in self.documents:
            searchable_text = self._searchable_text(document)
            tokens = tokenize(searchable_text, document.language)
            frequencies = Counter(tokens)
            self.document_tokens[document.doc_id] = tokens
            self.document_frequencies[document.doc_id] = frequencies
            self.document_lengths[document.doc_id] = len(tokens)
            total_length += len(tokens)

            for term, frequency in frequencies.items():
                self.inverted_index[term][document.doc_id] = frequency

        self.average_document_length = total_length / self.document_count if self.document_count else 0.0

    def _searchable_text(self, document: Document) -> str:
        return f"{document.title}. {document.text}".strip()

    def detect_query_language(self, query: str, override: str | None = None) -> str:
        if override in {"en", "am"}:
            return override
        return detect_language(query)

    def build_query_terms(self, query: str, language: str, expand: bool = True) -> list[tuple[str, float]]:
        query_tokens = tokenize(query, language)
        return expand_query_tokens(query_tokens, language, expand=expand)

    def _idf(self, term: str) -> float:
        document_frequency = len(self.inverted_index.get(term, {}))
        return math.log((self.document_count + 1) / (document_frequency + 1)) + 1.0

    def _bm25_term_score(self, term: str, document_id: str, query_weight: float, k1: float = 1.5, b: float = 0.75) -> float:
        postings = self.inverted_index.get(term)
        if not postings or document_id not in postings:
            return 0.0

        term_frequency = postings[document_id]
        document_length = self.document_lengths.get(document_id, 0)
        denominator = term_frequency + k1 * (1 - b + b * (document_length / self.average_document_length if self.average_document_length else 1.0))
        idf = self._idf(term)
        return query_weight * idf * ((term_frequency * (k1 + 1)) / denominator)

    def _tfidf_term_score(self, term: str, document_id: str, query_weight: float) -> float:
        postings = self.inverted_index.get(term)
        if not postings or document_id not in postings:
            return 0.0

        term_frequency = postings[document_id]
        idf = self._idf(term)
        return query_weight * term_frequency * idf

    def _score_document(self, document_id: str, query_terms: list[tuple[str, float]], ranking: str) -> float:
        score = 0.0
        for term, weight in query_terms:
            if ranking == "tfidf":
                score += self._tfidf_term_score(term, document_id, weight)
            else:
                score += self._bm25_term_score(term, document_id, weight)
        return score

    def _matched_terms(self, document_id: str, query_terms: list[tuple[str, float]]) -> list[str]:
        tokens = set(self.document_tokens.get(document_id, []))
        return [term for term, _ in query_terms if term in tokens]

    def _snippet(self, text: str, terms: list[str], window: int = 120) -> str:
        lowered = text.lower()
        match_positions = [lowered.find(term.lower()) for term in terms if term]
        match_positions = [position for position in match_positions if position >= 0]
        if not match_positions:
            return text[:window].strip()
        start = max(0, min(match_positions) - 40)
        end = min(len(text), start + window)
        snippet = text[start:end].strip()
        return re.sub(r"\s+", " ", snippet)

    def _fuzzy_score_document(self, query: str, document: Document, query_language: str) -> float:
        normalized_query = re.sub(r"[^\w\u1200-\u137F\s]", " ", query.lower())
        normalized_document = re.sub(r"[^\w\u1200-\u137F\s]", " ", self._searchable_text(document).lower())
        normalized_query = re.sub(r"\s+", " ", normalized_query).strip()
        normalized_document = re.sub(r"\s+", " ", normalized_document).strip()
        if not normalized_query or not normalized_document:
            return 0.0

        character_similarity = SequenceMatcher(None, normalized_query, normalized_document).ratio()
        query_tokens = set(tokenize(query, query_language))
        document_tokens = set(self.document_tokens.get(document.doc_id, []))
        if query_tokens:
            token_overlap = len(query_tokens & document_tokens) / len(query_tokens)
        else:
            token_overlap = 0.0

        score = (0.7 * character_similarity) + (0.3 * token_overlap)
        return score if score > 0 else 0.01

    def fallback_search(
        self,
        query: str,
        top_k: int = 5,
        language_override: str | None = None,
    ) -> list[dict[str, Any]]:
        query_language = self.detect_query_language(query, override=language_override)
        scored_documents = []

        for document in self.documents:
            score = self._fuzzy_score_document(query, document, query_language)
            scored_documents.append(
                {
                    "doc": document,
                    "score": score,
                    "matched_terms": [],
                    "snippet": self._snippet(self._searchable_text(document), []),
                }
            )

        scored_documents.sort(key=lambda item: item["score"], reverse=True)
        return scored_documents[:top_k]

    def search(
        self,
        query: str,
        top_k: int = 5,
        ranking: str = "bm25",
        language_override: str | None = None,
        expand_query: bool = True,
    ) -> list[dict[str, Any]]:
        query_language = self.detect_query_language(query, override=language_override)
        query_terms = self.build_query_terms(query, query_language, expand=expand_query)

        scored_documents = []
        for document in self.documents:
            score = self._score_document(document.doc_id, query_terms, ranking)
            if score > 0:
                matched_terms = self._matched_terms(document.doc_id, query_terms)
                scored_documents.append(
                    {
                        "doc": document,
                        "score": score,
                        "matched_terms": matched_terms,
                        "snippet": self._snippet(document.text, matched_terms or [term for term, _ in query_terms]),
                    }
                )

        scored_documents.sort(key=lambda item: item["score"], reverse=True)
        return scored_documents[:top_k]

    def evaluate(self, queries: list[dict[str, Any]], qrels: dict[str, list[str]], top_k: int = 3) -> list[dict[str, Any]]:
        results = []
        for query_item in queries:
            query_id = query_item["id"]
            retrieved = self.search(
                query_item["query"],
                top_k=top_k,
                ranking=query_item.get("ranking", "bm25"),
                language_override=query_item.get("language"),
                expand_query=True,
            )
            retrieved_ids = [item["doc"].doc_id for item in retrieved]
            relevant_ids = set(qrels.get(query_id, []))
            retrieved_relevant = [doc_id for doc_id in retrieved_ids if doc_id in relevant_ids]

            precision = len(retrieved_relevant) / top_k if top_k else 0.0
            recall = len(retrieved_relevant) / len(relevant_ids) if relevant_ids else 0.0

            results.append(
                {
                    "query_id": query_id,
                    "query": query_item["query"],
                    "language": query_item.get("language", "auto"),
                    "retrieved_ids": retrieved_ids,
                    "relevant_ids": sorted(relevant_ids),
                    "precision_at_k": round(precision, 3),
                    "recall_at_k": round(recall, 3),
                }
            )

        return results

    @staticmethod
    def average_metrics(results: list[dict[str, Any]]) -> dict[str, float]:
        if not results:
            return {"precision_at_k": 0.0, "recall_at_k": 0.0}
        precision = sum(item["precision_at_k"] for item in results) / len(results)
        recall = sum(item["recall_at_k"] for item in results) / len(results)
        return {"precision_at_k": round(precision, 3), "recall_at_k": round(recall, 3)}
