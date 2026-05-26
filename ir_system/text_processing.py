from __future__ import annotations

import re
from typing import Iterable

from .lexicon import AMHARIC_STOPWORDS, ENGLISH_STOPWORDS, AMHARIC_TO_ENGLISH, ENGLISH_TO_AMHARIC, TERM_GROUPS

AMHARIC_CHAR_RE = re.compile(r"[\u1200-\u137F]")
ENGLISH_WORD_RE = re.compile(r"[a-z0-9']+")
AMHARIC_WORD_RE = re.compile(r"[\u1200-\u137F]+")
WHITESPACE_RE = re.compile(r"\s+")

AMHARIC_SUFFIXES = (
    "ዎች",
    "ዎቹ",
    "ዎችን",
    "ዎቹን",
    "ውን",
    "ው",
    "ዋ",
    "ን",
    "ም",
    "ች",
    "ይ",
)

ENGLISH_SUFFIXES = (
    ("ies", "y"),
    ("ing", ""),
    ("ed", ""),
    ("es", ""),
    ("s", ""),
)


def detect_language(text: str) -> str:
    amharic_count = len(AMHARIC_CHAR_RE.findall(text))
    latin_count = len(re.findall(r"[A-Za-z]", text))
    if amharic_count > latin_count:
        return "am"
    if latin_count > 0:
        return "en"
    return "en"


def normalize_text(text: str, language: str) -> str:
    lowered = text.lower()
    lowered = lowered.replace("-", " ")
    lowered = re.sub(r"[\u2018\u2019\u201c\u201d]", "", lowered)
    if language == "am":
        cleaned = re.sub(r"[^\u1200-\u137F0-9\s]", " ", lowered)
    else:
        cleaned = re.sub(r"[^a-z0-9\s]", " ", lowered)
    return WHITESPACE_RE.sub(" ", cleaned).strip()


def tokenize(text: str, language: str) -> list[str]:
    normalized = normalize_text(text, language)
    if language == "am":
        tokens = AMHARIC_WORD_RE.findall(normalized)
        return [simple_normalize_token(token, language) for token in tokens if token and token not in AMHARIC_STOPWORDS]
    tokens = ENGLISH_WORD_RE.findall(normalized)
    return [simple_normalize_token(token, language) for token in tokens if token and token not in ENGLISH_STOPWORDS]


def simple_normalize_token(token: str, language: str) -> str:
    if language == "am":
        for suffix in AMHARIC_SUFFIXES:
            if token.endswith(suffix) and len(token) > len(suffix) + 1:
                return token[: -len(suffix)]
        return token

    if token.endswith("'s") and len(token) > 3:
        token = token[:-2]
    for suffix, replacement in ENGLISH_SUFFIXES:
        if token.endswith(suffix) and len(token) > len(suffix) + 1:
            return token[: -len(suffix)] + replacement
    return token


def translation_options(token: str, language: str) -> list[str]:
    if language == "en":
        return ENGLISH_TO_AMHARIC.get(token, [])
    return AMHARIC_TO_ENGLISH.get(token, [])


def related_tokens(token: str, language: str) -> list[str]:
    lookup_key = token
    if language == "en" and token not in TERM_GROUPS:
        for key, group in TERM_GROUPS.items():
            if token in group:
                lookup_key = key
                break
    elif language == "am" and token not in TERM_GROUPS:
        for key, group in TERM_GROUPS.items():
            if token in group:
                lookup_key = key
                break

    group = TERM_GROUPS.get(lookup_key, set())
    return [related for related in group if related != token]


def expand_query_tokens(tokens: Iterable[str], language: str, expand: bool = True) -> list[tuple[str, float]]:
    weighted_terms: list[tuple[str, float]] = []
    seen: set[str] = set()

    for token in tokens:
        if token and token not in seen:
            weighted_terms.append((token, 1.0))
            seen.add(token)

        for translated in translation_options(token, language):
            if translated not in seen:
                weighted_terms.append((translated, 0.9))
                seen.add(translated)

        if expand:
            for related in related_tokens(token, language):
                if related not in seen:
                    weighted_terms.append((related, 0.6))
                    seen.add(related)

    return weighted_terms
