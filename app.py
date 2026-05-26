from __future__ import annotations

import json
from pathlib import Path

from flask import Flask, render_template, request

from ir_system.engine import IRSystem
from ir_system.google_search import GoogleSearchClient

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
SYSTEM = IRSystem.from_json(DATA_DIR / "corpus.json")
GOOGLE_CLIENT = GoogleSearchClient.from_environment()

with (DATA_DIR / "test_queries.json").open("r", encoding="utf-8") as handle:
    TEST_QUERIES = json.load(handle)

with (DATA_DIR / "qrels.json").open("r", encoding="utf-8") as handle:
    QRELS = json.load(handle)

app = Flask(__name__)


@app.route("/", methods=["GET"])
def index() -> str:
    # Use 3 Amharic + 3 English short phrases for example chips
    def _shorten(text: str, max_words: int = 5) -> str:
        words = text.split()
        if len(words) <= max_words:
            return text
        return " ".join(words[:max_words]) + "..."

    am_queries = [q.get("query", "") for q in TEST_QUERIES if q.get("language") == "am"]
    en_queries = [q.get("query", "") for q in TEST_QUERIES if q.get("language") == "en"]

    selected = []
    for q in am_queries[:3]:
        selected.append(_shorten(q))
    for q in en_queries[:3]:
        selected.append(_shorten(q))

    if not selected:
        # fallback to first 6 queries if languages missing
        selected = [_shorten(q.get("query", "")) for q in TEST_QUERIES][:6]

    example_queries = [s.strip() for s in selected]
    return render_template("index.html", example_queries=example_queries, body_class="home-page", title="Multilingual IR System")


@app.route("/search", methods=["POST"])
def search() -> str:
    query = request.form.get("query", "").strip()
    language = request.form.get("language", "auto")
    ranking = request.form.get("ranking", "bm25")
    expand_query = request.form.get("expand_query") == "on"
    web_search = request.form.get("web_search") == "on"
    top_k = max(1, min(10, int(request.form.get("top_k", "5"))))

    detected_language = SYSTEM.detect_query_language(query, override=None if language == "auto" else language)
    query_terms = SYSTEM.build_query_terms(
        query,
        detected_language,
        expand=expand_query,
    ) if query else []
    results = []
    fallback_results = []
    web_results = []
    result_source = "local"
    if query:
        results = SYSTEM.search(
            query=query,
            top_k=top_k,
            ranking=ranking,
            language_override=None if language == "auto" else language,
            expand_query=expand_query,
        )
        if not results:
            fallback_results = SYSTEM.fallback_search(
                query=query,
                top_k=top_k,
                language_override=None if language == "auto" else language,
            )
        if web_search and GOOGLE_CLIENT.enabled:
            web_results = GOOGLE_CLIENT.search(query, top_k=top_k)

    visible_results = results or fallback_results
    if web_results and not visible_results:
        result_source = "web"
    elif fallback_results and not results:
        result_source = "fallback"

    return render_template(
        "results.html",
        query=query,
        language_choice=language,
        detected_language=detected_language,
        ranking=ranking,
        expand_query=expand_query,
        web_search=web_search,
        web_search_enabled=GOOGLE_CLIENT.enabled,
        top_k=top_k,
        query_terms=[term for term, _ in query_terms],
        results=results,
        fallback_results=fallback_results,
        web_results=web_results,
        visible_results=visible_results,
        result_source=result_source,
        body_class="results-page",
        title=f"{query or 'Search'} - Multilingual IR System",
    )


@app.route("/dataset", methods=["GET"])
def dataset() -> str:
    # pass documents list from IRSystem for template rendering
    documents = SYSTEM.documents
    corpus_count = len(documents)
    queries_count = len(TEST_QUERIES)
    qrels_count = len(QRELS)
    return render_template(
        "dataset.html",
        corpus_count=corpus_count,
        queries_count=queries_count,
        qrels_count=qrels_count,
        documents=documents,
        body_class="dataset-page",
        title="Dataset - Multilingual IR System",
    )


@app.route("/evaluation", methods=["GET"])
def evaluation() -> str:
    evaluation_k = max(1, min(10, int(request.args.get("k", "3"))))
    results = SYSTEM.evaluate(TEST_QUERIES, QRELS, top_k=evaluation_k)
    averages = IRSystem.average_metrics(results)
    return render_template(
        "evaluation.html",
        results=results,
        averages=averages,
        evaluation_k=evaluation_k,
        body_class="evaluation-page",
        title="Evaluation - Multilingual IR System",
    )


if __name__ == "__main__":
    app.run(debug=True)
