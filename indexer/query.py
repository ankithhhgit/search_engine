import json
import time
from pathlib import Path

from indexer.bm25 import BM25
from indexer.index import InvertedIndex
from indexer.database import get_pagerank_score, get_stats as db_stats
from indexer.text_processor import process

INDEX_PATH = Path("data/index.json")

BM25_WEIGHT     = 0.7
PAGERANK_WEIGHT = 0.3

_index: InvertedIndex | None = None
_bm25:  BM25 | None          = None


def _get_engine() -> tuple[InvertedIndex, BM25]:
    global _index, _bm25
    if _index is None:
        _index = InvertedIndex.load(INDEX_PATH)
        _bm25  = BM25(_index)
    return _index, _bm25


def _blend(bm25_score: float, url: str, bm25_max: float) -> float:
    normalised_bm25 = bm25_score / bm25_max if bm25_max > 0 else 0.0
    pr_score        = get_pagerank_score(url)
    return round(BM25_WEIGHT * normalised_bm25 + PAGERANK_WEIGHT * pr_score, 4)


def _make_snippet(doc_id: str, query_tokens: list[str], max_len: int = 160) -> str:
    clean_path = Path("data/clean") / f"{doc_id}.json"
    if not clean_path.exists():
        return ""
    try:
        with open(clean_path, encoding="utf-8") as f:
            page = json.load(f)
        tokens = page.get("tokens", [])
        if not tokens:
            return ""
        for i, token in enumerate(tokens):
            if token in query_tokens:
                start = max(0, i - 10)
                end   = min(len(tokens), i + 20)
                snippet_tokens = tokens[start:end]
                snippet = " ".join(snippet_tokens)
                if len(snippet) > max_len:
                    snippet = snippet[:max_len] + "..."
                return snippet
        snippet = " ".join(tokens[:30])
        if len(snippet) > max_len:
            snippet = snippet[:max_len] + "..."
        return snippet
    except Exception:
        return ""


def search(
    query_str: str,
    top_k: int = 10,
    mode: str = "OR",
) -> dict:
    start = time.perf_counter()

    index, bm25 = _get_engine()
    tokens = process(query_str)

    if not tokens:
        return _empty_result(query_str, tokens, start)

    missing       = [t for t in tokens if not index.contains(t)]
    active_tokens = [t for t in tokens if index.contains(t)]

    if not active_tokens:
        return _empty_result(query_str, tokens, start)

    if mode == "AND":
        candidates = _and_candidates(index, active_tokens)
    else:
        candidates = _or_candidates(index, active_tokens)

    if not candidates:
        return _empty_result(query_str, tokens, start)

    # ── build scored list with snippet ──────────────────────
    scored = []
    for doc_id in candidates:
        doc = index.get_doc(doc_id)
        scored.append({
            "doc_id":     doc_id,
            "bm25_score": round(bm25.score(active_tokens, doc_id), 4),
            "snippet":    _make_snippet(doc_id, active_tokens),
            **doc,
        })

    bm25_max = max((r["bm25_score"] for r in scored), default=1.0)

    for result in scored:
        result["score"] = _blend(result["bm25_score"], result["url"], bm25_max)

    scored.sort(key=lambda x: x["score"], reverse=True)
    results = scored[:top_k]

    elapsed_ms = round((time.perf_counter() - start) * 1000, 2)

    return {
        "query":      query_str,
        "tokens":     tokens,
        "missing":    missing,
        "mode":       mode,
        "total_hits": len(scored),
        "results":    results,
        "elapsed_ms": elapsed_ms,
    }


def _and_candidates(index: InvertedIndex, tokens: list[str]) -> set[str]:
    sets = [set(index.get_all_docs_with_term(t)) for t in tokens]
    return set.intersection(*sets)


def _or_candidates(index: InvertedIndex, tokens: list[str]) -> set[str]:
    sets = [set(index.get_all_docs_with_term(t)) for t in tokens]
    return set.union(*sets)


def _empty_result(query_str: str, tokens: list[str], start: float) -> dict:
    return {
        "query":      query_str,
        "tokens":     tokens,
        "missing":    tokens,
        "mode":       "OR",
        "total_hits": 0,
        "results":    [],
        "elapsed_ms": round((time.perf_counter() - start) * 1000, 2),
    }


if __name__ == "__main__":
    print("=== Blended search demo ===\n")

    queries = ["mystery books", "travel adventure", "fantasy"]
    for q in queries:
        response = search(q, top_k=3)
        print(f"Query: '{q}'  ({response['elapsed_ms']}ms)")
        for i, r in enumerate(response["results"], 1):
            print(f"  {i}. [{r['score']}]  bm25={r['bm25_score']}  "
                  f"pr={get_pagerank_score(r['url']):.4f}  "
                  f"snippet='{r['snippet'][:50]}...'  "
                  f"{r['title'][:45]}")
        print()