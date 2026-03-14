import json
import time
from functools import lru_cache
from pathlib import Path

from indexer.bm25 import BM25
from indexer.index import InvertedIndex
from indexer.database import get_pagerank_score
from indexer.text_processor import process

INDEX_PATH = Path("data/index.json")

BM25_WEIGHT     = 0.7
PAGERANK_WEIGHT = 0.3

_index: InvertedIndex | None = None
_bm25:  BM25 | None          = None

# ── cache state ──────────────────────────────────────────────
_cache_hits:   int = 0
_cache_misses: int = 0
_cache_log:    list[str] = []


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
                snippet = " ".join(tokens[start:end])
                return snippet[:max_len] + "..." if len(snippet) > max_len else snippet
        snippet = " ".join(tokens[:30])
        return snippet[:max_len] + "..." if len(snippet) > max_len else snippet
    except Exception:
        return ""


@lru_cache(maxsize=256)
def _cached_search(query_str: str, top_k: int, mode: str) -> dict:
    index, bm25 = _get_engine()
    tokens = process(query_str)

    if not tokens:
        return {
            "query": query_str, "tokens": tokens,
            "missing": tokens, "mode": mode,
            "total_hits": 0, "results": [], "elapsed_ms": 0.0,
        }

    missing       = [t for t in tokens if not index.contains(t)]
    active_tokens = [t for t in tokens if index.contains(t)]

    if not active_tokens:
        return {
            "query": query_str, "tokens": tokens,
            "missing": missing, "mode": mode,
            "total_hits": 0, "results": [], "elapsed_ms": 0.0,
        }

    if mode == "AND":
        candidates = _and_candidates(index, active_tokens)
    else:
        candidates = _or_candidates(index, active_tokens)

    if not candidates:
        return {
            "query": query_str, "tokens": tokens,
            "missing": missing, "mode": mode,
            "total_hits": 0, "results": [], "elapsed_ms": 0.0,
        }

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

    return {
        "query":      query_str,
        "tokens":     tokens,
        "missing":    missing,
        "mode":       mode,
        "total_hits": len(scored),
        "results":    scored[:top_k],
        "elapsed_ms": 0.0,
    }


def search(
    query_str: str,
    top_k: int = 10,
    mode: str = "OR",
) -> dict:
    global _cache_hits, _cache_misses, _cache_log

    start      = time.perf_counter()
    cache_info = _cached_search.cache_info()

    result = _cached_search(query_str.strip().lower(), top_k, mode)

    new_info = _cached_search.cache_info()
    cached   = new_info.hits > cache_info.hits

    if cached:
        _cache_hits += 1
        _cache_log.append(query_str)
    else:
        _cache_misses += 1

    elapsed_ms = round((time.perf_counter() - start) * 1000, 2)

    return {**result, "elapsed_ms": elapsed_ms, "cached": cached}


def get_cache_stats() -> dict:
    info       = _cached_search.cache_info()
    total      = _cache_hits + _cache_misses
    hit_rate   = round(_cache_hits / total * 100, 1) if total > 0 else 0.0
    top_cached = {}
    for q in _cache_log:
        top_cached[q] = top_cached.get(q, 0) + 1
    top_5 = sorted(top_cached.items(), key=lambda x: x[1], reverse=True)[:5]

    return {
        "hits":        _cache_hits,
        "misses":      _cache_misses,
        "hit_rate":    f"{hit_rate}%",
        "cache_size":  info.currsize,
        "max_size":    info.maxsize,
        "top_queries": [{"query": q, "hits": n} for q, n in top_5],
    }


def clear_cache() -> None:
    global _cache_hits, _cache_misses, _cache_log
    _cached_search.cache_clear()
    _cache_hits   = 0
    _cache_misses = 0
    _cache_log    = []


def _and_candidates(index: InvertedIndex, tokens: list[str]) -> set[str]:
    sets = [set(index.get_all_docs_with_term(t)) for t in tokens]
    return set.intersection(*sets)


def _or_candidates(index: InvertedIndex, tokens: list[str]) -> set[str]:
    sets = [set(index.get_all_docs_with_term(t)) for t in tokens]
    return set.union(*sets)


if __name__ == "__main__":
    print("=== Cache demo ===\n")

    queries = ["mystery books", "travel", "mystery books", "fantasy", "mystery books"]

    for q in queries:
        response = search(q, top_k=3)
        status = "HIT" if response["cached"] else "MISS"
        print(f"  [{status}]  '{q}'  →  {response['elapsed_ms']}ms  "
              f"({response['total_hits']} hits)")

    print("\n=== Cache stats ===")
    stats = get_cache_stats()
    print(f"  hits      : {stats['hits']}")
    print(f"  misses    : {stats['misses']}")
    print(f"  hit rate  : {stats['hit_rate']}")
    print(f"  cache size: {stats['cache_size']}/{stats['max_size']}")
    print(f"\n  top queries:")
    for entry in stats["top_queries"]:
        print(f"    '{entry['query']}' — {entry['hits']} hits")