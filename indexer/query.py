import time
from pathlib import Path

from indexer.bm25 import BM25
from indexer.index import InvertedIndex
from indexer.text_processor import process

INDEX_PATH = Path("data/index.json")

_index: InvertedIndex | None = None
_bm25: BM25 | None = None


def _get_engine() -> tuple[InvertedIndex, BM25]:
    global _index, _bm25
    if _index is None:
        _index = InvertedIndex.load(INDEX_PATH)
        _bm25 = BM25(_index)
    return _index, _bm25


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

    missing = [t for t in tokens if not index.contains(t)]
    active_tokens = [t for t in tokens if index.contains(t)]

    if not active_tokens:
        return _empty_result(query_str, tokens, start)

    if mode == "AND":
        candidates = _and_candidates(index, active_tokens)
    else:
        candidates = _or_candidates(index, active_tokens)

    scored = [
        {
            "doc_id":  doc_id,
            "score":   round(bm25.score(active_tokens, doc_id), 4),
            **index.get_doc(doc_id),
        }
        for doc_id in candidates
    ]
    scored.sort(key=lambda x: x["score"], reverse=True)
    results = scored[:top_k]

    elapsed_ms = round((time.perf_counter() - start) * 1000, 2)

    return {
        "query":        query_str,
        "tokens":       tokens,
        "missing":      missing,
        "mode":         mode,
        "total_hits":   len(scored),
        "results":      results,
        "elapsed_ms":   elapsed_ms,
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


def _print_results(response: dict) -> None:
    print(f"\n  Query      : '{response['query']}'")
    print(f"  Tokens     : {response['tokens']}")
    if response["missing"]:
        print(f"  Not in index: {response['missing']}")
    print(f"  Mode       : {response['mode']}")
    print(f"  Hits       : {response['total_hits']} docs")
    print(f"  Time       : {response['elapsed_ms']}ms")
    print()

    if not response["results"]:
        print("  No results found.\n")
        return

    for i, r in enumerate(response["results"], 1):
        print(f"  {i}. [{r['score']:.4f}]  {r['title']}")
        print(f"       {r['url']}")
    print()


def cli() -> None:
    print("\n" + "=" * 50)
    print(" Search Engine CLI")
    print("=" * 50)
    print(" Commands:")
    print("   <query>        search with OR mode")
    print("   and:<query>    search with AND mode")
    print("   quit           exit")
    print("=" * 50 + "\n")

    _get_engine()

    while True:
        try:
            raw = input("Search: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye.")
            break

        if not raw:
            continue
        if raw.lower() == "quit":
            print("Goodbye.")
            break

        if raw.lower().startswith("and:"):
            query_str = raw[4:].strip()
            mode = "AND"
        else:
            query_str = raw
            mode = "OR"

        response = search(query_str, top_k=5, mode=mode)
        _print_results(response)


if __name__ == "__main__":
    print("=== Automated test queries ===\n")

    test_queries = [
        ("mystery books",   "OR"),
        ("travel",          "OR"),
        ("light fantasy",   "AND"),
        ("price",           "OR"),
        ("xyznonexistent",  "OR"),
    ]

    for query_str, mode in test_queries:
        response = search(query_str, top_k=3, mode=mode)
        _print_results(response)

    print("\n" + "=" * 50)
    print("Starting interactive CLI...")
    print("=" * 50)
    cli()