import statistics
import time
from pathlib import Path

from indexer.index import InvertedIndex
from indexer.bm25 import BM25
from indexer.query import search

INDEX_PATH = Path("data/index.json")

QUERIES = [
    "mystery books",
    "travel adventure",
    "fantasy light",
    "crime thriller",
    "romance love",
    "science fiction",
    "history war",
    "children stories",
    "price cheap",
    "author review",
]


def benchmark_index_operations():
    print("=" * 50)
    print("INDEX OPERATIONS")
    print("=" * 50)

    start = time.perf_counter()
    index = InvertedIndex.build_from_disk()
    build_ms = (time.perf_counter() - start) * 1000
    print(f"  Build from clean files : {build_ms:.1f}ms")

    path = Path("data/index.json")
    start = time.perf_counter()
    index.save(path)
    save_ms = (time.perf_counter() - start) * 1000
    print(f"  Save to disk           : {save_ms:.1f}ms")

    start = time.perf_counter()
    InvertedIndex.load(path)
    load_ms = (time.perf_counter() - start) * 1000
    print(f"  Load from disk         : {load_ms:.1f}ms")
    print(f"  Speedup (build/load)   : {build_ms / load_ms:.1f}x\n")


def benchmark_queries():
    print("=" * 50)
    print("QUERY LATENCY  (10 queries x 5 runs)")
    print("=" * 50)

    search("warmup", top_k=5)

    all_times = []
    print(f"\n  {'query':<25} {'min':>7} {'avg':>7} {'max':>7}  hits")
    print(f"  {'-'*25} {'-'*7} {'-'*7} {'-'*7}  {'-'*4}")

    for query_str in QUERIES:
        times = []
        hits = 0
        for _ in range(5):
            response = search(query_str, top_k=10)
            times.append(response["elapsed_ms"])
            hits = response["total_hits"]

        all_times.extend(times)
        print(
            f"  {query_str:<25} "
            f"{min(times):>6.2f}ms "
            f"{statistics.mean(times):>6.2f}ms "
            f"{max(times):>6.2f}ms  "
            f"{hits}"
        )

    print(f"\n  Overall avg latency : {statistics.mean(all_times):.2f}ms")
    print(f"  Overall p95 latency : {sorted(all_times)[int(len(all_times)*0.95)]:.2f}ms")
    print(f"  Target              : <50ms  {'PASS' if statistics.mean(all_times) < 50 else 'FAIL'}\n")


def benchmark_index_stats():
    print("=" * 50)
    print("INDEX STATISTICS")
    print("=" * 50)
    index = InvertedIndex.load(INDEX_PATH)
    stats = index.stats()
    for k, v in stats.items():
        print(f"  {k:<22} : {v:,}" if isinstance(v, int) else f"  {k:<22} : {v}")

    print(f"\n  Index file size : "
          f"{INDEX_PATH.stat().st_size / 1024:.1f} KB")


if __name__ == "__main__":
    benchmark_index_operations()
    benchmark_queries()
    benchmark_index_stats()