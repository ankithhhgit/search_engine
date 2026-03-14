import statistics
import time
from pathlib import Path

import httpx

BASE_URL = "http://localhost:8000"

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


def benchmark_endpoint(
    client: httpx.Client,
    method: str,
    url: str,
    runs: int = 20,
    label: str = "",
) -> dict:
    times = []
    errors = 0

    for _ in range(runs):
        try:
            start = time.perf_counter()
            response = client.request(method, url)
            elapsed = (time.perf_counter() - start) * 1000
            if response.status_code == 200:
                times.append(elapsed)
            else:
                errors += 1
        except Exception:
            errors += 1

    if not times:
        return {"label": label, "error": "all requests failed"}

    sorted_times = sorted(times)
    return {
        "label":   label,
        "runs":    runs,
        "errors":  errors,
        "min":     round(min(times), 2),
        "p50":     round(sorted_times[len(sorted_times) // 2], 2),
        "p95":     round(sorted_times[int(len(sorted_times) * 0.95)], 2),
        "p99":     round(sorted_times[int(len(sorted_times) * 0.99)], 2),
        "max":     round(max(times), 2),
        "mean":    round(statistics.mean(times), 2),
        "stdev":   round(statistics.stdev(times), 2) if len(times) > 1 else 0,
    }


def print_result(result: dict) -> None:
    if "error" in result:
        print(f"  {result['label']:<35} ERROR: {result['error']}")
        return
    print(
        f"  {result['label']:<35} "
        f"p50={result['p50']:>7.2f}ms  "
        f"p95={result['p95']:>7.2f}ms  "
        f"p99={result['p99']:>7.2f}ms  "
        f"mean={result['mean']:>7.2f}ms"
    )


def run_benchmarks() -> None:
    print("=" * 70)
    print("SEARCH ENGINE API BENCHMARK")
    print("=" * 70)
    print(f"Target: {BASE_URL}")
    print(f"Runs per query: 20\n")

    with httpx.Client(base_url=BASE_URL, timeout=30) as client:

        try:
            r = client.get("/health")
            r.raise_for_status()
            data = r.json()
            print(f"Server healthy — {data['total_docs']} docs, "
                  f"{data['vocab_size']} terms\n")
        except Exception as e:
            print(f"Server not reachable: {e}")
            print("Start the server first: uvicorn api.main:app --reload")
            return

        print("=" * 70)
        print("HEALTH ENDPOINT")
        print("=" * 70)
        result = benchmark_endpoint(client, "GET", "/health", runs=20, label="/health")
        print_result(result)

        print("\n" + "=" * 70)
        print("SEARCH ENDPOINT — per query")
        print("=" * 70)
        print(f"  {'query':<35} {'p50':>10}  {'p95':>10}  {'p99':>10}  {'mean':>10}")
        print(f"  {'-'*35} {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}")

        all_times = []
        for query in QUERIES:
            url = f"/search?q={query.replace(' ', '+')}&top_k=10"
            result = benchmark_endpoint(client, "GET", url, runs=20, label=query)
            print_result(result)
            if "mean" in result:
                all_times.append(result["mean"])

        print("\n" + "=" * 70)
        print("SEARCH ENDPOINT — overall")
        print("=" * 70)
        if all_times:
            print(f"  Overall mean latency  : {statistics.mean(all_times):.2f}ms")
            print(f"  Overall stdev         : {statistics.stdev(all_times):.2f}ms")
            target = 50
            status = "PASS" if statistics.mean(all_times) < target else "FAIL"
            print(f"  Target (<{target}ms)        : {status}")

        print("\n" + "=" * 70)
        print("AND vs OR MODE")
        print("=" * 70)
        test_query = "mystery travel"
        for mode in ["OR", "AND"]:
            url = f"/search?q={test_query.replace(' ', '+')}&mode={mode}&top_k=50"
            result = benchmark_endpoint(
                client, "GET", url, runs=10, label=f"{mode} — '{test_query}'"
            )
            print_result(result)

        print("\n" + "=" * 70)
        print("VALIDATION OVERHEAD")
        print("=" * 70)
        invalid_url = "/search?q=mystery&top_k=0"
        result = benchmark_endpoint(
            client, "GET", invalid_url, runs=10, label="422 invalid top_k"
        )
        valid_url = "/search?q=mystery&top_k=10"
        result2 = benchmark_endpoint(
            client, "GET", valid_url, runs=10, label="200 valid request"
        )
        print_result(result)
        print_result(result2)

    print("\n" + "=" * 70)
    print("BENCHMARK COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    run_benchmarks()