import json
from pathlib import Path

CLEAN_DATA_DIR = Path("data/clean")
PAGERANK_PATH  = Path("data/pagerank.json")

DAMPING    = 0.85
MAX_ITER   = 100
CONVERGENCE = 1e-6


def build_graph(clean_dir: Path = CLEAN_DATA_DIR) -> tuple[dict, dict]:
    files = list(clean_dir.glob("*.json"))
    if not files:
        raise FileNotFoundError("No clean files found. Run the crawler first.")

    url_to_id: dict[str, str] = {}
    outlinks:  dict[str, list[str]] = {}

    for filepath in files:
        with open(filepath, encoding="utf-8") as f:
            page = json.load(f)
        url = page["url"]
        url_to_id[url] = filepath.stem
        outlinks[url]  = page.get("links", [])

    known_urls = set(url_to_id.keys())
    filtered: dict[str, list[str]] = {}
    for url, links in outlinks.items():
        filtered[url] = [l for l in links if l in known_urls]

    return url_to_id, filtered


def compute_pagerank(
    outlinks: dict[str, list[str]],
    damping: float = DAMPING,
    max_iter: int = MAX_ITER,
    convergence: float = CONVERGENCE,
) -> dict[str, float]:
    urls  = list(outlinks.keys())
    N     = len(urls)
    base  = (1 - damping) / N

    scores: dict[str, float] = {url: 1.0 / N for url in urls}

    inlinks: dict[str, list[str]] = {url: [] for url in urls}
    for src, targets in outlinks.items():
        for tgt in targets:
            if tgt in inlinks:
                inlinks[tgt].append(src)

    out_degree: dict[str, int] = {url: len(links) for url, links in outlinks.items()}

    for iteration in range(1, max_iter + 1):
        new_scores: dict[str, float] = {}

        for url in urls:
            incoming = sum(
                scores[src] / out_degree[src]
                for src in inlinks[url]
                if out_degree[src] > 0
            )
            new_scores[url] = base + damping * incoming

        dangling_mass = sum(
            scores[url] / N
            for url in urls
            if out_degree[url] == 0
        ) * damping
        new_scores = {url: s + dangling_mass for url, s in new_scores.items()}

        delta = sum(abs(new_scores[u] - scores[u]) for u in urls)
        scores = new_scores

        print(f"  Iteration {iteration:>3} — delta: {delta:.8f}")
        if delta < convergence:
            print(f"  Converged after {iteration} iterations.")
            break

    return scores


def normalise(scores: dict[str, float]) -> dict[str, float]:
    min_s = min(scores.values())
    max_s = max(scores.values())
    spread = max_s - min_s
    if spread == 0:
        return {url: 0.0 for url in scores}
    return {url: (s - min_s) / spread for url, s in scores.items()}


def save_pagerank(scores: dict[str, float], path: Path = PAGERANK_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(scores, f, indent=2, ensure_ascii=False)
    print(f"PageRank saved to {path} ({len(scores)} URLs)")


def load_pagerank(path: Path = PAGERANK_PATH) -> dict[str, float]:
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def get_score(url: str, scores: dict[str, float]) -> float:
    return scores.get(url, 0.0)


if __name__ == "__main__":
    print("=" * 50)
    print("PHASE 1 — building link graph")
    print("=" * 50)
    url_to_id, outlinks = build_graph()

    total_links = sum(len(v) for v in outlinks.values())
    avg_links   = total_links / len(outlinks) if outlinks else 0
    dangling    = sum(1 for v in outlinks.values() if len(v) == 0)

    print(f"  Pages     : {len(outlinks)}")
    print(f"  Total links: {total_links}")
    print(f"  Avg links  : {avg_links:.1f} per page")
    print(f"  Dangling   : {dangling} pages with no outlinks\n")

    print("=" * 50)
    print("PHASE 2 — computing PageRank")
    print("=" * 50)
    raw_scores = compute_pagerank(outlinks)

    print("\n=" * 50)
    print("PHASE 3 — normalising scores")
    print("=" * 50)
    norm_scores = normalise(raw_scores)

    print("\n=" * 50)
    print("TOP 10 PAGES BY PAGERANK")
    print("=" * 50)
    ranked = sorted(norm_scores.items(), key=lambda x: x[1], reverse=True)
    for i, (url, score) in enumerate(ranked[:10], 1):
        short_url = url.replace("https://books.toscrape.com", "")
        print(f"  {i:>2}. [{score:.4f}]  {short_url or '/'}")

    print("\n" + "=" * 50)
    print("BOTTOM 5 PAGES BY PAGERANK")
    print("=" * 50)
    for url, score in ranked[-5:]:
        short_url = url.replace("https://books.toscrape.com", "")
        print(f"      [{score:.4f}]  {short_url}")

    print("\n" + "=" * 50)
    print("PHASE 4 — saving")
    print("=" * 50)
    save_pagerank(norm_scores)
    print("\nDone.")