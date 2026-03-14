import json
from collections import defaultdict
from pathlib import Path

CLEAN_DATA_DIR = Path("data/clean")


class InvertedIndex:
    def __init__(self):
        # term -> {doc_id -> term_frequency}
        self._index: dict[str, dict[str, int]] = defaultdict(dict)

        # doc_id -> {url, title, token_count}
        self._docs: dict[str, dict] = {}

        self._total_tokens: int = 0

    def add_document(self, doc_id: str, tokens: list[str], url: str, title: str) -> None:
        token_count = len(tokens)
        self._docs[doc_id] = {
            "url": url,
            "title": title,
            "token_count": token_count,
        }
        self._total_tokens += token_count

        term_freq = defaultdict(int)
        for token in tokens:
            term_freq[token] += 1

        for term, freq in term_freq.items():
            self._index[term][doc_id] = freq

    def get_postings(self, term: str) -> dict[str, int]:
        return self._index.get(term, {})

    def get_doc(self, doc_id: str) -> dict | None:
        return self._docs.get(doc_id)

    def get_doc_length(self, doc_id: str) -> int:
        doc = self._docs.get(doc_id)
        return doc["token_count"] if doc else 0

    def get_avg_doc_length(self) -> float:
        if not self._docs:
            return 0.0
        return self._total_tokens / len(self._docs)

    def get_doc_count(self) -> int:
        return len(self._docs)

    def get_vocab_size(self) -> int:
        return len(self._index)

    def get_doc_frequency(self, term: str) -> int:
        return len(self._index.get(term, {}))
    
    def get_tf(self, term: str, doc_id: str) -> int:
        return self._index.get(term, {}).get(doc_id, 0)

    def get_all_docs_with_term(self, term: str) -> list[str]:
        return list(self._index.get(term, {}).keys())

    def get_all_doc_ids(self) -> list[str]:
        return list(self._docs.keys())

    def contains(self, term: str) -> bool:
        return term in self._index

    def stats(self) -> dict:
        return {
            "total_docs": self.get_doc_count(),
            "vocab_size": self.get_vocab_size(),
            "total_tokens": self._total_tokens,
            "avg_doc_length": round(self.get_avg_doc_length(), 2),
        }

    @classmethod
    def build_from_disk(cls, clean_dir: Path = CLEAN_DATA_DIR) -> "InvertedIndex":
        index = cls()
        files = list(clean_dir.glob("*.json"))

        if not files:
            print("No clean files found. Run the crawler first.")
            return index

        print(f"Building index from {len(files)} documents...")
        for filepath in files:
            with open(filepath, encoding="utf-8") as f:
                page = json.load(f)

            index.add_document(
                doc_id=filepath.stem,
                tokens=page["tokens"],
                url=page["url"],
                title=page["title"],
            )

        print(f"Index built. Stats: {index.stats()}")
        return index


if __name__ == "__main__":
    import time

    start = time.perf_counter()
    index = InvertedIndex.build_from_disk()
    build_time = time.perf_counter() - start

    print(f"\nBuild time: {build_time:.3f}s\n")

    print("=== Corpus statistics (BM25 inputs) ===")
    stats = index.stats()
    print(f"  N  (total docs)      : {stats['total_docs']:>8,}")
    print(f"  V  (vocab size)      : {stats['vocab_size']:>8,}")
    print(f"  avgdl (avg doc len)  : {stats['avg_doc_length']:>8,.2f}")
    print(f"  total tokens         : {stats['total_tokens']:>8,}")

    print("\n=== Per-term statistics ===")
    test_terms = ["book", "price", "mysteri", "travel", "light"]
    print(f"  {'term':<12} {'df':>6} {'idf_raw':>10}")
    print(f"  {'-'*12} {'-'*6} {'-'*10}")

    import math
    N = stats["total_docs"]
    for term in test_terms:
        df = index.get_doc_frequency(term)
        if df > 0:
            idf = math.log((N - df + 0.5) / (df + 0.5) + 1)
        else:
            idf = 0.0
        print(f"  {term:<12} {df:>6,} {idf:>10.4f}")

    print("\n=== Per-doc statistics (sample 5 docs) ===")
    print(f"  {'doc_id':<14} {'length':>8} {'vs avg':>10}  title")
    print(f"  {'-'*14} {'-'*8} {'-'*10}  {'-'*30}")
    avgdl = stats["avg_doc_length"]
    for doc_id in index.get_all_doc_ids()[:5]:
        doc = index.get_doc(doc_id)
        length = index.get_doc_length(doc_id)
        ratio = length / avgdl if avgdl else 0
        print(f"  {doc_id[:12]:<14} {length:>8,} {ratio:>9.2f}x  {doc['title'][:35]}")

    print("\n=== TF spot check ===")
    term = "book"
    postings = index.get_postings(term)
    top5 = sorted(postings.items(), key=lambda x: x[1], reverse=True)[:5]
    print(f"\n  Top 5 docs by TF for '{term}':")
    print(f"  {'doc_id':<14} {'tf':>4}  title")
    print(f"  {'-'*14} {'-'*4}  {'-'*35}")
    for doc_id, tf in top5:
        doc = index.get_doc(doc_id)
        print(f"  {doc_id[:12]:<14} {tf:>4}  {doc['title'][:35]}")

    print("\n=== get_tf() helper check ===")
    if top5:
        sample_doc_id, expected_tf = top5[0]
        retrieved_tf = index.get_tf(term, sample_doc_id)
        match = "PASS" if retrieved_tf == expected_tf else "FAIL"
        print(f"  get_tf('{term}', '{sample_doc_id[:12]}...')")
        print(f"  Expected: {expected_tf}  Got: {retrieved_tf}  [{match}]")

    print("\n=== All BM25 inputs verified ===")
    print("  N        -> get_doc_count()")
    print("  df(t)    -> get_doc_frequency(term)")
    print("  tf(t,d)  -> get_tf(term, doc_id)")
    print("  len(d)   -> get_doc_length(doc_id)")
    print("  avgdl    -> get_avg_doc_length()")
    print("\nReady for Wednesday — BM25 implementation.")