import math

from indexer.index import InvertedIndex


class BM25:
    def __init__(self, index: InvertedIndex, k1: float = 1.5, b: float = 0.75):
        self._index = index
        self.k1 = k1
        self.b = b
        self._N = index.get_doc_count()
        self._avgdl = index.get_avg_doc_length()

    def idf(self, term: str) -> float:
        df = self._index.get_doc_frequency(term)
        if df == 0:
            return 0.0
        return math.log((self._N - df + 0.5) / (df + 0.5) + 1)

    def tf_norm(self, term: str, doc_id: str) -> float:
        tf = self._index.get_tf(term, doc_id)
        if tf == 0:
            return 0.0
        dl = self._index.get_doc_length(doc_id)
        normaliser = tf + self.k1 * (1 - self.b + self.b * dl / self._avgdl)
        return (tf * (self.k1 + 1)) / normaliser

    def score(self, query_tokens: list[str], doc_id: str) -> float:
        return sum(self.idf(term) * self.tf_norm(term, doc_id) for term in query_tokens)

    def search(self, query_tokens: list[str], top_k: int = 10) -> list[dict]:
        if not query_tokens:
            return []

        candidates = self._get_candidates(query_tokens)
        if not candidates:
            return []

        scored = [
            {
                "doc_id": doc_id,
                "score": round(self.score(query_tokens, doc_id), 4),
                **self._index.get_doc(doc_id),
            }
            for doc_id in candidates
        ]

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]

    def _get_candidates(self, query_tokens: list[str]) -> set[str]:
        candidate_sets = [
            set(self._index.get_all_docs_with_term(term))
            for term in query_tokens
            if self._index.contains(term)
        ]
        if not candidate_sets:
            return set()

        # OR logic: any doc containing at least one query term
        return set.union(*candidate_sets)


if __name__ == "__main__":
    from indexer.text_processor import process

    print("Loading index...")
    index = InvertedIndex.build_from_disk()
    bm25 = BM25(index)

    print(f"Index loaded: {index.get_doc_count()} docs\n")

    print("=== IDF spot check ===")
    print("(rare terms should have higher IDF than common terms)\n")
    terms = ["book", "price", "mysteri", "travel", "light", "phantom"]
    print(f"  {'term':<12} {'df':>6}  {'idf':>8}")
    print(f"  {'-'*12} {'-'*6}  {'-'*8}")
    for term in terms:
        df = index.get_doc_frequency(term)
        idf = bm25.idf(term)
        print(f"  {term:<12} {df:>6}  {idf:>8.4f}")

    print("\n=== Search results ===\n")
    test_queries = [
        "mystery books",
        "travel adventure",
        "light fantasy",
        "price cheap",
    ]

    for query_str in test_queries:
        tokens = process(query_str)
        results = bm25.search(tokens, top_k=3)

        print(f"  Query : '{query_str}'")
        print(f"  Tokens: {tokens}")
        if results:
            for i, r in enumerate(results, 1):
                print(f"    {i}. [{r['score']:.4f}] {r['title'][:55]}")
        else:
            print("    No results found")
        print()

    print("=== Score breakdown for top result ===\n")
    query_str = "mystery books"
    tokens = process(query_str)
    results = bm25.search(tokens, top_k=1)

    if results:
        top = results[0]
        print(f"  Query : '{query_str}'  →  tokens: {tokens}")
        print(f"  Doc   : {top['title']}")
        print(f"  Total score: {top['score']}\n")
        print(f"  {'term':<12} {'idf':>8}  {'tf_norm':>8}  {'contrib':>8}")
        print(f"  {'-'*12} {'-'*8}  {'-'*8}  {'-'*8}")
        for term in tokens:
            idf = bm25.idf(term)
            tf_n = bm25.tf_norm(term, top["doc_id"])
            contrib = idf * tf_n
            print(f"  {term:<12} {idf:>8.4f}  {tf_n:>8.4f}  {contrib:>8.4f}")