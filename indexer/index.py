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
    index = InvertedIndex.build_from_disk()

    print("\n=== Index stats ===")
    stats = index.stats()
    for key, value in stats.items():
        print(f"  {key:<20} : {value:,}" if isinstance(value, int) else f"  {key:<20} : {value}")

    print("\n=== Sample queries ===")
    test_terms = ["book", "price", "travel", "mysteri", "light"]
    for term in test_terms:
        postings = index.get_postings(term)
        df = index.get_doc_frequency(term)
        print(f"\n  term: '{term}'")
        print(f"  appears in {df} docs")
        if postings:
            top = sorted(postings.items(), key=lambda x: x[1], reverse=True)[:3]
            for doc_id, tf in top:
                doc = index.get_doc(doc_id)
                print(f"    tf={tf}  |  {doc['title'][:50]}")

    print("\n=== Postings list structure ===")
    sample_term = "book"
    postings = index.get_postings(sample_term)
    print(f"\n  index['{sample_term}'] = {{")
    for doc_id, tf in list(postings.items())[:4]:
        doc = index.get_doc(doc_id)
        print(f"    '{doc_id[:12]}...': {tf},   # {doc['title'][:35]}")
    print("    ...}")