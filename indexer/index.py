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
        
    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "index": {
                term: postings
                for term, postings in self._index.items()
            },
            "docs": self._docs,
            "total_tokens": self._total_tokens,
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)

        size_kb = path.stat().st_size / 1024
        print(f"Index saved to {path} ({size_kb:.1f} KB)")

    @classmethod
    def load(cls, path: Path) -> "InvertedIndex":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"No index found at {path}. Run the crawler first.")

        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        index = cls()
        index._index = defaultdict(dict, {
            term: postings
            for term, postings in data["index"].items()
        })
        index._docs = data["docs"]
        index._total_tokens = data["total_tokens"]

        print(f"Index loaded from {path} — {index.get_doc_count()} docs, {index.get_vocab_size()} terms")
        return index

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

    INDEX_PATH = Path("data/index.json")

    print("=" * 50)
    print("PHASE 1 — build from disk")
    print("=" * 50)
    start = time.perf_counter()
    index = InvertedIndex.build_from_disk()
    build_time = time.perf_counter() - start
    print(f"Build time : {build_time:.3f}s")
    print(f"Stats      : {index.stats()}\n")

    print("=" * 50)
    print("PHASE 2 — save to disk")
    print("=" * 50)
    start = time.perf_counter()
    index.save(INDEX_PATH)
    save_time = time.perf_counter() - start
    print(f"Save time  : {save_time:.3f}s\n")

    print("=" * 50)
    print("PHASE 3 — load from disk")
    print("=" * 50)
    start = time.perf_counter()
    loaded = InvertedIndex.load(INDEX_PATH)
    load_time = time.perf_counter() - start
    print(f"Load time  : {load_time:.3f}s\n")

    print("=" * 50)
    print("PHASE 4 — round-trip verification")
    print("=" * 50)

    test_cases = [
        ("book",    list(index.get_postings("book").keys())[:1]),
        ("mysteri", list(index.get_postings("mysteri").keys())[:1]),
        ("price",   list(index.get_postings("price").keys())[:1]),
    ]

    all_passed = True
    for term, doc_ids in test_cases:
        if not doc_ids:
            continue
        doc_id = doc_ids[0]

        original_tf  = index.get_tf(term, doc_id)
        loaded_tf    = loaded.get_tf(term, doc_id)
        original_doc = index.get_doc(doc_id)
        loaded_doc   = loaded.get_doc(doc_id)

        tf_match  = original_tf == loaded_tf
        doc_match = original_doc == loaded_doc
        status    = "PASS" if tf_match and doc_match else "FAIL"
        if not tf_match or not doc_match:
            all_passed = False

        print(f"  [{status}] term='{term}' doc='{doc_id[:10]}...'")
        print(f"         tf: {original_tf} → {loaded_tf}  |  "
              f"title: '{original_doc['title'][:30]}'")

    print()
    if all_passed:
        print("  All round-trip checks passed.")
    else:
        print("  Some checks FAILED — inspect the diffs above.")

    print()
    print("=" * 50)
    print("PERFORMANCE SUMMARY")
    print("=" * 50)
    print(f"  Build from 100 clean files : {build_time:.3f}s")
    print(f"  Save to index.json         : {save_time:.3f}s")
    print(f"  Load from index.json       : {load_time:.3f}s")
    print(f"  Speedup (build vs load)    : {build_time / load_time:.1f}x faster")