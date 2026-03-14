import pytest
from collections import defaultdict
from pathlib import Path

from indexer.index import InvertedIndex
from indexer.bm25 import BM25
from indexer.query import search


@pytest.fixture
def small_index():
    index = InvertedIndex()
    index.add_document(
        doc_id="doc1",
        tokens=["python", "web", "crawl", "search", "python"],
        url="https://example.com/python",
        title="Python Web Crawling",
    )
    index.add_document(
        doc_id="doc2",
        tokens=["search", "engine", "index", "rank", "search", "search"],
        url="https://example.com/search",
        title="Search Engine Basics",
    )
    index.add_document(
        doc_id="doc3",
        tokens=["python", "data", "science", "pandas", "numpy"],
        url="https://example.com/data",
        title="Python Data Science",
    )
    return index


@pytest.fixture
def bm25_engine(small_index):
    return BM25(small_index)


class TestInvertedIndex:
    def test_doc_count(self, small_index):
        assert small_index.get_doc_count() == 3

    def test_vocab_contains_indexed_terms(self, small_index):
        assert small_index.contains("python")
        assert small_index.contains("search")
        assert small_index.contains("engine")

    def test_unknown_term_not_in_index(self, small_index):
        assert not small_index.contains("xyznonexistent")

    def test_term_frequency_counted_correctly(self, small_index):
        assert small_index.get_tf("python", "doc1") == 2
        assert small_index.get_tf("search", "doc2") == 3
        assert small_index.get_tf("python", "doc2") == 0

    def test_doc_frequency(self, small_index):
        assert small_index.get_doc_frequency("python") == 2
        assert small_index.get_doc_frequency("search") == 2
        assert small_index.get_doc_frequency("pandas") == 1
        assert small_index.get_doc_frequency("xyznonexistent") == 0

    def test_doc_length(self, small_index):
        assert small_index.get_doc_length("doc1") == 5
        assert small_index.get_doc_length("doc2") == 6
        assert small_index.get_doc_length("doc3") == 5

    def test_avg_doc_length(self, small_index):
        assert small_index.get_avg_doc_length() == pytest.approx(16 / 3, rel=1e-3)

    def test_postings_list_correctness(self, small_index):
        postings = small_index.get_postings("python")
        assert "doc1" in postings
        assert "doc3" in postings
        assert "doc2" not in postings

    def test_get_doc_metadata(self, small_index):
        doc = small_index.get_doc("doc1")
        assert doc["url"] == "https://example.com/python"
        assert doc["title"] == "Python Web Crawling"

    def test_unknown_doc_returns_none(self, small_index):
        assert small_index.get_doc("nonexistent") is None

    def test_stats_keys(self, small_index):
        stats = small_index.stats()
        assert "total_docs" in stats
        assert "vocab_size" in stats
        assert "avg_doc_length" in stats
        assert "total_tokens" in stats


class TestPersistence:
    def test_save_and_load_round_trip(self, small_index, tmp_path):
        path = tmp_path / "test_index.json"
        small_index.save(path)
        loaded = InvertedIndex.load(path)

        assert loaded.get_doc_count() == small_index.get_doc_count()
        assert loaded.get_vocab_size() == small_index.get_vocab_size()
        assert loaded.get_tf("python", "doc1") == small_index.get_tf("python", "doc1")
        assert loaded.get_tf("search", "doc2") == small_index.get_tf("search", "doc2")
        assert loaded.get_doc("doc1") == small_index.get_doc("doc1")

    def test_load_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            InvertedIndex.load(tmp_path / "nonexistent.json")

    def test_stats_survive_round_trip(self, small_index, tmp_path):
        path = tmp_path / "test_index.json"
        small_index.save(path)
        loaded = InvertedIndex.load(path)
        assert loaded.stats() == small_index.stats()


class TestBM25:
    def test_idf_rare_term_higher_than_common(self, bm25_engine, small_index):
        idf_python = bm25_engine.idf("python")
        idf_pandas = bm25_engine.idf("pandas")
        assert idf_pandas > idf_python

    def test_idf_unknown_term_is_zero(self, bm25_engine):
        assert bm25_engine.idf("xyznonexistent") == 0.0

    def test_tf_norm_zero_for_missing_term(self, bm25_engine):
        assert bm25_engine.tf_norm("pandas", "doc1") == 0.0

    def test_score_higher_for_more_relevant_doc(self, bm25_engine):
        tokens = ["search", "engine"]
        score_doc2 = bm25_engine.score(tokens, "doc2")
        score_doc1 = bm25_engine.score(tokens, "doc1")
        assert score_doc2 > score_doc1

    def test_search_returns_correct_count(self, bm25_engine):
        results = bm25_engine.search(["python"], top_k=2)
        assert len(results) == 2

    def test_search_top_result_most_relevant(self, bm25_engine):
        results = bm25_engine.search(["search", "engine"], top_k=3)
        assert results[0]["doc_id"] == "doc2"

    def test_search_results_sorted_by_score(self, bm25_engine):
        results = bm25_engine.search(["python"], top_k=3)
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_search_empty_tokens_returns_empty(self, bm25_engine):
        assert bm25_engine.search([]) == []

    def test_search_unknown_term_returns_empty(self, bm25_engine):
        assert bm25_engine.search(["xyznonexistent"]) == []


class TestQueryProcessor:
    def test_search_returns_dict_with_required_keys(self):
        response = search("mystery", top_k=3)
        assert "query" in response
        assert "tokens" in response
        assert "results" in response
        assert "total_hits" in response
        assert "elapsed_ms" in response

    def test_search_results_have_required_fields(self):
        response = search("mystery", top_k=3)
        if response["results"]:
            result = response["results"][0]
            assert "url" in result
            assert "title" in result
            assert "score" in result

    def test_search_elapsed_ms_is_positive(self):
        response = search("book", top_k=5)
        assert response["elapsed_ms"] > 0

    def test_search_nonexistent_query(self):
        response = search("xyznonexistent123")
        assert response["total_hits"] == 0
        assert response["results"] == []

    def test_search_or_mode_returns_more_than_and(self):
        or_response  = search("mystery travel", top_k=100, mode="OR")
        and_response = search("mystery travel", top_k=100, mode="AND")
        assert or_response["total_hits"] >= and_response["total_hits"]

    def test_search_results_ordered_by_score(self):
        response = search("book", top_k=10)
        scores = [r["score"] for r in response["results"]]
        assert scores == sorted(scores, reverse=True)