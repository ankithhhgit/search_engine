import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

from api.main import app
from api.schemas import SearchResponse, HealthResponse


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture
def mock_search_response():
    return {
        "query":      "mystery",
        "tokens":     ["mysteri"],
        "missing":    [],
        "mode":       "OR",
        "total_hits": 23,
        "elapsed_ms": 2.87,
        "results": [
            {
                "doc_id":      "abc123",
                "score":       0.8934,
                "bm25_score":  2.1834,
                "url":         "https://books.toscrape.com/mystery",
                "title":       "Mystery | Books to Scrape",
                "token_count": 312,
            },
            {
                "doc_id":      "def456",
                "score":       0.7201,
                "bm25_score":  1.9201,
                "url":         "https://books.toscrape.com/crime",
                "title":       "Crime | Books to Scrape",
                "token_count": 287,
            },
        ],
    }


class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_response_structure(self, client):
        response = client.get("/health")
        data = response.json()
        assert "status" in data
        assert "total_docs" in data
        assert "vocab_size" in data
        assert "avg_doc_length" in data
        assert "total_tokens" in data

    def test_health_status_is_ok(self, client):
        response = client.get("/health")
        assert response.json()["status"] == "ok"

    def test_health_total_docs_positive(self, client):
        response = client.get("/health")
        assert response.json()["total_docs"] > 0

    def test_health_vocab_size_positive(self, client):
        response = client.get("/health")
        assert response.json()["vocab_size"] > 0

    def test_health_has_response_time_header(self, client):
        response = client.get("/health")
        assert "x-response-time" in response.headers

    def test_health_response_time_header_format(self, client):
        response = client.get("/health")
        header = response.headers["x-response-time"]
        assert header.endswith("ms")
        value = float(header.replace("ms", ""))
        assert value > 0


class TestSearchEndpoint:
    def test_search_returns_200(self, client):
        response = client.get("/search?q=mystery")
        assert response.status_code == 200

    def test_search_response_structure(self, client):
        response = client.get("/search?q=mystery")
        data = response.json()
        assert "query" in data
        assert "tokens" in data
        assert "missing" in data
        assert "mode" in data
        assert "total_hits" in data
        assert "results" in data
        assert "elapsed_ms" in data

    def test_search_query_echoed_in_response(self, client):
        response = client.get("/search?q=mystery")
        assert response.json()["query"] == "mystery"

    def test_search_results_is_list(self, client):
        response = client.get("/search?q=mystery")
        assert isinstance(response.json()["results"], list)

    def test_search_result_fields(self, client):
        response = client.get("/search?q=mystery")
        results = response.json()["results"]
        if results:
            result = results[0]
            assert "doc_id" in result
            assert "score" in result
            assert "bm25_score" in result
            assert "url" in result
            assert "title" in result
            assert "token_count" in result

    def test_search_results_sorted_by_score(self, client):
        response = client.get("/search?q=mystery")
        results = response.json()["results"]
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_search_respects_top_k(self, client):
        response = client.get("/search?q=book&top_k=3")
        assert len(response.json()["results"]) <= 3

    def test_search_top_k_default_is_10(self, client):
        response = client.get("/search?q=book")
        assert len(response.json()["results"]) <= 10

    def test_search_or_mode(self, client):
        response = client.get("/search?q=mystery&mode=OR")
        assert response.json()["mode"] == "OR"

    def test_search_and_mode(self, client):
        response = client.get("/search?q=mystery&mode=AND")
        assert response.json()["mode"] == "AND"

    def test_search_or_returns_gte_and(self, client):
        or_hits  = client.get("/search?q=mystery+travel&mode=OR&top_k=50").json()["total_hits"]
        and_hits = client.get("/search?q=mystery+travel&mode=AND&top_k=50").json()["total_hits"]
        assert or_hits >= and_hits

    def test_search_nonexistent_query_returns_empty(self, client):
        response = client.get("/search?q=xyznonexistent123abc")
        data = response.json()
        assert data["total_hits"] == 0
        assert data["results"] == []

    def test_search_elapsed_ms_positive(self, client):
        response = client.get("/search?q=mystery")
        assert response.json()["elapsed_ms"] > 0

    def test_search_has_response_time_header(self, client):
        response = client.get("/search?q=mystery")
        assert "x-response-time" in response.headers

    def test_search_missing_tokens_reported(self, client):
        response = client.get("/search?q=xyznonexistent123")
        data = response.json()
        assert isinstance(data["missing"], list)


class TestSearchValidation:
    def test_missing_query_returns_422(self, client):
        response = client.get("/search")
        assert response.status_code == 422

    def test_top_k_zero_returns_422(self, client):
        response = client.get("/search?q=mystery&top_k=0")
        assert response.status_code == 422

    def test_top_k_negative_returns_422(self, client):
        response = client.get("/search?q=mystery&top_k=-1")
        assert response.status_code == 422

    def test_top_k_too_large_returns_422(self, client):
        response = client.get("/search?q=mystery&top_k=999")
        assert response.status_code == 422

    def test_top_k_max_boundary_valid(self, client):
        response = client.get("/search?q=mystery&top_k=50")
        assert response.status_code == 200

    def test_top_k_min_boundary_valid(self, client):
        response = client.get("/search?q=mystery&top_k=1")
        assert response.status_code == 200

    def test_invalid_mode_returns_422(self, client):
        response = client.get("/search?q=mystery&mode=INVALID")
        assert response.status_code == 422

    def test_mode_case_sensitive(self, client):
        response = client.get("/search?q=mystery&mode=or")
        assert response.status_code == 422

    def test_422_has_detail_field(self, client):
        response = client.get("/search?q=mystery&top_k=0")
        assert "detail" in response.json()


class TestResponseContract:
    def test_scores_are_floats(self, client):
        response = client.get("/search?q=mystery")
        for result in response.json()["results"]:
            assert isinstance(result["score"], float)
            assert isinstance(result["bm25_score"], float)

    def test_scores_between_zero_and_one(self, client):
        response = client.get("/search?q=mystery")
        for result in response.json()["results"]:
            assert 0.0 <= result["score"] <= 1.0

    def test_urls_are_strings(self, client):
        response = client.get("/search?q=mystery")
        for result in response.json()["results"]:
            assert isinstance(result["url"], str)
            assert result["url"].startswith("http")

    def test_token_count_positive(self, client):
        response = client.get("/search?q=mystery")
        for result in response.json()["results"]:
            assert result["token_count"] > 0

    def test_total_hits_matches_scored_count(self, client):
        response = client.get("/search?q=book&top_k=50")
        data = response.json()
        assert data["total_hits"] >= len(data["results"])

    def test_content_type_is_json(self, client):
        response = client.get("/search?q=mystery")
        assert "application/json" in response.headers["content-type"]