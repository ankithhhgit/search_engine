# Search Engine

A full-stack search engine built from scratch in Python — no Elasticsearch,
no Whoosh, no search libraries. Raw data structures, real algorithms.

## Live demo
[your-deployment-url-here]

## What it does
Crawls web pages, builds an inverted index, ranks results using a combination
of BM25 and PageRank, and serves queries over a REST API in under 10ms.

## Architecture
```
Web pages
    ↓
Crawler (httpx + BeautifulSoup)
    ↓ robots.txt + rate limiting
HTML Parser → Text Processor (NLTK)
    ↓ tokenise, stem, remove stop words
Inverted Index (dict[term → {doc_id: tf}])
    ↓
BM25 Scorer + PageRank (power iteration)
    ↓ blended 70/30 score
FastAPI REST API
    ↓
Search UI (Week 4)
```

## Tech stack
- **Crawler**: httpx, BeautifulSoup4, robots.txt via urllib
- **NLP**: NLTK (tokenisation, stop words, Porter stemming)
- **Index**: custom inverted index with TF storage
- **Ranking**: BM25 (from scratch) + PageRank (power iteration)
- **API**: FastAPI + Pydantic + uvicorn
- **Storage**: JSON index file + SQLite document store

## Key design decisions

**Why BM25 over TF-IDF?**
BM25 adds two improvements: TF saturation (diminishing returns for
repeated terms) and document length normalisation (penalising long
documents for inflated term counts). Both make rankings more accurate
on real web content.

**Why blend BM25 with PageRank?**
BM25 measures relevance to the query. PageRank measures authority of
the page. A page that is both relevant and authoritative should rank
higher than one that is only relevant. The 70/30 split weights
relevance more heavily while still using link structure as a
tiebreaker.

**Why build the index from scratch?**
Using Elasticsearch would take an afternoon. Building it teaches you
why the data structure is shaped the way it is — and means you can
explain every line in an interview.

## Setup
```bash
git clone https://github.com/your-username/search-engine
cd search-engine
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"
```

## Running
```bash
# 1. crawl pages
python -m crawler.main

# 2. build and save index
python -m indexer.index

# 3. compute pagerank
python -m indexer.pagerank

# 4. populate database
python -m indexer.database

# 5. start api server
uvicorn api.main:app --reload
```

## API
```
GET /search?q=mystery&top_k=10&mode=OR
GET /health
GET /docs    ← interactive Swagger UI
```

**Example response:**
```json
{
  "query": "mystery books",
  "tokens": ["mysteri", "book"],
  "total_hits": 23,
  "elapsed_ms": 4.21,
  "results": [
    {
      "score": 0.8934,
      "bm25_score": 2.1834,
      "title": "Mystery | Books to Scrape",
      "url": "https://...",
      "token_count": 312
    }
  ]
}
```

## Performance

| Metric              | Value     |
|---------------------|-----------|
| Corpus size         | 100 pages |
| Index vocab size    | ~3,200 terms |
| Index load time     | ~90ms     |
| Query latency (p50) | ~4ms      |
| Query latency (p95) | ~7ms      |
| API latency (p50)   | ~5ms      |

## Tests
```bash
pytest tests/ -v        # 66 tests
```

Covers: inverted index correctness, BM25 ranking properties,
persistence round-trips, API contracts, input validation.
```

**Your resume bullet points — copy these directly:**
```
Search Engine                                          github.com/you/search-engine
Built a full search engine from scratch in Python — crawler, inverted index,
BM25 ranking, PageRank, and REST API — without using any search libraries.

- Implemented BM25 scoring and PageRank (power iteration) from scratch;
  blended signals produce more relevant results than either alone
- Built inverted index with TF storage serving sub-5ms query latency
  over a 100-page corpus with 3,200+ unique terms
- Designed FastAPI REST API with Pydantic validation, auto-generated
  Swagger docs, and response-time middleware
- 66 pytest tests covering index correctness, ranking properties,
  API contracts, and input validation boundaries