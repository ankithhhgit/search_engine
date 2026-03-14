import time
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Query, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path

from api.schemas import HealthResponse, SearchResponse, ErrorResponse
from indexer.index import InvertedIndex
from indexer.bm25 import BM25
from indexer.query import search as engine_search
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

INDEX_PATH = Path("data/index.json")

state: dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading index...")
    state["index"] = InvertedIndex.load(INDEX_PATH)
    state["bm25"]  = BM25(state["index"])
    state["start_time"] = time.time()
    print(f"Index ready — {state['index'].get_doc_count()} docs loaded")
    yield
    state.clear()
    print("Server shutting down — index cleared")


app = FastAPI(
    title="Search Engine API",
    description="A BM25-powered search engine built from scratch",
    version="1.0.0",
    lifespan=lifespan,
)
app.mount("/static", StaticFiles(directory="frontend"), name="static")

@app.get("/", include_in_schema=False)
def serve_frontend():
    return FileResponse("frontend/index.html")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_response_time_header(request: Request, call_next) -> Response:
    start = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = round((time.perf_counter() - start) * 1000, 2)
    response.headers["X-Response-Time"] = f"{elapsed_ms}ms"
    return response


from indexer.database import get_stats as db_stats, populate as db_populate

# replace the health endpoint with this:
@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Server health and index statistics",
)
def health():
    if "index" not in state:
        raise HTTPException(status_code=503, detail="Index not loaded")

    index  = state["index"]
    stats  = index.stats()
    db     = db_stats()

    return HealthResponse(
        status="ok",
        total_docs=stats["total_docs"],
        vocab_size=stats["vocab_size"],
        avg_doc_length=stats["avg_doc_length"],
        total_tokens=stats["total_tokens"],
    )

@app.get(
    "/search",
    response_model=SearchResponse,
    summary="Search indexed documents",
    responses={
        400: {"model": ErrorResponse, "description": "Empty query"},
        503: {"model": ErrorResponse, "description": "Index not ready"},
    },
)
def search_endpoint(
    q: str     = Query(..., min_length=1, description="Search query string"),
    top_k: int = Query(10, ge=1, le=50,  description="Max results to return"),
    mode: str  = Query("OR", pattern="^(OR|AND)$", description="OR or AND logic"),
):
    if "index" not in state:
        raise HTTPException(status_code=503, detail="Index not loaded yet")

    if not q.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    raw = engine_search(q, top_k=top_k, mode=mode)

    return SearchResponse(
        query=raw["query"],
        tokens=raw["tokens"],
        missing=raw["missing"],
        mode=raw["mode"],
        total_hits=raw["total_hits"],
        elapsed_ms=raw["elapsed_ms"],
        results=raw["results"],
    )
@app.get("/suggest", include_in_schema=False)
def suggest(q: str = Query("", min_length=1)):
    if "index" not in state or not q.strip():
        return {"suggestions": []}

    index  = state["index"]
    prefix = q.lower().strip()
    matches = [
        term for term in index._index.keys()
        if term.startswith(prefix)
    ]
    matches.sort(key=lambda t: (-index.get_doc_frequency(t), t))
    return {"suggestions": matches[:8]}

from indexer.query import get_cache_stats, clear_cache

@app.get("/cache/stats", summary="Query cache statistics")
def cache_stats():
    return get_cache_stats()


@app.get("/cache/clear", summary="Clear the query cache")
def cache_clear():
    clear_cache()
    return {"status": "ok", "message": "Cache cleared"}