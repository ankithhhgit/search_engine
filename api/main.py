from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from indexer.index import InvertedIndex
from indexer.bm25 import BM25
from indexer.query import search
from pathlib import Path

INDEX_PATH = Path("data/index.json")

state: dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading index...")
    state["index"] = InvertedIndex.load(INDEX_PATH)
    state["bm25"]  = BM25(state["index"])
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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    if "index" not in state:
        raise HTTPException(status_code=503, detail="Index not loaded")

    index = state["index"]
    stats = index.stats()
    return {
        "status":       "ok",
        "total_docs":   stats["total_docs"],
        "vocab_size":   stats["vocab_size"],
        "avg_doc_length": stats["avg_doc_length"],
        "total_tokens": stats["total_tokens"],
    }


@app.get("/search")
def search_endpoint(
    q: str = Query(..., min_length=1, description="Search query"),
    top_k: int = Query(10, ge=1, le=50, description="Number of results to return"),
    mode: str = Query("OR", pattern="^(OR|AND)$", description="OR or AND query mode"),
):
    if not q.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    response = search(q, top_k=top_k, mode=mode)
    return response