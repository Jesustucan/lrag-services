"""
Minimal reranker server — Cohere-compatible API.
LightRAG connects via RERANK_BINDING=cohere pointing to this service.
"""

import logging
from typing import List, Optional

from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import CrossEncoder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Reranker API")

MODEL_NAME = "BAAI/bge-reranker-v2-m3"
logger.info(f"Loading model {MODEL_NAME} ...")
# max_length=1024 covers chunks of 1200 tokens with query overhead included
model = CrossEncoder(MODEL_NAME, device="cpu", max_length=1024)
logger.info("Model ready")


class RerankRequest(BaseModel):
    model: str = MODEL_NAME
    query: str
    documents: List[str]
    top_n: Optional[int] = None


class RerankResponse(BaseModel):
    results: List[dict]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/rerank", response_model=RerankResponse)
def rerank(request: RerankRequest):
    pairs = [[request.query, doc] for doc in request.documents]
    scores = model.predict(pairs, show_progress_bar=False)

    results = [
        {"index": i, "relevance_score": float(score)}
        for i, score in enumerate(scores)
    ]
    results.sort(key=lambda x: x["relevance_score"], reverse=True)

    if request.top_n is not None:
        results = results[: request.top_n]

    return {"results": results}
