import logging

from fastapi import APIRouter, HTTPException

from app.schemas.query import QueryRequest, QueryResponse
from app.services.retrieval import retrieve_relevant_chunks
from app.services.generation import generate_answer

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/health")
def health_check():
    """Check if the API is running."""
    return {"status": "ok"}


@router.post("/query", response_model=QueryResponse)
def query_documents(request: QueryRequest):
    """Retrieve relevant chunks and generate a grounded answer."""
    try:
        retrieved_chunks = retrieve_relevant_chunks(request.question)
    except RuntimeError as e:
        logger.error("Retrieval failed: %s", e)
        raise HTTPException(status_code=503, detail="Vector store not available.") from e
    except Exception as e:
        logger.error("Unexpected retrieval error: %s", e)
        raise HTTPException(status_code=500, detail="Retrieval failed.") from e

    try:
        answer, sources = generate_answer(request.question, retrieved_chunks)
    except RuntimeError as e:
        logger.error("Generation failed: %s", e)
        raise HTTPException(
            status_code=503,
            detail="LLM service unavailable. Ensure Ollama is running.",
        ) from e
    except Exception as e:
        logger.error("Unexpected generation error: %s", e)
        raise HTTPException(status_code=500, detail="Answer generation failed.") from e

    return QueryResponse(answer=answer, sources=sources)