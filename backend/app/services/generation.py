import logging
import ollama
from app.core.config import settings

logger = logging.getLogger(__name__)


def build_rag_prompt(query: str, retrieved_chunks: list) -> str:
    context_blocks = []
    for i, chunk in enumerate(retrieved_chunks):
        context_blocks.append(f"[Source {i+1}: {chunk['source']}]\n{chunk['text']}")
    
    context = "\n\n".join(context_blocks)
    
    prompt = f"""You are a helpful assistant answering questions about PyTorch
documentation. Use ONLY the context below to answer the question. If the
context doesn't contain enough information to answer, say so clearly instead
of making up an answer.

Do NOT add any "Source:" line, citation, filename, or reference at the end of
your answer — that will be added separately by the system. Just answer the
question directly using the context.

Context:
{context}

Question: {query}

Answer:"""
    
    return prompt


def generate_answer(query: str, retrieved_chunks: list) -> tuple[str, list[str]]:
    prompt = build_rag_prompt(query, retrieved_chunks)
    
    logger.info("Calling Ollama model '%s' for query: %s", settings.llm_model, query[:80])
    
    try:
        response = ollama.chat(
            model=settings.llm_model,
            messages=[{"role": "user", "content": prompt}],
        )
    except Exception as e:
        logger.error("Ollama call failed: %s", e)
        raise RuntimeError(
            f"Failed to get response from Ollama ({settings.llm_model}). "
            f"Ensure Ollama is running and the model is pulled. Error: {e}"
        ) from e
    
    raw_answer = response["message"]["content"].strip()
    unique_sources = list(dict.fromkeys(chunk["source"] for chunk in retrieved_chunks))
    
    logger.info("Generated answer (%d chars) with %d sources", len(raw_answer), len(unique_sources))
    return raw_answer, unique_sources