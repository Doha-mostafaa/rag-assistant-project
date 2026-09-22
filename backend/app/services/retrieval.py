import logging
import os

# Force HuggingFace Hub to use only local cache — avoids SSL/network delays
os.environ["HF_HUB_OFFLINE"] = "1"

import chromadb
from sentence_transformers import SentenceTransformer

from app.core.config import settings

logger = logging.getLogger(__name__)

_embedding_model = None
_collection = None


def load_vector_store():
    global _embedding_model, _collection
    
    logger.info("Loading embedding model: %s", settings.embedding_model)
    _embedding_model = SentenceTransformer(settings.embedding_model)
    
    logger.info("Loading vector store from: %s", settings.vector_store_path)
    chroma_client = chromadb.PersistentClient(path=settings.vector_store_path)
    _collection = chroma_client.get_collection(name=settings.collection_name)
    logger.info(
        "Vector store loaded: %d documents in collection '%s'",
        _collection.count(),
        settings.collection_name,
    )


def retrieve_relevant_chunks(query: str, n_results: int = None):
    if _embedding_model is None or _collection is None:
        raise RuntimeError("Vector store not loaded. Call load_vector_store() at startup.")
    
    n_results = n_results or settings.retrieval_n_results
    query_embedding = _embedding_model.encode([query]).tolist()
    
    results = _collection.query(
        query_embeddings=query_embedding,
        n_results=n_results,
    )
    
    retrieved = []
    for i in range(len(results["ids"][0])):
        retrieved.append({
            "text": results["documents"][0][i],
            "source": results["metadatas"][0][i]["source"],
            "distance": results["distances"][0][i],
        })
    
    logger.debug("Retrieved %d chunks for query: %s", len(retrieved), query[:80])
    return retrieved