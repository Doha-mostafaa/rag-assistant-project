import os
import requests
from dotenv import load_dotenv

load_dotenv()

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")


def check_health() -> bool:
    """Check if the backend API is healthy."""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.status_code == 200
    except requests.ConnectionError:
        return False


def query_documents(question: str) -> dict:
    """Send a question to the RAG backend and return the response."""
    response = requests.post(
        f"{API_BASE_URL}/query",
        json={"question": question},
        timeout=120,
    )
    response.raise_for_status()
    return response.json()
