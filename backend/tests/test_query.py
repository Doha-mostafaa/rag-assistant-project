from unittest.mock import patch

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_check():
    """Test GET /health returns 200 with status ok."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_query_invalid_input():
    """Test POST /query with missing 'question' field returns 422."""
    response = client.post("/query", json={"wrong_field": "test"})
    assert response.status_code == 422


@patch("app.api.routes.query.generate_answer")
@patch("app.api.routes.query.retrieve_relevant_chunks")
def test_query_happy_path(mock_retrieve, mock_generate):
    """Test POST /query with valid question returns answer and sources."""
    mock_retrieve.return_value = [
        {
            "text": "A tensor is a multi-dimensional array in PyTorch.",
            "source": "Tensors — PyTorch Tutorials.pdf",
            "distance": 0.25,
        }
    ]
    mock_generate.return_value = (
        "A tensor is a multi-dimensional array used in PyTorch for computation.",
        ["Tensors — PyTorch Tutorials.pdf"],
    )

    response = client.post("/query", json={"question": "What is a tensor?"})
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert "sources" in data
    assert isinstance(data["answer"], str)
    assert isinstance(data["sources"], list)
    assert len(data["answer"]) > 0
    assert len(data["sources"]) > 0


@patch("app.api.routes.query.generate_answer")
@patch("app.api.routes.query.retrieve_relevant_chunks")
def test_query_empty_question(mock_retrieve, mock_generate):
    """Test POST /query with an empty string question still returns 200."""
    mock_retrieve.return_value = [
        {
            "text": "PyTorch is a deep learning framework.",
            "source": "Tensors — PyTorch Tutorials.pdf",
            "distance": 0.90,
        }
    ]
    mock_generate.return_value = (
        "I need more context to answer your question.",
        ["Tensors — PyTorch Tutorials.pdf"],
    )

    response = client.post("/query", json={"question": ""})
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert "sources" in data