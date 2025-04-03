import pytest
from fastapi.testclient import TestClient
import os

try:
    from main import app  # Import FastAPI app
except ModuleNotFoundError:
    import sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from main import app

client = TestClient(app)

def test_docs_available():
    """Test if FastAPI docs are accessible."""
    response = client.get("/docs")
    assert response.status_code == 200

def test_query_pdf():
    """Test the PDF query endpoint with a sample question."""
    files = {"file": ("sample.pdf", b"Fake PDF content", "application/pdf")}
    params = {"query": "What is AI?"}
    
    response = client.post("/query_pdf/", files=files, params=params)
    
    assert response.status_code == 200
    assert "answer" in response.json()
