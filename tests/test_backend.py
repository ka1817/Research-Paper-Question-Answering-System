import os
import pytest
from fastapi.testclient import TestClient
from main import app  

# Ensure GROQ_API_KEY is set before running tests
if not os.getenv("GROQ_API_KEY"):
    pytest.skip("Skipping tests: GROQ_API_KEY is not set", allow_module_level=True)

client = TestClient(app)

@pytest.fixture(scope="module")
def test_client():
    """Fixture to provide a test client for FastAPI."""
    yield client

def test_health_check(test_client):
    """Test if the backend is running and health check endpoint is available."""
    response = test_client.get("/")
    assert response.status_code == 200, "Health check failed!"
    assert response.json() == {"message": "Backend is running"}

def test_query_pdf(test_client):
    """Test the PDF query endpoint with a sample question."""
    files = {"file": ("sample.pdf", b"Fake PDF content", "application/pdf")}
    params = {"query": "What is AI?"}
    
    response = test_client.post("/query_pdf/", files=files, params=params)

    assert response.status_code == 200, f"Unexpected response code: {response.status_code}"
    assert "answer" in response.json(), "Response missing 'answer' key"

