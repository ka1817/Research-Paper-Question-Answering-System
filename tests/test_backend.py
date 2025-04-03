import pytest
from fastapi.testclient import TestClient
from main import app 

client = TestClient(app)

def test_health_check():
    """Test if the backend is running."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Backend is running"}

def test_query_pdf():
    """Test the PDF query endpoint with a sample question."""
    files = {"file": ("sample.pdf", b"Fake PDF content", "application/pdf")}
    params = {"query": "What is AI?"}
    
    response = client.post("/query_pdf/", files=files, params=params)
    
    assert response.status_code == 200
    assert "answer" in response.json()
