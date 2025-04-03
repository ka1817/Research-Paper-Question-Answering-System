import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_health_check():
    """Test if the backend is running by checking an available endpoint."""
    response = client.get("/query_pdf/")  # Change from `/` to `/query_pdf/`
    assert response.status_code in [405, 422], f"Expected 405 or 422, got {response.status_code}"

def test_query_pdf():
    """Test querying a sample PDF."""
    pdf_path = "tests/sample.pdf"  # Ensure this file exists in your tests folder.

    with open(pdf_path, "rb") as pdf_file:
        files = {"file": ("sample.pdf", pdf_file, "application/pdf")}
        data = {"query": "What is the main topic of the paper?"}

        response = client.post("/query_pdf/", files=files, data=data)

    assert response.status_code == 200, f"Query failed! Expected 200, got {response.status_code}"
    assert "answer" in response.json(), "Response does not contain an answer."
