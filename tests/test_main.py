from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_query_success():
    """
    Test the query endpoint with a valid query.
    """
    response = client.post("/query", json={"query": "What is the return policy?"})
    assert response.status_code == 200
    assert "response" in response.json()
