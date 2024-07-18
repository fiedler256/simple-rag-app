import pytest
from fastapi import HTTPException
from app.models import generate_response


def test_generate_response_empty_query():
    """
    Test generate_response with an empty query.
    """
    with pytest.raises(HTTPException, match="400: Query cannot be empty."):
        generate_response("", "Document content")


def test_generate_response_long_query():
    """
    Test generate_response with a long query.
    """
    long_query = "example word" * 150
    with pytest.raises(HTTPException, match="400: Query too long."):
        generate_response(long_query, "Document content")


def test_generate_response_success():
    """
    Test generate_response with a valid query and document.
    """
    response = generate_response("What is the return policy?", "Document content")
    assert type(response) is str
