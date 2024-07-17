import pytest
from app.retrieval import get_embeddings, build_faiss_index, retrieve_documents


def test_get_embeddings_empty():
    """
    Test get_embeddings with an empty document list.
    """
    with pytest.raises(ValueError, match="Document list is empty."):
        get_embeddings([])


def test_build_faiss_index():
    """
    Test build_faiss_index with valid documents.
    """
    documents = ["This is a test document."]
    index = build_faiss_index(documents)
    assert index.ntotal == 1


def test_retrieve_documents():
    """
    Test retrieve_documents with a valid query and documents.
    """
    documents = ["This is a test document."]
    index = build_faiss_index(documents)
    retrieved_docs = retrieve_documents("test", documents, index)
    assert "This is a test document." in retrieved_docs
