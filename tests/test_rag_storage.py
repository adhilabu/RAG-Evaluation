"""Test Qdrant connection and basic operations."""
import pytest
from rag_storage import QdrantManager


def test_qdrant_connection():
    """Test Qdrant connection."""
    manager = QdrantManager(host="localhost", port=6333)
    
    # This should not raise an error
    info = manager.get_collection_info()
    print(f"Collection info: {info}")


def test_create_collection():
    """Test collection creation."""
    manager = QdrantManager(host="localhost", port=6333)
    
    # Create test collection
    manager.collection_name = "test_collection"
    manager.create_collection(vector_size=1536, force=True)
    
    info = manager.get_collection_info()
    assert info.get("name") == "test_collection"
    
    # Cleanup
    manager.client.delete_collection("test_collection")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
