#!/usr/bin/env python3
"""Helper script to list document IDs from Qdrant for creating evaluation datasets.

This script helps you find the actual document chunk IDs stored in Qdrant,
which you need to use in your evaluation dataset's relevant_doc_ids.
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rag_storage.qdrant_client import QdrantManager
from backend.app.config import get_settings


def main():
    """List all documents and their chunks in Qdrant."""
    print("=" * 60)
    print("QDRANT DOCUMENT ID LISTER")
    print("=" * 60)
    print("\nThis script lists all document chunks stored in Qdrant.")
    print("Use these IDs in your evaluation dataset's 'relevant_doc_ids'.\n")
    
    # Load settings
    settings = get_settings()
    
    # Connect to Qdrant
    print("🔌 Connecting to Qdrant...")
    qdrant_manager = QdrantManager(
        host=settings.qdrant_host,
        port=settings.qdrant_port
    )
    
    # Get collection info
    try:
        info = qdrant_manager.get_collection_info()
        print(f"✅ Connected to collection: {info.get('name', 'documents')}")
        print(f"   Total points: {info.get('points_count', 0)}\n")
    except Exception as e:
        print(f"❌ Error connecting to Qdrant: {e}")
        return 1
    
    # Scroll through all points
    print("📄 Fetching all document chunks...\n")
    
    try:
        # Use scroll to get all points
        points, _ = qdrant_manager.client.scroll(
            collection_name=qdrant_manager.collection_name,
            limit=100,
            with_payload=True,
            with_vectors=False
        )
        
        if not points:
            print("⚠️  No documents found in Qdrant.")
            print("\nMake sure you've uploaded documents to the RAG system first.")
            return 0
        
        # Group by document_id
        docs_by_id = {}
        for point in points:
            payload = point.payload or {}
            doc_id = payload.get('document_id', 'unknown')
            chunk_id = str(point.id)
            
            if doc_id not in docs_by_id:
                docs_by_id[doc_id] = []
            
            docs_by_id[doc_id].append({
                'chunk_id': chunk_id,
                'page': payload.get('page_number', 'N/A'),
                'text': payload.get('text', '')[:100]
            })
        
        # Display results
        print(f"Found {len(docs_by_id)} document(s):\n")
        
        for doc_id, chunks in docs_by_id.items():
            print(f"📄 Document: {doc_id}")
            print(f"   Chunks: {len(chunks)}")
            print(f"\n   Chunk IDs (use these in your evaluation dataset):")
            
            for i, chunk in enumerate(chunks[:10], 1):  # Show first 10 chunks
                print(f"   {i}. {chunk['chunk_id']}")
                print(f"      Page: {chunk['page']}")
                print(f"      Text: {chunk['text']}...")
                print()
            
            if len(chunks) > 10:
                print(f"   ... and {len(chunks) - 10} more chunks\n")
        
        # Print example evaluation dataset entry
        print("\n" + "=" * 60)
        print("EXAMPLE EVALUATION DATASET ENTRY")
        print("=" * 60)
        
        if points:
            first_chunk = str(points[0].id)
            print(f"""
{{
  "examples": [
    {{
      "query": "Your question here",
      "relevant_doc_ids": ["{first_chunk}"],
      "relevance_scores": {{"{first_chunk}": 1.0}},
      "ground_truth_answer": "Expected answer (optional)"
    }}
  ]
}}
""")
        
        print("\n💡 TIP: Copy the chunk IDs from above and use them in your")
        print("   evaluation dataset's 'relevant_doc_ids' field.")
        
        return 0
        
    except Exception as e:
        print(f"❌ Error fetching documents: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
