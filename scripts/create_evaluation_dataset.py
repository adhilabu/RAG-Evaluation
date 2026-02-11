#!/usr/bin/env python3
"""Interactive script to create evaluation datasets.

This script helps you create evaluation datasets by:
1. Listing available documents in Qdrant
2. Allowing you to add evaluation examples interactively
3. Saving the dataset to JSON format
"""
import argparse
import sys
import json
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation import EvaluationDataset, EvaluationExample
from rag_storage.qdrant_client import QdrantManager
from backend.app.config import get_settings


def list_available_documents(qdrant_manager: QdrantManager):
    """List available documents in Qdrant."""
    try:
        info = qdrant_manager.get_collection_info()
        print(f"\n📚 Collection: {info.get('name', 'documents')}")
        print(f"   Total vectors: {info.get('vectors_count', 0)}")
        print(f"   Points count: {info.get('points_count', 0)}")
    except Exception as e:
        print(f"⚠️  Could not get collection info: {e}")


def create_example_interactively() -> EvaluationExample:
    """Create an evaluation example interactively."""
    print("\n" + "=" * 60)
    print("CREATE NEW EVALUATION EXAMPLE")
    print("=" * 60)
    
    # Get query
    query = input("\nEnter the query: ").strip()
    while not query:
        print("Query cannot be empty!")
        query = input("Enter the query: ").strip()
    
    # Get relevant document IDs
    print("\nEnter relevant document chunk IDs (one per line, empty line to finish):")
    relevant_doc_ids = []
    while True:
        doc_id = input(f"  Document ID {len(relevant_doc_ids) + 1}: ").strip()
        if not doc_id:
            break
        relevant_doc_ids.append(doc_id)
    
    if not relevant_doc_ids:
        print("⚠️  Warning: No relevant document IDs provided. Adding a placeholder.")
        relevant_doc_ids = ["placeholder_doc_id"]
    
    # Get relevance scores (optional)
    use_graded = input("\nUse graded relevance scores? (y/n, default: n): ").strip().lower()
    relevance_scores = None
    
    if use_graded == 'y':
        relevance_scores = {}
        print("Enter relevance scores (higher = more relevant):")
        for doc_id in relevant_doc_ids:
            while True:
                try:
                    score = input(f"  Score for {doc_id}: ").strip()
                    if score:
                        relevance_scores[doc_id] = float(score)
                        break
                    else:
                        relevance_scores[doc_id] = 1.0
                        break
                except ValueError:
                    print("    Invalid score, please enter a number")
    
    # Get ground truth answer (optional)
    ground_truth = input("\nEnter ground truth answer (optional, press Enter to skip): ").strip()
    if not ground_truth:
        ground_truth = None
    
    # Create example
    example = EvaluationExample(
        query=query,
        relevant_doc_ids=relevant_doc_ids,
        relevance_scores=relevance_scores,
        ground_truth_answer=ground_truth
    )
    
    print("\n✅ Example created!")
    print(f"   Query: {query[:60]}...")
    print(f"   Relevant docs: {len(relevant_doc_ids)}")
    
    return example


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Create evaluation datasets interactively"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output path for evaluation dataset JSON"
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append to existing dataset if it exists"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("EVALUATION DATASET CREATOR")
    print("=" * 60)
    
    # Load settings
    settings = get_settings()
    
    # Connect to Qdrant
    print("\n🔌 Connecting to Qdrant...")
    qdrant_manager = QdrantManager(
        host=settings.qdrant_host,
        port=settings.qdrant_port
    )
    
    list_available_documents(qdrant_manager)
    
    # Load existing dataset if appending
    dataset = EvaluationDataset()
    if args.append and Path(args.output).exists():
        print(f"\n📂 Loading existing dataset from {args.output}...")
        try:
            dataset = EvaluationDataset.from_json(args.output)
            print(f"✅ Loaded {len(dataset)} existing examples")
        except Exception as e:
            print(f"⚠️  Could not load existing dataset: {e}")
            print("   Starting with empty dataset")
    
    # Create examples interactively
    print("\n" + "=" * 60)
    print("ADD EVALUATION EXAMPLES")
    print("=" * 60)
    print("(Press Ctrl+C to finish and save)")
    
    try:
        while True:
            example = create_example_interactively()
            dataset.add_example(example)
            
            cont = input("\nAdd another example? (y/n, default: y): ").strip().lower()
            if cont == 'n':
                break
    except KeyboardInterrupt:
        print("\n\n⏸️  Interrupted by user")
    
    # Save dataset
    if len(dataset) > 0:
        print(f"\n💾 Saving dataset to {args.output}...")
        try:
            dataset.to_json(args.output)
            print(f"✅ Saved {len(dataset)} examples to {args.output}")
        except Exception as e:
            print(f"❌ Error saving dataset: {e}")
            return 1
    else:
        print("\n⚠️  No examples to save")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
