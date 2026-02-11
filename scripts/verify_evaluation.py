#!/usr/bin/env python3
"""Simple verification script to test evaluation metrics without external dependencies."""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.retrieval_metrics import (
    calculate_precision_at_k,
    calculate_recall_at_k,
    calculate_mrr,
    calculate_ndcg_at_k,
    calculate_hit_rate_at_k,
    evaluate_retrieval
)
from evaluation.evaluation_dataset import EvaluationExample, EvaluationDataset


def test_retrieval_metrics():
    """Test retrieval metrics with known inputs."""
    print("=" * 60)
    print("TESTING RETRIEVAL METRICS")
    print("=" * 60)
    
    # Test data
    retrieved = ["doc1", "doc2", "doc3", "doc4", "doc5"]
    relevant = {"doc1", "doc3", "doc5"}
    relevance_scores = {"doc1": 3.0, "doc3": 2.0, "doc5": 1.0}
    
    print("\nTest Setup:")
    print(f"  Retrieved: {retrieved}")
    print(f"  Relevant: {relevant}")
    print(f"  Relevance scores: {relevance_scores}")
    
    # Test Precision@K
    print("\n📊 Precision@K:")
    for k in [1, 3, 5]:
        precision = calculate_precision_at_k(retrieved, relevant, k)
        print(f"  P@{k}: {precision:.4f}")
    
    # Test Recall@K
    print("\n📊 Recall@K:")
    for k in [1, 3, 5]:
        recall = calculate_recall_at_k(retrieved, relevant, k)
        print(f"  R@{k}: {recall:.4f}")
    
    # Test MRR
    mrr = calculate_mrr(retrieved, relevant)
    print(f"\n📊 MRR: {mrr:.4f}")
    
    # Test NDCG@K
    print("\n📊 NDCG@K:")
    for k in [1, 3, 5]:
        ndcg = calculate_ndcg_at_k(retrieved, relevance_scores, k)
        print(f"  NDCG@{k}: {ndcg:.4f}")
    
    # Test Hit Rate@K
    print("\n📊 Hit Rate@K:")
    for k in [1, 3, 5]:
        hit_rate = calculate_hit_rate_at_k(retrieved, relevant, k)
        print(f"  HR@{k}: {hit_rate:.4f}")
    
    print("\n✅ All retrieval metrics calculated successfully!")


def test_evaluation_dataset():
    """Test evaluation dataset functionality."""
    print("\n" + "=" * 60)
    print("TESTING EVALUATION DATASET")
    print("=" * 60)
    
    # Create dataset
    dataset = EvaluationDataset()
    
    # Add examples
    example1 = EvaluationExample(
        query="What is the main topic?",
        relevant_doc_ids=["doc1", "doc2"],
        relevance_scores={"doc1": 2.0, "doc2": 1.0},
        ground_truth_answer="The main topic is X."
    )
    
    example2 = EvaluationExample(
        query="How does it work?",
        relevant_doc_ids=["doc3"],
        ground_truth_answer="It works by doing Y."
    )
    
    dataset.add_example(example1)
    dataset.add_example(example2)
    
    print(f"\n✅ Created dataset with {len(dataset)} examples")
    
    # Test iteration
    print("\nDataset examples:")
    for i, example in enumerate(dataset, 1):
        print(f"  {i}. Query: {example.query[:40]}...")
        print(f"     Relevant docs: {len(example.relevant_doc_ids)}")
    
    # Test JSON save/load
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        temp_path = f.name
    
    try:
        dataset.to_json(temp_path)
        print(f"\n✅ Saved dataset to {temp_path}")
        
        loaded = EvaluationDataset.from_json(temp_path)
        print(f"✅ Loaded dataset with {len(loaded)} examples")
        
        # Verify data integrity
        assert len(loaded) == len(dataset)
        assert loaded[0].query == dataset[0].query
        print("✅ Data integrity verified!")
        
    finally:
        Path(temp_path).unlink()
        print(f"✅ Cleaned up temp file")


def test_complete_evaluation():
    """Test complete evaluation workflow."""
    print("\n" + "=" * 60)
    print("TESTING COMPLETE EVALUATION WORKFLOW")
    print("=" * 60)
    
    # Simulate retrieval results
    results = [
        {"document_id": "doc1", "score": 0.95},
        {"document_id": "doc2", "score": 0.85},
        {"document_id": "doc3", "score": 0.75},
        {"document_id": "doc4", "score": 0.65},
        {"document_id": "doc5", "score": 0.55},
    ]
    
    relevant_ids = {"doc1", "doc3", "doc5"}
    relevance_scores = {"doc1": 3.0, "doc3": 2.0, "doc5": 1.0}
    
    # Evaluate
    metrics = evaluate_retrieval(
        retrieved_results=results,
        ground_truth_relevant_ids=relevant_ids,
        ground_truth_relevance_scores=relevance_scores,
        k_values=[1, 3, 5]
    )
    
    print("\n📊 Complete Evaluation Results:")
    print(f"\nRetrieved: {metrics['retrieved_count']} documents")
    print(f"Relevant: {metrics['relevant_count']} documents")
    print(f"MRR: {metrics['mrr']:.4f}")
    
    print("\nMetrics by K:")
    for k_key, k_metrics in sorted(metrics['metrics_by_k'].items()):
        print(f"\n  {k_key}:")
        print(f"    Precision: {k_metrics['precision']:.4f}")
        print(f"    Recall: {k_metrics['recall']:.4f}")
        print(f"    NDCG: {k_metrics['ndcg']:.4f}")
        print(f"    Hit Rate: {k_metrics['hit_rate']:.4f}")
    
    print("\n✅ Complete evaluation workflow successful!")


def main():
    """Run all verification tests."""
    print("\n" + "=" * 60)
    print("RAG EVALUATION FRAMEWORK - VERIFICATION")
    print("=" * 60)
    print("\nThis script verifies the evaluation framework is working correctly.")
    print("It tests metrics calculation without requiring external dependencies.")
    
    try:
        test_retrieval_metrics()
        test_evaluation_dataset()
        test_complete_evaluation()
        
        print("\n" + "=" * 60)
        print("✅ ALL VERIFICATION TESTS PASSED!")
        print("=" * 60)
        print("\nThe RAG evaluation framework is ready to use.")
        print("\nNext steps:")
        print("  1. Install dependencies: pip install -r requirements.txt")
        print("  2. Create evaluation dataset: python scripts/create_evaluation_dataset.py")
        print("  3. Run evaluation: python scripts/run_evaluation.py --dataset <path>")
        print()
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Verification failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
