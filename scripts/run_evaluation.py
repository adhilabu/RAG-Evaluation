#!/usr/bin/env python3
"""Main evaluation runner script.

Usage:
    python scripts/run_evaluation.py \\
        --dataset evaluation_datasets/sample_eval.json \\
        --output evaluation_results/run_001 \\
        --k-values 1,3,5,10 \\
        --include-generation
"""
import argparse
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation import RAGEvaluator, EvaluationDataset
from evaluation.visualizations import create_all_visualizations
from rag_storage.qdrant_client import QdrantManager
from backend.app.config import get_settings


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run RAG evaluation on a dataset"
    )
    
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to evaluation dataset JSON file"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory for results"
    )
    
    parser.add_argument(
        "--k-values",
        type=str,
        default="1,3,5,10",
        help="Comma-separated K values for metrics (default: 1,3,5,10)"
    )
    
    parser.add_argument(
        "--include-generation",
        action="store_true",
        help="Include generation quality evaluation (requires LLM calls)"
    )
    
    parser.add_argument(
        "--include-rouge",
        action="store_true",
        help="Include ROUGE scores (requires ground truth answers)"
    )
    
    parser.add_argument(
        "--retrieval-limit",
        type=int,
        default=10,
        help="Number of documents to retrieve (default: 10)"
    )
    
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.0,
        help="Minimum similarity score threshold (default: 0.0)"
    )
    
    parser.add_argument(
        "--no-visualizations",
        action="store_true",
        help="Skip generating visualizations"
    )
    
    return parser.parse_args()


def main():
    """Main evaluation runner."""
    args = parse_args()
    
    # Parse K values
    k_values = [int(k.strip()) for k in args.k_values.split(",")]
    
    print("=" * 60)
    print("RAG EVALUATION")
    print("=" * 60)
    print(f"Dataset: {args.dataset}")
    print(f"Output: {args.output}")
    print(f"K values: {k_values}")
    print(f"Include generation: {args.include_generation}")
    print(f"Retrieval limit: {args.retrieval_limit}")
    print("=" * 60)
    print()
    
    # Load settings
    settings = get_settings()
    
    # Initialize Qdrant manager
    print("🔌 Connecting to Qdrant...")
    qdrant_manager = QdrantManager(
        host=settings.qdrant_host,
        port=settings.qdrant_port
    )
    
    # Load evaluation dataset
    print(f"📂 Loading evaluation dataset from {args.dataset}...")
    try:
        dataset = EvaluationDataset.from_json(args.dataset)
        print(f"✅ Loaded {len(dataset)} evaluation examples")
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return 1
    
    # Initialize evaluator
    print("🚀 Initializing evaluator...")
    evaluator = RAGEvaluator(
        qdrant_manager=qdrant_manager,
        api_key=settings.openai_api_key,
        llm_model=settings.llm_model
    )
    
    # Run evaluation
    print("\n" + "=" * 60)
    print("RUNNING EVALUATION")
    print("=" * 60)
    
    if args.include_generation:
        print("Mode: End-to-end (retrieval + generation)")
        results = evaluator.evaluate_end_to_end(
            dataset=dataset,
            k_values=k_values,
            score_threshold=args.score_threshold,
            retrieval_limit=args.retrieval_limit,
            include_generation=True,
            include_rouge=args.include_rouge
        )
    else:
        print("Mode: Retrieval only")
        results = evaluator.evaluate_retrieval_only(
            dataset=dataset,
            k_values=k_values,
            score_threshold=args.score_threshold,
            limit=args.retrieval_limit
        )
    
    # Generate report
    print("\n" + "=" * 60)
    print("GENERATING REPORT")
    print("=" * 60)
    
    evaluator.generate_report(results, args.output)
    
    # Create visualizations
    if not args.no_visualizations:
        try:
            create_all_visualizations(results, args.output)
        except Exception as e:
            print(f"⚠️  Warning: Could not create visualizations: {e}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    
    if "aggregated_retrieval_metrics" in results:
        agg = results["aggregated_retrieval_metrics"]
        
        print("\n📊 Retrieval Metrics:")
        if "mrr" in agg:
            print(f"  MRR: {agg['mrr']['mean']:.4f}")
        
        if "metrics_by_k" in agg:
            for k_key in sorted(agg["metrics_by_k"].keys()):
                metrics = agg["metrics_by_k"][k_key]
                print(f"\n  Metrics {k_key}:")
                print(f"    Precision: {metrics['precision']['mean']:.4f}")
                print(f"    Recall: {metrics['recall']['mean']:.4f}")
                print(f"    NDCG: {metrics['ndcg']['mean']:.4f}")
                print(f"    Hit Rate: {metrics['hit_rate']['mean']:.4f}")
    
    if "aggregated_generation_metrics" in results:
        agg_gen = results["aggregated_generation_metrics"]
        
        print("\n🤖 Generation Metrics (RAGAS):")
        if "ragas" in agg_gen:
            for metric_name, metric_data in agg_gen["ragas"].items():
                print(f"  {metric_name.replace('_', ' ').title()}: {metric_data['mean']:.4f}")
    
    print("\n" + "=" * 60)
    print("✅ EVALUATION COMPLETE!")
    print("=" * 60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
