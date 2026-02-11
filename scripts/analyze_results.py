#!/usr/bin/env python3
"""Analyze evaluation results and show what was actually retrieved."""

import json
import sys
from pathlib import Path

def analyze_results(results_path):
    """Analyze and display evaluation results in a human-readable format."""
    
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    print("=" * 80)
    print("RAG EVALUATION RESULTS ANALYSIS")
    print("=" * 80)
    
    # Overall summary
    print(f"\n📊 Overall Summary:")
    print(f"   Evaluation Type: {results['evaluation_type']}")
    print(f"   Number of Queries: {results['num_queries']}")
    print(f"   Timestamp: {results['timestamp']}")
    
    # Retrieval metrics summary
    if 'aggregated_retrieval_metrics' in results:
        agg = results['aggregated_retrieval_metrics']
        print(f"\n📈 Retrieval Metrics (Aggregated):")
        print(f"   MRR: {agg['mrr']['mean']:.4f}")
        
        if agg['mrr']['mean'] == 0:
            print("\n   ⚠️  WARNING: All retrieval metrics are ZERO!")
            print("   This likely means your ground truth document IDs don't match")
            print("   the actual document IDs in Qdrant.")
            print("\n   Run: python scripts/list_document_ids.py")
            print("   to see the actual document IDs you should use.")
    
    # Generation metrics summary
    if 'aggregated_generation_metrics' in results:
        agg_gen = results['aggregated_generation_metrics']
        print(f"\n🤖 Generation Metrics (RAGAS):")
        
        if 'ragas' in agg_gen:
            for metric_name, metric_data in agg_gen['ragas'].items():
                value = metric_data['mean']
                
                # Interpret the metric
                if metric_name == 'faithfulness':
                    interpretation = "✅ Excellent" if value > 0.8 else "⚠️ Needs improvement"
                    print(f"   {metric_name.replace('_', ' ').title()}: {value:.4f} - {interpretation}")
                    print(f"      → Measures if answers are grounded in retrieved context")
                elif metric_name == 'answer_relevancy':
                    if str(value) == 'nan':
                        print(f"   {metric_name.replace('_', ' ').title()}: N/A (error in calculation)")
                    else:
                        interpretation = "✅ Good" if value > 0.7 else "⚠️ Needs improvement"
                        print(f"   {metric_name.replace('_', ' ').title()}: {value:.4f} - {interpretation}")
                    print(f"      → Measures how relevant the answer is to the query")
                elif metric_name == 'context_precision':
                    interpretation = "✅ Good" if value > 0.6 else "⚠️ Needs improvement"
                    print(f"   {metric_name.replace('_', ' ').title()}: {value:.4f} - {interpretation}")
                    print(f"      → Measures how relevant the retrieved contexts are")
                elif metric_name == 'context_recall':
                    interpretation = "✅ Very Good" if value > 0.7 else "⚠️ Needs improvement"
                    print(f"   {metric_name.replace('_', ' ').title()}: {value:.4f} - {interpretation}")
                    print(f"      → Measures if all relevant info was retrieved")
    
    # Detailed query-by-query analysis
    print("\n" + "=" * 80)
    print("QUERY-BY-QUERY ANALYSIS")
    print("=" * 80)
    
    for i, result in enumerate(results.get('detailed_results', []), 1):
        print(f"\n📝 Query {i}:")
        print(f"   Question: {result['query'][:70]}...")
        print(f"   Retrieved: {result.get('retrieved_count', 0)} documents")
        
        # Show what was actually retrieved
        if 'retrieved_ids' in result:
            print(f"\n   Retrieved Document IDs:")
            for j, doc_id in enumerate(result['retrieved_ids'][:3], 1):
                print(f"      {j}. {doc_id}")
            if len(result['retrieved_ids']) > 3:
                print(f"      ... and {len(result['retrieved_ids']) - 3} more")
        
        # Show ground truth
        if 'ground_truth_ids' in result:
            print(f"\n   Expected Document IDs (Ground Truth):")
            for j, doc_id in enumerate(result['ground_truth_ids'], 1):
                print(f"      {j}. {doc_id}")
        
        # Show if there's a mismatch
        if 'retrieved_ids' in result and 'ground_truth_ids' in result:
            retrieved_set = set(result['retrieved_ids'])
            ground_truth_set = set(result['ground_truth_ids'])
            matches = retrieved_set.intersection(ground_truth_set)
            
            if matches:
                print(f"\n   ✅ Matches found: {len(matches)} document(s)")
            else:
                print(f"\n   ❌ No matches - IDs don't match!")
        
        # Show generation metrics if available
        if 'generation_metrics' in result and 'ragas' in result['generation_metrics']:
            ragas = result['generation_metrics']['ragas']
            print(f"\n   Generation Quality:")
            if 'faithfulness' in ragas:
                print(f"      Faithfulness: {ragas['faithfulness']:.4f}")
            if 'context_precision' in ragas:
                print(f"      Context Precision: {ragas['context_precision']:.4f}")
    
    # Recommendations
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)
    
    if 'aggregated_retrieval_metrics' in results:
        if results['aggregated_retrieval_metrics']['mrr']['mean'] == 0:
            print("\n🔧 Fix Retrieval Metrics:")
            print("   1. Run: python scripts/list_document_ids.py")
            print("   2. Update your evaluation dataset with the actual UUIDs")
            print("   3. Re-run the evaluation")
    
    if 'aggregated_generation_metrics' in results:
        agg_gen = results['aggregated_generation_metrics']
        if 'ragas' in agg_gen:
            faithfulness = agg_gen['ragas'].get('faithfulness', {}).get('mean', 0)
            context_precision = agg_gen['ragas'].get('context_precision', {}).get('mean', 0)
            
            if faithfulness > 0.8:
                print("\n✅ Your RAG system has excellent faithfulness!")
                print("   Answers are well-grounded in the retrieved context.")
            
            if context_precision < 0.7:
                print("\n⚠️  Consider improving context precision:")
                print("   - Adjust similarity threshold")
                print("   - Reduce number of retrieved documents")
                print("   - Improve embedding quality")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    results_path = "evaluation_results/run_001/results.json"
    
    if len(sys.argv) > 1:
        results_path = sys.argv[1]
    
    if not Path(results_path).exists():
        print(f"❌ Results file not found: {results_path}")
        print(f"\nUsage: python {sys.argv[0]} [path/to/results.json]")
        sys.exit(1)
    
    analyze_results(results_path)
