"""Visualization utilities for evaluation results."""
import json
from pathlib import Path
from typing import Dict, Any, List
import matplotlib.pyplot as plt
import seaborn as sns


def plot_metrics_by_k(
    aggregated_metrics: Dict[str, Any],
    output_path: str,
    metrics_to_plot: List[str] = ["precision", "recall", "ndcg", "hit_rate"]
):
    """Plot retrieval metrics across different K values.
    
    Args:
        aggregated_metrics: Aggregated metrics from evaluation
        output_path: Path to save plot
        metrics_to_plot: List of metric names to plot
    """
    if "metrics_by_k" not in aggregated_metrics:
        print("No metrics_by_k found in aggregated_metrics")
        return
    
    # Extract data
    k_values = []
    metric_data = {metric: [] for metric in metrics_to_plot}
    
    for k_key in sorted(aggregated_metrics["metrics_by_k"].keys()):
        k_val = int(k_key.replace("@", ""))
        k_values.append(k_val)
        
        metrics = aggregated_metrics["metrics_by_k"][k_key]
        for metric in metrics_to_plot:
            if metric in metrics:
                metric_data[metric].append(metrics[metric]["mean"])
    
    # Create plot
    plt.figure(figsize=(12, 6))
    
    for metric in metrics_to_plot:
        if metric_data[metric]:
            plt.plot(k_values, metric_data[metric], marker='o', label=metric.replace('_', ' ').title())
    
    plt.xlabel('K (Number of Retrieved Documents)', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.title('Retrieval Metrics by K', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1.05)
    
    # Save plot
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Metrics plot saved to: {output_file}")


def plot_score_distribution(
    detailed_results: List[Dict[str, Any]],
    output_path: str,
    k: int = 5
):
    """Plot distribution of scores for a specific K value.
    
    Args:
        detailed_results: Detailed results from evaluation
        output_path: Path to save plot
        k: K value to plot
    """
    # Extract scores
    precision_scores = []
    recall_scores = []
    ndcg_scores = []
    
    k_key = f"@{k}"
    
    for result in detailed_results:
        if "retrieval_metrics" in result:
            metrics = result["retrieval_metrics"].get("metrics_by_k", {}).get(k_key, {})
            if metrics:
                precision_scores.append(metrics.get("precision", 0))
                recall_scores.append(metrics.get("recall", 0))
                ndcg_scores.append(metrics.get("ndcg", 0))
    
    # Check if we have any scores
    if not precision_scores:
        print(f"⚠️  No scores found for K={k}, skipping distribution plot")
        return
    
    # Check if all scores are zero (avoid meaningless plots)
    if all(s == 0 for s in precision_scores + recall_scores + ndcg_scores):
        print(f"⚠️  All scores are zero for K={k}, skipping distribution plot")
        return
    
    # Create subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Precision distribution
    axes[0].hist(precision_scores, bins=20, edgecolor='black', alpha=0.7)
    axes[0].set_xlabel(f'Precision@{k}')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title(f'Precision@{k} Distribution')
    if precision_scores and sum(precision_scores) > 0:
        axes[0].axvline(sum(precision_scores)/len(precision_scores), color='red', linestyle='--', label='Mean')
        axes[0].legend()
    
    # Recall distribution
    axes[1].hist(recall_scores, bins=20, edgecolor='black', alpha=0.7, color='orange')
    axes[1].set_xlabel(f'Recall@{k}')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title(f'Recall@{k} Distribution')
    if recall_scores and sum(recall_scores) > 0:
        axes[1].axvline(sum(recall_scores)/len(recall_scores), color='red', linestyle='--', label='Mean')
        axes[1].legend()
    
    # NDCG distribution
    axes[2].hist(ndcg_scores, bins=20, edgecolor='black', alpha=0.7, color='green')
    axes[2].set_xlabel(f'NDCG@{k}')
    axes[2].set_ylabel('Frequency')
    axes[2].set_title(f'NDCG@{k} Distribution')
    if ndcg_scores and sum(ndcg_scores) > 0:
        axes[2].axvline(sum(ndcg_scores)/len(ndcg_scores), color='red', linestyle='--', label='Mean')
        axes[2].legend()
    
    plt.tight_layout()
    
    # Save plot
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Distribution plot saved to: {output_file}")


def plot_ragas_metrics(
    aggregated_generation_metrics: Dict[str, Any],
    output_path: str
):
    """Plot RAGAS generation metrics.
    
    Args:
        aggregated_generation_metrics: Aggregated generation metrics
        output_path: Path to save plot
    """
    if "ragas" not in aggregated_generation_metrics:
        print("No RAGAS metrics found")
        return
    
    ragas_metrics = aggregated_generation_metrics["ragas"]
    
    # Extract metric names and values
    metric_names = []
    metric_values = []
    
    for metric_name, metric_data in ragas_metrics.items():
        metric_names.append(metric_name.replace('_', ' ').title())
        metric_values.append(metric_data["mean"])
    
    # Create bar plot
    plt.figure(figsize=(10, 6))
    bars = plt.bar(metric_names, metric_values, edgecolor='black', alpha=0.7)
    
    # Color bars based on score
    for bar, value in zip(bars, metric_values):
        if value >= 0.8:
            bar.set_color('green')
        elif value >= 0.6:
            bar.set_color('orange')
        else:
            bar.set_color('red')
    
    plt.xlabel('Metric', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.title('RAGAS Generation Quality Metrics', fontsize=14, fontweight='bold')
    plt.ylim(0, 1.05)
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, (name, value) in enumerate(zip(metric_names, metric_values)):
        plt.text(i, value + 0.02, f'{value:.3f}', ha='center', fontsize=10)
    
    plt.tight_layout()
    
    # Save plot
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ RAGAS metrics plot saved to: {output_file}")


def export_results_to_csv(
    results: Dict[str, Any],
    output_path: str
):
    """Export detailed results to CSV.
    
    Args:
        results: Evaluation results
        output_path: Path to save CSV
    """
    import csv
    
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Write header
        header = ['Query', 'Retrieved Count', 'Precision@5', 'Recall@5', 'NDCG@5', 'MRR']
        
        # Add generation metrics if available
        if results.get("detailed_results") and "generation_metrics" in results["detailed_results"][0]:
            header.extend(['Faithfulness', 'Answer Relevancy'])
        
        writer.writerow(header)
        
        # Write data
        for result in results.get("detailed_results", []):
            row = [
                result.get("query", "")[:100],
                result.get("retrieved_count", 0)
            ]
            
            # Retrieval metrics
            if "retrieval_metrics" in result:
                metrics_5 = result["retrieval_metrics"].get("metrics_by_k", {}).get("@5", {})
                row.extend([
                    metrics_5.get("precision", 0),
                    metrics_5.get("recall", 0),
                    metrics_5.get("ndcg", 0)
                ])
                row.append(result["retrieval_metrics"].get("mrr", 0))
            else:
                row.extend([0, 0, 0, 0])
            
            # Generation metrics
            if "generation_metrics" in result and "ragas" in result["generation_metrics"]:
                ragas = result["generation_metrics"]["ragas"]
                row.extend([
                    ragas.get("faithfulness", 0),
                    ragas.get("answer_relevancy", 0)
                ])
            
            writer.writerow(row)
    
    print(f"✅ Results exported to CSV: {output_file}")


def create_all_visualizations(
    results: Dict[str, Any],
    output_dir: str
):
    """Create all visualizations for evaluation results.
    
    Args:
        results: Evaluation results
        output_dir: Directory to save visualizations
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("\n📊 Generating visualizations...")
    
    # Metrics by K plot
    if "aggregated_retrieval_metrics" in results:
        plot_metrics_by_k(
            results["aggregated_retrieval_metrics"],
            str(output_path / "metrics_by_k.png")
        )
    
    # Score distribution
    if "detailed_results" in results:
        plot_score_distribution(
            results["detailed_results"],
            str(output_path / "score_distribution.png"),
            k=5
        )
    
    # RAGAS metrics
    if "aggregated_generation_metrics" in results:
        plot_ragas_metrics(
            results["aggregated_generation_metrics"],
            str(output_path / "ragas_metrics.png")
        )
    
    # Export to CSV
    export_results_to_csv(
        results,
        str(output_path / "results.csv")
    )
    
    print("✅ All visualizations created!")
