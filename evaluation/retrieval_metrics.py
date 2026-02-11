"""Retrieval quality metrics for RAG evaluation.

This module implements standard information retrieval metrics:
- Precision@K: Proportion of retrieved docs that are relevant
- Recall@K: Proportion of relevant docs that were retrieved
- MRR: Mean Reciprocal Rank
- NDCG@K: Normalized Discounted Cumulative Gain
- Hit Rate@K: Whether at least one relevant doc appears in top K
"""
from typing import List, Dict, Set
import math
from collections import defaultdict


def calculate_precision_at_k(
    retrieved_ids: List[str],
    relevant_ids: Set[str],
    k: int
) -> float:
    """Calculate Precision@K.
    
    Precision@K = (# relevant docs in top K) / K
    
    Args:
        retrieved_ids: List of retrieved document IDs (ordered by relevance)
        relevant_ids: Set of ground truth relevant document IDs
        k: Number of top results to consider
        
    Returns:
        Precision@K score (0.0 to 1.0)
    """
    if k <= 0:
        return 0.0
    
    top_k = retrieved_ids[:k]
    relevant_in_top_k = sum(1 for doc_id in top_k if doc_id in relevant_ids)
    
    return relevant_in_top_k / k


def calculate_recall_at_k(
    retrieved_ids: List[str],
    relevant_ids: Set[str],
    k: int
) -> float:
    """Calculate Recall@K.
    
    Recall@K = (# relevant docs in top K) / (total # relevant docs)
    
    Args:
        retrieved_ids: List of retrieved document IDs (ordered by relevance)
        relevant_ids: Set of ground truth relevant document IDs
        k: Number of top results to consider
        
    Returns:
        Recall@K score (0.0 to 1.0)
    """
    if not relevant_ids:
        return 0.0
    
    top_k = retrieved_ids[:k]
    relevant_in_top_k = sum(1 for doc_id in top_k if doc_id in relevant_ids)
    
    return relevant_in_top_k / len(relevant_ids)


def calculate_mrr(
    retrieved_ids: List[str],
    relevant_ids: Set[str]
) -> float:
    """Calculate Mean Reciprocal Rank (MRR).
    
    MRR = 1 / rank of first relevant document
    
    Args:
        retrieved_ids: List of retrieved document IDs (ordered by relevance)
        relevant_ids: Set of ground truth relevant document IDs
        
    Returns:
        MRR score (0.0 to 1.0)
    """
    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in relevant_ids:
            return 1.0 / rank
    
    return 0.0


def calculate_ndcg_at_k(
    retrieved_ids: List[str],
    relevance_scores: Dict[str, float],
    k: int
) -> float:
    """Calculate Normalized Discounted Cumulative Gain (NDCG@K).
    
    NDCG accounts for graded relevance and position of results.
    
    DCG@K = sum(rel_i / log2(i + 1)) for i in 1..K
    NDCG@K = DCG@K / IDCG@K
    
    Args:
        retrieved_ids: List of retrieved document IDs (ordered by relevance)
        relevance_scores: Dict mapping doc_id to relevance score (higher = more relevant)
        k: Number of top results to consider
        
    Returns:
        NDCG@K score (0.0 to 1.0)
    """
    if k <= 0:
        return 0.0
    
    # Calculate DCG@K
    dcg = 0.0
    for i, doc_id in enumerate(retrieved_ids[:k], start=1):
        rel = relevance_scores.get(doc_id, 0.0)
        dcg += rel / math.log2(i + 1)
    
    # Calculate IDCG@K (ideal DCG with perfect ranking)
    sorted_relevances = sorted(relevance_scores.values(), reverse=True)
    idcg = 0.0
    for i, rel in enumerate(sorted_relevances[:k], start=1):
        idcg += rel / math.log2(i + 1)
    
    # Avoid division by zero
    if idcg == 0.0:
        return 0.0
    
    return dcg / idcg


def calculate_hit_rate_at_k(
    retrieved_ids: List[str],
    relevant_ids: Set[str],
    k: int
) -> float:
    """Calculate Hit Rate@K.
    
    Hit Rate@K = 1 if at least one relevant doc in top K, else 0
    
    Args:
        retrieved_ids: List of retrieved document IDs (ordered by relevance)
        relevant_ids: Set of ground truth relevant document IDs
        k: Number of top results to consider
        
    Returns:
        Hit rate (0.0 or 1.0)
    """
    top_k = retrieved_ids[:k]
    for doc_id in top_k:
        if doc_id in relevant_ids:
            return 1.0
    return 0.0


def evaluate_retrieval(
    retrieved_results: List[Dict[str, any]],
    ground_truth_relevant_ids: Set[str],
    ground_truth_relevance_scores: Dict[str, float],
    k_values: List[int] = [1, 3, 5, 10]
) -> Dict[str, any]:
    """Evaluate retrieval results with multiple metrics.
    
    Args:
        retrieved_results: List of search results (each with 'document_id' or 'id' key)
        ground_truth_relevant_ids: Set of relevant document IDs
        ground_truth_relevance_scores: Dict of doc_id -> relevance score
        k_values: List of K values to evaluate at
        
    Returns:
        Dictionary with metrics for each K value and overall metrics
    """
    # Extract document IDs from results
    retrieved_ids = []
    for result in retrieved_results:
        # Handle different possible key names
        doc_id = result.get('document_id') or result.get('id') or result.get('chunk_id')
        if doc_id:
            retrieved_ids.append(doc_id)
    
    # Calculate metrics for each K
    metrics = {
        "retrieved_count": len(retrieved_ids),
        "relevant_count": len(ground_truth_relevant_ids),
        "metrics_by_k": {}
    }
    
    for k in k_values:
        metrics["metrics_by_k"][f"@{k}"] = {
            "precision": calculate_precision_at_k(retrieved_ids, ground_truth_relevant_ids, k),
            "recall": calculate_recall_at_k(retrieved_ids, ground_truth_relevant_ids, k),
            "ndcg": calculate_ndcg_at_k(retrieved_ids, ground_truth_relevance_scores, k),
            "hit_rate": calculate_hit_rate_at_k(retrieved_ids, ground_truth_relevant_ids, k),
        }
    
    # Calculate MRR (not K-dependent)
    metrics["mrr"] = calculate_mrr(retrieved_ids, ground_truth_relevant_ids)
    
    return metrics


def aggregate_metrics(
    all_metrics: List[Dict[str, any]],
    k_values: List[int] = [1, 3, 5, 10]
) -> Dict[str, any]:
    """Aggregate metrics across multiple queries.
    
    Args:
        all_metrics: List of metric dictionaries from evaluate_retrieval
        k_values: List of K values
        
    Returns:
        Aggregated metrics with means and standard deviations
    """
    if not all_metrics:
        return {}
    
    aggregated = {
        "num_queries": len(all_metrics),
        "metrics_by_k": {},
        "mrr": {
            "mean": sum(m["mrr"] for m in all_metrics) / len(all_metrics),
            "values": [m["mrr"] for m in all_metrics]
        }
    }
    
    for k in k_values:
        k_key = f"@{k}"
        
        precision_values = [m["metrics_by_k"][k_key]["precision"] for m in all_metrics]
        recall_values = [m["metrics_by_k"][k_key]["recall"] for m in all_metrics]
        ndcg_values = [m["metrics_by_k"][k_key]["ndcg"] for m in all_metrics]
        hit_rate_values = [m["metrics_by_k"][k_key]["hit_rate"] for m in all_metrics]
        
        aggregated["metrics_by_k"][k_key] = {
            "precision": {
                "mean": sum(precision_values) / len(precision_values),
                "values": precision_values
            },
            "recall": {
                "mean": sum(recall_values) / len(recall_values),
                "values": recall_values
            },
            "ndcg": {
                "mean": sum(ndcg_values) / len(ndcg_values),
                "values": ndcg_values
            },
            "hit_rate": {
                "mean": sum(hit_rate_values) / len(hit_rate_values),
                "values": hit_rate_values
            }
        }
    
    return aggregated
