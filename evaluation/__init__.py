"""RAG Evaluation Framework.

This module provides comprehensive evaluation tools for RAG systems including:
- Retrieval quality metrics (Precision@K, Recall@K, MRR, NDCG)
- Generation quality metrics (RAGAS-based)
- End-to-end evaluation pipelines
"""

from .evaluation_dataset import EvaluationExample, EvaluationDataset
from .retrieval_metrics import (
    calculate_precision_at_k,
    calculate_recall_at_k,
    calculate_mrr,
    calculate_ndcg_at_k,
    calculate_hit_rate_at_k,
    evaluate_retrieval,
)
from .evaluation_pipeline import RAGEvaluator

__all__ = [
    "EvaluationExample",
    "EvaluationDataset",
    "calculate_precision_at_k",
    "calculate_recall_at_k",
    "calculate_mrr",
    "calculate_ndcg_at_k",
    "calculate_hit_rate_at_k",
    "evaluate_retrieval",
    "RAGEvaluator",
]
