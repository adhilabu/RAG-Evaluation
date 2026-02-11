"""Generation quality metrics using RAGAS framework.

RAGAS (Retrieval Augmented Generation Assessment) provides LLM-based metrics:
- Faithfulness: Whether answer is grounded in retrieved context
- Answer Relevancy: How relevant the answer is to the query
- Context Precision: How relevant the retrieved contexts are
- Context Recall: Whether all relevant information was retrieved
"""
from typing import List, Dict, Optional
import os


def calculate_ragas_metrics(
    query: str,
    answer: str,
    contexts: List[str],
    ground_truth: Optional[str] = None,
    api_key: Optional[str] = None
) -> Dict[str, float]:
    """Calculate RAGAS metrics for a generated answer.
    
    Args:
        query: The user query
        answer: Generated answer
        contexts: List of retrieved context strings
        ground_truth: Optional ground truth answer (for context recall)
        api_key: OpenAI API key (uses env var if not provided)
        
    Returns:
        Dictionary with RAGAS metrics
    """
    try:
        from ragas import evaluate
        from ragas.metrics import (
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        )
        from datasets import Dataset
    except ImportError:
        raise ImportError(
            "RAGAS not installed. Install with: pip install ragas datasets"
        )
    
    # Set API key
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
    
    # Prepare data in RAGAS format
    data = {
        "question": [query],
        "answer": [answer],
        "contexts": [contexts],
    }
    
    # Add ground truth if available
    if ground_truth:
        data["ground_truth"] = [ground_truth]
    
    dataset = Dataset.from_dict(data)
    
    # Select metrics based on available data
    metrics_to_use = [
        faithfulness,
        answer_relevancy,
        context_precision,
    ]
    
    # Context recall requires ground truth
    if ground_truth:
        metrics_to_use.append(context_recall)
    
    # Run evaluation
    result = evaluate(dataset, metrics=metrics_to_use)
    
    # Extract scores
    scores = {}
    if hasattr(result, 'to_pandas'):
        df = result.to_pandas()
        for metric in metrics_to_use:
            metric_name = metric.name
            if metric_name in df.columns:
                scores[metric_name] = float(df[metric_name].iloc[0])
    else:
        # Fallback for different RAGAS versions
        for metric in metrics_to_use:
            metric_name = metric.name
            if metric_name in result:
                scores[metric_name] = float(result[metric_name])
    
    return scores


def calculate_rouge_scores(
    generated: str,
    reference: str
) -> Dict[str, float]:
    """Calculate ROUGE scores (optional, requires ground truth).
    
    Args:
        generated: Generated answer
        reference: Reference/ground truth answer
        
    Returns:
        Dictionary with ROUGE-1, ROUGE-2, ROUGE-L scores
    """
    try:
        from rouge_score import rouge_scorer
    except ImportError:
        raise ImportError(
            "rouge-score not installed. Install with: pip install rouge-score"
        )
    
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, generated)
    
    return {
        "rouge1": scores['rouge1'].fmeasure,
        "rouge2": scores['rouge2'].fmeasure,
        "rougeL": scores['rougeL'].fmeasure,
    }


def evaluate_generation(
    query: str,
    generated_answer: str,
    retrieved_contexts: List[str],
    ground_truth_answer: Optional[str] = None,
    api_key: Optional[str] = None,
    include_rouge: bool = False
) -> Dict[str, any]:
    """Comprehensive generation evaluation.
    
    Args:
        query: User query
        generated_answer: Generated answer from RAG system
        retrieved_contexts: List of retrieved context strings
        ground_truth_answer: Optional ground truth answer
        api_key: OpenAI API key
        include_rouge: Whether to calculate ROUGE scores (requires ground truth)
        
    Returns:
        Dictionary with all generation metrics
    """
    results = {}
    
    # RAGAS metrics (LLM-based)
    try:
        ragas_scores = calculate_ragas_metrics(
            query=query,
            answer=generated_answer,
            contexts=retrieved_contexts,
            ground_truth=ground_truth_answer,
            api_key=api_key
        )
        results["ragas"] = ragas_scores
    except Exception as e:
        results["ragas_error"] = str(e)
    
    # ROUGE scores (if ground truth available and requested)
    if include_rouge and ground_truth_answer:
        try:
            rouge_scores = calculate_rouge_scores(generated_answer, ground_truth_answer)
            results["rouge"] = rouge_scores
        except Exception as e:
            results["rouge_error"] = str(e)
    
    return results


def aggregate_generation_metrics(
    all_metrics: List[Dict[str, any]]
) -> Dict[str, any]:
    """Aggregate generation metrics across multiple queries.
    
    Args:
        all_metrics: List of metric dictionaries from evaluate_generation
        
    Returns:
        Aggregated metrics with means
    """
    if not all_metrics:
        return {}
    
    # Extract RAGAS metrics
    ragas_metrics = [m.get("ragas", {}) for m in all_metrics if "ragas" in m]
    
    if not ragas_metrics:
        return {"error": "No valid RAGAS metrics found"}
    
    # Get all metric names
    metric_names = set()
    for metrics in ragas_metrics:
        metric_names.update(metrics.keys())
    
    # Aggregate each metric
    aggregated = {
        "num_queries": len(all_metrics),
        "ragas": {}
    }
    
    for metric_name in metric_names:
        values = [m[metric_name] for m in ragas_metrics if metric_name in m]
        if values:
            aggregated["ragas"][metric_name] = {
                "mean": sum(values) / len(values),
                "values": values
            }
    
    # Aggregate ROUGE if present
    rouge_metrics = [m.get("rouge", {}) for m in all_metrics if "rouge" in m]
    if rouge_metrics:
        aggregated["rouge"] = {}
        for metric_name in ["rouge1", "rouge2", "rougeL"]:
            values = [m[metric_name] for m in rouge_metrics if metric_name in m]
            if values:
                aggregated["rouge"][metric_name] = {
                    "mean": sum(values) / len(values),
                    "values": values
                }
    
    return aggregated
