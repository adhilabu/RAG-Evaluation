"""Unit tests for RAG evaluation metrics."""
import pytest
from evaluation.retrieval_metrics import (
    calculate_precision_at_k,
    calculate_recall_at_k,
    calculate_mrr,
    calculate_ndcg_at_k,
    calculate_hit_rate_at_k,
    evaluate_retrieval,
    aggregate_metrics
)
from evaluation.evaluation_dataset import EvaluationExample, EvaluationDataset
import tempfile
import json
from pathlib import Path


class TestRetrievalMetrics:
    """Test retrieval metrics calculations."""
    
    def test_precision_at_k_perfect(self):
        """Test precision@k with perfect retrieval."""
        retrieved = ["doc1", "doc2", "doc3"]
        relevant = {"doc1", "doc2", "doc3"}
        
        assert calculate_precision_at_k(retrieved, relevant, 3) == 1.0
        assert calculate_precision_at_k(retrieved, relevant, 2) == 1.0
        assert calculate_precision_at_k(retrieved, relevant, 1) == 1.0
    
    def test_precision_at_k_partial(self):
        """Test precision@k with partial retrieval."""
        retrieved = ["doc1", "doc2", "doc3", "doc4"]
        relevant = {"doc1", "doc3"}
        
        assert calculate_precision_at_k(retrieved, relevant, 4) == 0.5
        assert calculate_precision_at_k(retrieved, relevant, 2) == 0.5
        assert calculate_precision_at_k(retrieved, relevant, 1) == 1.0
    
    def test_recall_at_k_perfect(self):
        """Test recall@k with perfect retrieval."""
        retrieved = ["doc1", "doc2", "doc3"]
        relevant = {"doc1", "doc2", "doc3"}
        
        assert calculate_recall_at_k(retrieved, relevant, 3) == 1.0
    
    def test_recall_at_k_partial(self):
        """Test recall@k with partial retrieval."""
        retrieved = ["doc1", "doc2", "doc4"]
        relevant = {"doc1", "doc2", "doc3"}
        
        # Retrieved 2 out of 3 relevant docs
        assert calculate_recall_at_k(retrieved, relevant, 3) == pytest.approx(2/3)
        assert calculate_recall_at_k(retrieved, relevant, 2) == pytest.approx(2/3)
    
    def test_mrr_first_position(self):
        """Test MRR when first result is relevant."""
        retrieved = ["doc1", "doc2", "doc3"]
        relevant = {"doc1"}
        
        assert calculate_mrr(retrieved, relevant) == 1.0
    
    def test_mrr_second_position(self):
        """Test MRR when second result is relevant."""
        retrieved = ["doc1", "doc2", "doc3"]
        relevant = {"doc2"}
        
        assert calculate_mrr(retrieved, relevant) == 0.5
    
    def test_mrr_no_relevant(self):
        """Test MRR when no relevant docs retrieved."""
        retrieved = ["doc1", "doc2", "doc3"]
        relevant = {"doc4"}
        
        assert calculate_mrr(retrieved, relevant) == 0.0
    
    def test_ndcg_perfect(self):
        """Test NDCG with perfect ranking."""
        retrieved = ["doc1", "doc2", "doc3"]
        relevance = {"doc1": 3.0, "doc2": 2.0, "doc3": 1.0}
        
        # Perfect ranking should give NDCG = 1.0
        assert calculate_ndcg_at_k(retrieved, relevance, 3) == pytest.approx(1.0)
    
    def test_ndcg_reversed(self):
        """Test NDCG with reversed ranking."""
        retrieved = ["doc3", "doc2", "doc1"]
        relevance = {"doc1": 3.0, "doc2": 2.0, "doc3": 1.0}
        
        # Reversed ranking should give lower NDCG
        ndcg = calculate_ndcg_at_k(retrieved, relevance, 3)
        assert 0.0 < ndcg < 1.0
    
    def test_hit_rate_hit(self):
        """Test hit rate when relevant doc is found."""
        retrieved = ["doc1", "doc2", "doc3"]
        relevant = {"doc2"}
        
        assert calculate_hit_rate_at_k(retrieved, relevant, 3) == 1.0
        assert calculate_hit_rate_at_k(retrieved, relevant, 2) == 1.0
    
    def test_hit_rate_miss(self):
        """Test hit rate when no relevant doc is found."""
        retrieved = ["doc1", "doc2", "doc3"]
        relevant = {"doc4"}
        
        assert calculate_hit_rate_at_k(retrieved, relevant, 3) == 0.0
    
    def test_evaluate_retrieval(self):
        """Test complete retrieval evaluation."""
        results = [
            {"document_id": "doc1"},
            {"document_id": "doc2"},
            {"document_id": "doc3"},
        ]
        relevant_ids = {"doc1", "doc3"}
        relevance_scores = {"doc1": 2.0, "doc3": 1.0}
        
        metrics = evaluate_retrieval(
            results,
            relevant_ids,
            relevance_scores,
            k_values=[1, 2, 3]
        )
        
        assert "metrics_by_k" in metrics
        assert "@1" in metrics["metrics_by_k"]
        assert "@2" in metrics["metrics_by_k"]
        assert "@3" in metrics["metrics_by_k"]
        assert "mrr" in metrics
        
        # Check specific values
        assert metrics["metrics_by_k"]["@3"]["precision"] == pytest.approx(2/3)
        assert metrics["metrics_by_k"]["@3"]["recall"] == 1.0
        assert metrics["mrr"] == 1.0  # First result is relevant


class TestEvaluationDataset:
    """Test evaluation dataset functionality."""
    
    def test_create_example(self):
        """Test creating an evaluation example."""
        example = EvaluationExample(
            query="Test query",
            relevant_doc_ids=["doc1", "doc2"],
            ground_truth_answer="Test answer"
        )
        
        assert example.query == "Test query"
        assert len(example.relevant_doc_ids) == 2
        assert example.ground_truth_answer == "Test answer"
        assert "doc1" in example.relevance_scores
        assert "doc2" in example.relevance_scores
    
    def test_example_with_graded_relevance(self):
        """Test example with graded relevance scores."""
        example = EvaluationExample(
            query="Test query",
            relevant_doc_ids=["doc1", "doc2"],
            relevance_scores={"doc1": 2.0, "doc2": 1.0}
        )
        
        assert example.relevance_scores["doc1"] == 2.0
        assert example.relevance_scores["doc2"] == 1.0
    
    def test_dataset_creation(self):
        """Test creating a dataset."""
        dataset = EvaluationDataset()
        
        example1 = EvaluationExample(
            query="Query 1",
            relevant_doc_ids=["doc1"]
        )
        example2 = EvaluationExample(
            query="Query 2",
            relevant_doc_ids=["doc2"]
        )
        
        dataset.add_example(example1)
        dataset.add_example(example2)
        
        assert len(dataset) == 2
        assert dataset[0].query == "Query 1"
        assert dataset[1].query == "Query 2"
    
    def test_dataset_json_roundtrip(self):
        """Test saving and loading dataset from JSON."""
        # Create dataset
        dataset = EvaluationDataset()
        dataset.add_example(EvaluationExample(
            query="Test query",
            relevant_doc_ids=["doc1", "doc2"],
            relevance_scores={"doc1": 2.0, "doc2": 1.0},
            ground_truth_answer="Test answer"
        ))
        
        # Save to temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            dataset.to_json(temp_path)
            
            # Load back
            loaded_dataset = EvaluationDataset.from_json(temp_path)
            
            assert len(loaded_dataset) == 1
            assert loaded_dataset[0].query == "Test query"
            assert len(loaded_dataset[0].relevant_doc_ids) == 2
            assert loaded_dataset[0].relevance_scores["doc1"] == 2.0
            assert loaded_dataset[0].ground_truth_answer == "Test answer"
        finally:
            Path(temp_path).unlink()
    
    def test_dataset_iteration(self):
        """Test iterating over dataset."""
        dataset = EvaluationDataset()
        dataset.add_example(EvaluationExample(query="Q1", relevant_doc_ids=["d1"]))
        dataset.add_example(EvaluationExample(query="Q2", relevant_doc_ids=["d2"]))
        
        queries = [ex.query for ex in dataset]
        assert queries == ["Q1", "Q2"]


class TestMetricAggregation:
    """Test metric aggregation."""
    
    def test_aggregate_metrics(self):
        """Test aggregating metrics across queries."""
        metrics1 = {
            "mrr": 1.0,
            "metrics_by_k": {
                "@5": {
                    "precision": 0.8,
                    "recall": 0.9,
                    "ndcg": 0.85,
                    "hit_rate": 1.0
                }
            }
        }
        
        metrics2 = {
            "mrr": 0.5,
            "metrics_by_k": {
                "@5": {
                    "precision": 0.6,
                    "recall": 0.7,
                    "ndcg": 0.65,
                    "hit_rate": 1.0
                }
            }
        }
        
        aggregated = aggregate_metrics([metrics1, metrics2], k_values=[5])
        
        assert aggregated["num_queries"] == 2
        assert aggregated["mrr"]["mean"] == 0.75
        assert aggregated["metrics_by_k"]["@5"]["precision"]["mean"] == 0.7
        assert aggregated["metrics_by_k"]["@5"]["recall"]["mean"] == 0.8


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
