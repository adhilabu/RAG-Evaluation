"""Evaluation dataset structures and loaders."""
from dataclasses import dataclass, field
from typing import List, Dict, Optional
import json
from pathlib import Path


@dataclass
class EvaluationExample:
    """Single evaluation example with query and ground truth.
    
    Attributes:
        query: The search query
        relevant_doc_ids: List of document chunk IDs that are relevant
        relevance_scores: Optional graded relevance scores (doc_id -> score)
        ground_truth_answer: Optional expected answer for generation evaluation
        metadata: Additional metadata
    """
    query: str
    relevant_doc_ids: List[str]
    relevance_scores: Optional[Dict[str, float]] = None
    ground_truth_answer: Optional[str] = None
    metadata: Dict[str, any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate the evaluation example."""
        if not self.query:
            raise ValueError("Query cannot be empty")
        if not self.relevant_doc_ids:
            raise ValueError("Must have at least one relevant document ID")
        
        # If relevance scores not provided, default to binary (all relevant docs = 1)
        if self.relevance_scores is None:
            self.relevance_scores = {doc_id: 1.0 for doc_id in self.relevant_doc_ids}
        
        # Validate that all relevant_doc_ids have scores
        for doc_id in self.relevant_doc_ids:
            if doc_id not in self.relevance_scores:
                self.relevance_scores[doc_id] = 1.0


class EvaluationDataset:
    """Manage evaluation datasets for RAG systems.
    
    Supports loading from JSON files with the following format:
    {
        "examples": [
            {
                "query": "What is X?",
                "relevant_doc_ids": ["doc1_chunk_1", "doc1_chunk_2"],
                "relevance_scores": {"doc1_chunk_1": 2, "doc1_chunk_2": 1},
                "ground_truth_answer": "X is..."
            }
        ]
    }
    """
    
    def __init__(self, examples: Optional[List[EvaluationExample]] = None):
        """Initialize dataset.
        
        Args:
            examples: List of evaluation examples
        """
        self.examples = examples or []
    
    @classmethod
    def from_json(cls, filepath: str) -> "EvaluationDataset":
        """Load evaluation dataset from JSON file.
        
        Args:
            filepath: Path to JSON file
            
        Returns:
            EvaluationDataset instance
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If JSON format is invalid
        """
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Evaluation dataset not found: {filepath}")
        
        with open(path, 'r') as f:
            data = json.load(f)
        
        if "examples" not in data:
            raise ValueError("JSON must contain 'examples' key")
        
        examples = []
        for i, example_data in enumerate(data["examples"]):
            try:
                example = EvaluationExample(
                    query=example_data["query"],
                    relevant_doc_ids=example_data["relevant_doc_ids"],
                    relevance_scores=example_data.get("relevance_scores"),
                    ground_truth_answer=example_data.get("ground_truth_answer"),
                    metadata=example_data.get("metadata", {}),
                )
                examples.append(example)
            except KeyError as e:
                raise ValueError(f"Example {i} missing required field: {e}")
            except Exception as e:
                raise ValueError(f"Error parsing example {i}: {e}")
        
        return cls(examples=examples)
    
    def to_json(self, filepath: str):
        """Save evaluation dataset to JSON file.
        
        Args:
            filepath: Path to save JSON file
        """
        data = {
            "examples": [
                {
                    "query": ex.query,
                    "relevant_doc_ids": ex.relevant_doc_ids,
                    "relevance_scores": ex.relevance_scores,
                    "ground_truth_answer": ex.ground_truth_answer,
                    "metadata": ex.metadata,
                }
                for ex in self.examples
            ]
        }
        
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def add_example(self, example: EvaluationExample):
        """Add an evaluation example to the dataset.
        
        Args:
            example: EvaluationExample to add
        """
        self.examples.append(example)
    
    def __len__(self) -> int:
        """Return number of examples in dataset."""
        return len(self.examples)
    
    def __getitem__(self, idx: int) -> EvaluationExample:
        """Get example by index."""
        return self.examples[idx]
    
    def __iter__(self):
        """Iterate over examples."""
        return iter(self.examples)
