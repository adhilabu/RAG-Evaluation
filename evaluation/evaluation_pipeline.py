"""End-to-end RAG evaluation pipeline."""
from typing import List, Dict, Optional, Any
import json
from pathlib import Path
from datetime import datetime

from .evaluation_dataset import EvaluationDataset, EvaluationExample
from .retrieval_metrics import evaluate_retrieval, aggregate_metrics
from .generation_metrics import evaluate_generation, aggregate_generation_metrics
from rag_storage import search_documents
from rag_storage.qdrant_client import QdrantManager


class RAGEvaluator:
    """End-to-end RAG evaluation orchestrator.
    
    Handles:
    - Loading evaluation datasets
    - Running retrieval evaluation
    - Running generation evaluation (optional)
    - Aggregating results
    - Generating reports
    """
    
    def __init__(
        self,
        qdrant_manager: QdrantManager,
        api_key: str,
        llm_model: Optional[str] = None
    ):
        """Initialize evaluator.
        
        Args:
            qdrant_manager: Qdrant manager for retrieval
            api_key: OpenAI API key
            llm_model: Optional LLM model name for generation
        """
        self.qdrant_manager = qdrant_manager
        self.api_key = api_key
        self.llm_model = llm_model or "gpt-4o-mini"
    
    def evaluate_retrieval_only(
        self,
        dataset: EvaluationDataset,
        k_values: List[int] = [1, 3, 5, 10],
        score_threshold: float = 0.0,
        limit: int = 20
    ) -> Dict[str, Any]:
        """Evaluate retrieval quality only.
        
        Args:
            dataset: Evaluation dataset
            k_values: K values for metrics
            score_threshold: Minimum similarity score
            limit: Maximum results to retrieve
            
        Returns:
            Evaluation results with metrics
        """
        all_metrics = []
        detailed_results = []
        
        for i, example in enumerate(dataset):
            print(f"Evaluating query {i+1}/{len(dataset)}: {example.query[:50]}...")
            
            # Perform retrieval
            try:
                results = search_documents(
                    query=example.query,
                    qdrant_manager=self.qdrant_manager,
                    api_key=self.api_key,
                    limit=limit,
                    score_threshold=score_threshold
                )
            except Exception as e:
                print(f"  Error during retrieval: {e}")
                detailed_results.append({
                    "query": example.query,
                    "error": str(e)
                })
                continue
            
            # Evaluate retrieval
            metrics = evaluate_retrieval(
                retrieved_results=results,
                ground_truth_relevant_ids=set(example.relevant_doc_ids),
                ground_truth_relevance_scores=example.relevance_scores,
                k_values=k_values
            )
            
            all_metrics.append(metrics)
            
            # Store detailed results
            detailed_results.append({
                "query": example.query,
                "retrieved_count": len(results),
                "relevant_count": len(example.relevant_doc_ids),
                "metrics": metrics,
                "retrieved_ids": [r.get('document_id') or r.get('id') for r in results[:10]],
                "ground_truth_ids": example.relevant_doc_ids
            })
        
        # Aggregate metrics
        aggregated = aggregate_metrics(all_metrics, k_values=k_values)
        
        return {
            "evaluation_type": "retrieval_only",
            "num_queries": len(dataset),
            "k_values": k_values,
            "aggregated_metrics": aggregated,
            "detailed_results": detailed_results,
            "timestamp": datetime.now().isoformat()
        }
    
    def evaluate_end_to_end(
        self,
        dataset: EvaluationDataset,
        k_values: List[int] = [1, 3, 5, 10],
        score_threshold: float = 0.0,
        retrieval_limit: int = 5,
        include_generation: bool = True,
        include_rouge: bool = False
    ) -> Dict[str, Any]:
        """Evaluate both retrieval and generation.
        
        Args:
            dataset: Evaluation dataset
            k_values: K values for retrieval metrics
            score_threshold: Minimum similarity score
            retrieval_limit: Number of contexts to retrieve
            include_generation: Whether to evaluate generation quality
            include_rouge: Whether to calculate ROUGE scores
            
        Returns:
            Complete evaluation results
        """
        retrieval_metrics_list = []
        generation_metrics_list = []
        detailed_results = []
        
        for i, example in enumerate(dataset):
            print(f"Evaluating query {i+1}/{len(dataset)}: {example.query[:50]}...")
            
            # Perform retrieval
            try:
                results = search_documents(
                    query=example.query,
                    qdrant_manager=self.qdrant_manager,
                    api_key=self.api_key,
                    limit=retrieval_limit,
                    score_threshold=score_threshold
                )
            except Exception as e:
                print(f"  Error during retrieval: {e}")
                detailed_results.append({
                    "query": example.query,
                    "error": str(e)
                })
                continue
            
            # Evaluate retrieval
            retrieval_metrics = evaluate_retrieval(
                retrieved_results=results,
                ground_truth_relevant_ids=set(example.relevant_doc_ids),
                ground_truth_relevance_scores=example.relevance_scores,
                k_values=k_values
            )
            retrieval_metrics_list.append(retrieval_metrics)
            
            result_entry = {
                "query": example.query,
                "retrieval_metrics": retrieval_metrics,
                "retrieved_count": len(results)
            }
            
            # Evaluate generation if requested
            if include_generation and results:
                try:
                    # Extract context texts
                    contexts = [r.get('text', '') for r in results]
                    
                    # For now, we'll use a simple concatenation as the "generated answer"
                    # In a real scenario, you'd call your LLM here
                    generated_answer = self._generate_answer(example.query, contexts)
                    
                    generation_metrics = evaluate_generation(
                        query=example.query,
                        generated_answer=generated_answer,
                        retrieved_contexts=contexts,
                        ground_truth_answer=example.ground_truth_answer,
                        api_key=self.api_key,
                        include_rouge=include_rouge
                    )
                    
                    generation_metrics_list.append(generation_metrics)
                    result_entry["generation_metrics"] = generation_metrics
                    result_entry["generated_answer"] = generated_answer[:200] + "..."
                    
                except Exception as e:
                    print(f"  Error during generation evaluation: {e}")
                    result_entry["generation_error"] = str(e)
            
            detailed_results.append(result_entry)
        
        # Aggregate metrics
        aggregated_retrieval = aggregate_metrics(retrieval_metrics_list, k_values=k_values)
        
        results = {
            "evaluation_type": "end_to_end",
            "num_queries": len(dataset),
            "k_values": k_values,
            "aggregated_retrieval_metrics": aggregated_retrieval,
            "detailed_results": detailed_results,
            "timestamp": datetime.now().isoformat()
        }
        
        if generation_metrics_list:
            aggregated_generation = aggregate_generation_metrics(generation_metrics_list)
            results["aggregated_generation_metrics"] = aggregated_generation
        
        return results
    
    def _generate_answer(self, query: str, contexts: List[str]) -> str:
        """Generate answer using LLM (placeholder for now).
        
        Args:
            query: User query
            contexts: Retrieved contexts
            
        Returns:
            Generated answer
        """
        # TODO: Implement actual LLM generation
        # For now, return a simple concatenation
        from langchain_openai import ChatOpenAI
        from langchain_core.prompts import ChatPromptTemplate
        
        llm = ChatOpenAI(
            model=self.llm_model,
            temperature=0.3,
            api_key=self.api_key
        )
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant. Answer the question based on the provided context."),
            ("user", """Context:
{context}

Question: {question}

Answer:""")
        ])
        
        chain = prompt | llm
        
        context_text = "\n\n".join(contexts)
        response = chain.invoke({
            "context": context_text,
            "question": query
        })
        
        return response.content
    
    @staticmethod
    def _nan_to_none(obj):
        """Convert NaN values to None for JSON serialization.
        
        Args:
            obj: Object to convert
            
        Returns:
            Converted object with NaN replaced by None
        """
        import math
        
        if isinstance(obj, float) and math.isnan(obj):
            return None
        elif isinstance(obj, dict):
            return {k: RAGEvaluator._nan_to_none(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [RAGEvaluator._nan_to_none(item) for item in obj]
        return obj
    
    def generate_report(
        self,
        results: Dict[str, Any],
        output_path: str
    ):
        """Generate evaluation report.
        
        Args:
            results: Evaluation results
            output_path: Directory to save report
        """
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save full results as JSON
        json_path = output_dir / "results.json"
        with open(json_path, 'w') as f:
            # Convert NaN to None for valid JSON
            cleaned_results = self._nan_to_none(results)
            json.dump(cleaned_results, f, indent=2)
        
        # Generate markdown report
        report_path = output_dir / "report.md"
        with open(report_path, 'w') as f:
            f.write(self._format_markdown_report(results))
        
        print(f"\n✅ Report generated:")
        print(f"  - JSON: {json_path}")
        print(f"  - Markdown: {report_path}")
    
    def _format_markdown_report(self, results: Dict[str, Any]) -> str:
        """Format results as markdown report.
        
        Args:
            results: Evaluation results
            
        Returns:
            Markdown formatted report
        """
        lines = []
        lines.append("# RAG Evaluation Report")
        lines.append(f"\n**Evaluation Type**: {results['evaluation_type']}")
        lines.append(f"**Timestamp**: {results['timestamp']}")
        lines.append(f"**Number of Queries**: {results['num_queries']}")
        lines.append("")
        
        # Retrieval metrics
        if "aggregated_retrieval_metrics" in results:
            lines.append("## Retrieval Metrics")
            lines.append("")
            
            agg = results["aggregated_retrieval_metrics"]
            
            # MRR
            if "mrr" in agg:
                lines.append(f"**Mean Reciprocal Rank (MRR)**: {agg['mrr']['mean']:.4f}")
                lines.append("")
            
            # Metrics by K
            if "metrics_by_k" in agg:
                lines.append("### Metrics by K")
                lines.append("")
                lines.append("| K | Precision | Recall | NDCG | Hit Rate |")
                lines.append("|---|-----------|--------|------|----------|")
                
                for k_key in sorted(agg["metrics_by_k"].keys()):
                    metrics = agg["metrics_by_k"][k_key]
                    k_val = k_key.replace("@", "")
                    lines.append(
                        f"| {k_val} | "
                        f"{metrics['precision']['mean']:.4f} | "
                        f"{metrics['recall']['mean']:.4f} | "
                        f"{metrics['ndcg']['mean']:.4f} | "
                        f"{metrics['hit_rate']['mean']:.4f} |"
                    )
                lines.append("")
        
        # Generation metrics
        if "aggregated_generation_metrics" in results:
            lines.append("## Generation Metrics (RAGAS)")
            lines.append("")
            
            agg_gen = results["aggregated_generation_metrics"]
            
            if "ragas" in agg_gen:
                for metric_name, metric_data in agg_gen["ragas"].items():
                    lines.append(f"**{metric_name.replace('_', ' ').title()}**: {metric_data['mean']:.4f}")
                lines.append("")
            
            if "rouge" in agg_gen:
                lines.append("### ROUGE Scores")
                lines.append("")
                for metric_name, metric_data in agg_gen["rouge"].items():
                    lines.append(f"**{metric_name.upper()}**: {metric_data['mean']:.4f}")
                lines.append("")
        
        # Detailed results summary
        lines.append("## Query-Level Results")
        lines.append("")
        
        for i, result in enumerate(results.get("detailed_results", [])[:10], 1):
            lines.append(f"### Query {i}")
            lines.append(f"**Query**: {result.get('query', 'N/A')[:100]}...")
            
            if "retrieval_metrics" in result:
                metrics_5 = result["retrieval_metrics"].get("metrics_by_k", {}).get("@5", {})
                lines.append(f"- Precision@5: {metrics_5.get('precision', 0):.4f}")
                lines.append(f"- Recall@5: {metrics_5.get('recall', 0):.4f}")
            
            if "generation_metrics" in result and "ragas" in result["generation_metrics"]:
                ragas = result["generation_metrics"]["ragas"]
                if "faithfulness" in ragas:
                    lines.append(f"- Faithfulness: {ragas['faithfulness']:.4f}")
            
            lines.append("")
        
        return "\n".join(lines)
