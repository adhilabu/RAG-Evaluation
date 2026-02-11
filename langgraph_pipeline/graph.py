"""LangGraph workflow definition for document summarization."""
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from .state import DocumentState
from .nodes import distribute_chunks, map_summarize, reduce_synthesize


def create_summarization_graph(checkpointer=None):
    """
    Create the LangGraph workflow for map-reduce summarization.
    
    Args:
        checkpointer: Optional checkpointer for state persistence
        
    Returns:
        Compiled graph
    """
    # Create the graph
    workflow = StateGraph(DocumentState)
    
    # Add nodes
    workflow.add_node("distribute", distribute_chunks)
    workflow.add_node("map_summarize", map_summarize)
    workflow.add_node("reduce_synthesize", reduce_synthesize)
    
    # Define edges
    workflow.set_entry_point("distribute")
    workflow.add_edge("distribute", "map_summarize")
    workflow.add_edge("map_summarize", "reduce_synthesize")
    workflow.add_edge("reduce_synthesize", END)
    
    # Use default MemorySaver if no checkpointer provided
    if checkpointer is None:
        checkpointer = MemorySaver()
    
    # Compile the graph
    graph = workflow.compile(checkpointer=checkpointer)
    
    return graph
