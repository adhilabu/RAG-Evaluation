"""LangGraph nodes for map-reduce summarization."""
from typing import List, Dict
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import asyncio
from .state import DocumentState


def distribute_chunks(state: DocumentState) -> DocumentState:
    """
    Distributor node: Prepare chunks for parallel summarization.
    
    This is the entry point that validates the chunks are ready.
    """
    state["status"] = "mapping"
    state["total_chunks"] = len(state["large_chunks"])
    state["chunk_summaries"] = []
    state["summaries_completed"] = 0
    
    return state


async def summarize_chunk_async(
    chunk: Dict[str, any],
    llm: ChatOpenAI,
    total_chunks: int,
) -> str:
    """
    Summarize a single chunk asynchronously.
    
    Args:
        chunk: Chunk dictionary with text and metadata
        llm: Language model instance
        total_chunks: Total number of chunks (for context)
        
    Returns:
        Summary text
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert at summarizing sections of large documents.
Your summaries should be:
- Dense with information but readable
- Focused on key facts, figures, dates, and entities
- Coherent and well-structured
- Maximum 500 words"""),
        ("user", """Summarize the following section of a large document.

SECTION CONTEXT:
- Section {chunk_index} of {total_chunks}
- Page range: {page_range}
- Approximate length: {char_count} characters

SECTION TEXT:
{chunk_text}

Provide a concise but comprehensive summary.""")
    ])
    
    chain = prompt | llm
    
    response = await chain.ainvoke({
        "chunk_index": chunk["chunk_index"] + 1,  # 1-indexed for readability
        "total_chunks": total_chunks,
        "page_range": chunk.get("page_range", "Unknown"),
        "char_count": chunk["char_count"],
        "chunk_text": chunk["text"],
    })
    
    return response.content


async def map_summarize(state: DocumentState, llm: ChatOpenAI = None) -> DocumentState:
    """
    Map node: Summarize all chunks in parallel.
    
    Args:
        state: Current state
        llm: Optional LLM instance (if None, creates default)
        
    Returns:
        Updated state with chunk_summaries
    """
    if llm is None:
        from backend.app.config import get_settings
        settings = get_settings()
        llm = ChatOpenAI(
            model=settings.llm_model,
            temperature=0.3,
            api_key=settings.openai_api_key,
        )
    
    chunks = state["large_chunks"]
    total_chunks = state["total_chunks"]
    
    # Run summarization in parallel
    tasks = [
        summarize_chunk_async(chunk, llm, total_chunks)
        for chunk in chunks
    ]
    summaries = await asyncio.gather(*tasks)
    
    state["chunk_summaries"] = summaries
    state["summaries_completed"] = len(summaries)
    state["status"] = "reducing"
    
    return state


def reduce_synthesize(state: DocumentState, llm: ChatOpenAI = None) -> DocumentState:
    """
    Reduce node: Synthesize all chunk summaries into final summary.
    
    Args:
        state: Current state with chunk_summaries
        llm: Optional LLM instance
        
    Returns:
        Updated state with final_summary
    """
    if llm is None:
        from backend.app.config import get_settings
        settings = get_settings()
        llm = ChatOpenAI(
            model=settings.llm_model,
            temperature=0.3,
            api_key=settings.openai_api_key,
        )
    
    summaries = state["chunk_summaries"]
    metadata = state["document_metadata"]
    
    # Concatenate all summaries with section markers
    concatenated = "\n\n---\n\n".join([
        f"SECTION {i+1} SUMMARY:\n{summary}"
        for i, summary in enumerate(summaries)
    ])
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert at synthesizing multiple summaries into a coherent executive summary.

Your task is to:
1. Read all section summaries carefully
2. Identify main themes and key information
3. Resolve any redundancies across sections
4. Create a well-structured final summary that flows naturally
5. Maintain all critical facts, figures, and dates
6. Ensure the summary is complete but concise (maximum 2000 words)"""),
        ("user", """Synthesize the following section summaries into a final executive summary.

DOCUMENT METADATA:
- Title: {title}
- Total Pages: {page_count}
- Number of Sections: {section_count}

SECTION SUMMARIES:
{concatenated_summaries}

Create a comprehensive executive summary that captures the essence of this {page_count}-page document.""")
    ])
    
    chain = prompt | llm
    
    response = chain.invoke({
        "title": metadata.get("title", "Unknown Document"),
        "page_count": metadata.get("page_count", "Unknown"),
        "section_count": len(summaries),
        "concatenated_summaries": concatenated,
    })
    
    state["final_summary"] = response.content
    state["status"] = "complete"
    
    return state
