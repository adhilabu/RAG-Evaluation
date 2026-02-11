"""Streamlit UI for Document Processing System."""
import streamlit as st
import requests
from pathlib import Path

# API Configuration
API_URL = "http://localhost:8085/api/v1"

st.set_page_config(
    page_title="Document Processing System",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
        font-weight: bold;
    }
    .success-box {
        padding: 1rem;
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 5px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Main page
st.markdown('<div class="main-header">📚 Document Processing System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Map-Reduce Summarization + RAG Storage</div>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("Navigation")
    page = st.radio(
        "Select Page",
        ["📤 Upload Document", "📝 Summarize", "🔍 Search Documents"],
    )
    
    st.divider()
    
    # System status
    st.subheader("System Status")
    try:
        response = requests.get(f"{API_URL.replace('/api/v1', '')}/health")
        if response.status_code == 200:
            st.success("✅ API: Online")
        else:
            st.error("❌ API: Offline")
    except:
        st.error("❌ API: Offline")
    
    try:
        response = requests.get(f"{API_URL}/collection/info")
        if response.status_code == 200:
            info = response.json()
            st.success(f"✅ Qdrant: {info.get('points_count', 0)} chunks")
        else:
            st.warning("⚠️ Qdrant: Unknown")
    except:
        st.warning("⚠️ Qdrant: Offline")

# Page content
if page == "📤 Upload Document":
    st.header("Upload Document")
    
    st.write("""
    Upload a PDF document to process it through our dual pipeline:
    - **RAG Storage**: Small chunks indexed in Qdrant for semantic search
    - **Summarization**: Large chunks prepared for map-reduce summarization
    """)
    
    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=['pdf'],
        help="Upload a PDF document (up to 50MB)"
    )
    
    if uploaded_file is not None:
        # Show file details
        st.info(f"📄 **File**: {uploaded_file.name} ({uploaded_file.size / 1024:.2f} KB)")
        
        if st.button("🚀 Process Document", type="primary"):
            with st.spinner("Processing document..."):
                try:
                    files = {"file": (uploaded_file.name, uploaded_file, "application/pdf")}
                    response = requests.post(f"{API_URL}/upload", files=files)
                    
                    if response.status_code == 200:
                        result = response.json()
                        
                        st.success("✅ Document processed successfully!")
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Document ID", result["document_id"][:8] + "...")
                        with col2:
                            st.metric("Pages", result["page_count"])
                        with col3:
                            st.metric("RAG Chunks", result["rag_chunks"])
                        
                        st.info(f"""
                        **Next Steps:**
                        1. Go to the **Summarize** page to generate a summary
                        2. Go to the **Search** page to query the document
                        
                        **Document ID**: `{result["document_id"]}`
                        """)
                        
                        # Store document ID in session state
                        st.session_state.last_uploaded_doc_id = result["document_id"]
                        
                    else:
                        st.error(f"Error: {response.text}")
                        
                except Exception as e:
                    st.error(f"Error processing document: {str(e)}")
    
    # List existing documents
    st.divider()
    st.subheader("Existing Documents")
    
    try:
        response = requests.get(f"{API_URL}/documents")
        if response.status_code == 200:
            data = response.json()
            documents = data.get("documents", [])
            
            if documents:
                for doc in documents:
                    with st.expander(f"📄 {doc['filename']}"):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write(f"**ID**: `{doc['document_id'][:16]}...`")
                            st.write(f"**Pages**: {doc['page_count']}")
                        with col2:
                            st.write(f"**Uploaded**: {doc['uploaded_at'][:19]}")
                            st.write(f"**Status**: {doc['status']}")
            else:
                st.info("No documents uploaded yet")
        else:
            st.error("Error loading documents")
    except:
        st.error("Could not connect to API")

elif page == "📝 Summarize":
    st.header("Document Summarization")
    
    st.write("""
    Generate a comprehensive summary using the **Map-Reduce** approach:
    1. **Map**: Parallel summarization of document sections
    2. **Reduce**: Synthesis into final executive summary
    """)
    
    # Get list of documents
    try:
        response = requests.get(f"{API_URL}/documents")
        if response.status_code == 200:
            documents = response.json().get("documents", [])
            
            if documents:
                # Document selector
                doc_options = {f"{doc['filename']} ({doc['document_id'][:8]}...)": doc['document_id'] 
                              for doc in documents}
                
                selected_doc_name = st.selectbox(
                    "Select Document",
                    options=list(doc_options.keys()),
                )
                
                document_id = doc_options[selected_doc_name]
                
                # Show document details
                doc_details = next(d for d in documents if d['document_id'] == document_id)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Pages", doc_details['page_count'])
                with col2:
                    st.metric("Status", doc_details['status'])
                with col3:
                    st.metric("Has Summary", "✅" if doc_details['has_summary'] else "❌")
                
                # Summarize button
                if st.button("🧠 Generate Summary", type="primary"):
                    with st.spinner("Running map-reduce summarization..."):
                        try:
                            response = requests.post(
                                f"{API_URL}/summarize",
                                json={"document_id": document_id}
                            )
                            
                            if response.status_code == 200:
                                result = response.json()
                                
                                st.success("✅ Summarization complete!")
                                st.metric("Chunks Processed", result.get("chunks_processed", 0))
                                
                                st.divider()
                                st.subheader("Executive Summary")
                                st.markdown(result["summary"])
                                
                            else:
                                st.error(f"Error: {response.text}")
                                
                        except Exception as e:
                            st.error(f"Error during summarization: {str(e)}")
                
                # Show existing summary if available
                if doc_details['has_summary']:
                    st.divider()
                    st.subheader("Existing Summary")
                    
                    try:
                        response = requests.get(f"{API_URL}/summarize/{document_id}/status")
                        if response.status_code == 200:
                            data = response.json()
                            if data.get('summary'):
                                st.markdown(data['summary'])
                    except:
                        st.error("Could not load summary")
                        
            else:
                st.info("📤 No documents available. Upload a document first!")
                
    except:
        st.error("Could not connect to API")

elif page == "🔍 Search Documents":
    st.header("Semantic Search")
    
    st.write("""
    Search across all document chunks using **semantic similarity**.
    The system uses embeddings to find the most relevant content.
    """)
    
    # Search query
    query = st.text_input(
        "Enter your search query",
        placeholder="e.g., What are the main findings?",
    )
    
    col1, col2 = st.columns(2)
    with col1:
        limit = st.slider("Number of results", min_value=1, max_value=10, value=5)
    with col2:
        score_threshold = st.slider("Minimum similarity score", min_value=0.0, max_value=1.0, value=0.5, step=0.1)
    
    # Optional document filter
    try:
        response = requests.get(f"{API_URL}/documents")
        if response.status_code == 200:
            documents = response.json().get("documents", [])
            
            if documents:
                doc_options = {"All Documents": None}
                doc_options.update({
                    f"{doc['filename']}": doc['document_id'] 
                    for doc in documents
                })
                
                selected_doc = st.selectbox("Filter by document", options=list(doc_options.keys()))
                document_id = doc_options[selected_doc]
            else:
                document_id = None
    except:
        document_id = None
    
    if st.button("🔍 Search", type="primary") and query:
        with st.spinner("Searching..."):
            try:
                response = requests.post(
                    f"{API_URL}/query",
                    json={
                        "query": query,
                        "limit": limit,
                        "document_id": document_id,
                        "score_threshold": score_threshold,
                    }
                )
                
                if response.status_code == 200:
                    data = response.json()
                    results = data.get("results", [])
                    
                    st.success(f"Found {len(results)} results")
                    
                    for i, result in enumerate(results, 1):
                        with st.expander(f"Result {i} - Score: {result['score']:.3f} (Page {result['page_number']})"):
                            st.write(f"**Document ID**: `{result['document_id'][:16]}...`")
                            st.write(f"**Page**: {result['page_number']}")
                            st.write(f"**Similarity Score**: {result['score']:.3f}")
                            st.divider()
                            st.markdown(result['text'])
                            
                else:
                    st.error(f"Error: {response.text}")
                    
            except Exception as e:
                st.error(f"Error during search: {str(e)}")
    
    elif not query:
        st.info("👆 Enter a search query above")
