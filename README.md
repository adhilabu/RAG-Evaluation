# Document Processing System

A production-grade system for processing large documents (500+ pages) with parallel pipelines for **Map-Reduce Summarization** and **RAG Storage** using LangGraph, FastAPI, and Streamlit.

## 🏗️ Architecture

### Dual Pipeline System

1. **Summarization Pipeline** (LangGraph Map-Reduce)
   - Extract and clean PDF text
   - Split into large chunks (10k-20k tokens)
   - **Map**: Parallel LLM summarization of each chunk
   - **Reduce**: Synthesize final cohesive summary

2. **Storage Pipeline** (RAG with Qdrant)
   - Split into small chunks (512-1024 tokens)
   - Generate embeddings (OpenAI text-embedding-3-small)
   - Store in Qdrant vector database
   - Enable semantic search

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- Docker (for Qdrant)
- OpenAI API key

### 1. Start Qdrant (if not running)

```bash
docker run -p 6333:6333 -p 6334:6334 \
    -v $(pwd)/qdrant_storage:/qdrant/storage:z \
    qdrant/qdrant:v1.12.0
```

### 2. Install Dependencies

```bash
cd /Users/adhilabubacker/Projects/ai-projects/RAG
pip install -r requirements.txt
```

### 3. Configure Environment

```bash
cp .env.example .env
# Edit .env and add your OpenAI API key
```

### 4. Start Backend API

```bash
cd backend
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

API will be available at: http://localhost:8000
API docs: http://localhost:8000/docs

### 5. Start Streamlit UI

```bash
# In a new terminal
cd frontend
streamlit run streamlit_app.py
```

UI will be available at: http://localhost:8501

## 📁 Project Structure

```
RAG/
├── backend/               # FastAPI application
│   └── app/
│       ├── main.py       # FastAPI app
│       ├── config.py     # Configuration
│       └── api/          # API endpoints
│           ├── upload.py      # Document upload
│           ├── summarize.py   # Summarization
│           └── query.py       # Search
├── document_processor/    # PDF extraction & chunking
│   ├── extractor.py      # PyMuPDF extraction
│   ├── cleaner.py        # Text cleaning
│   └── chunker.py        # Dual chunking strategy
├── langgraph_pipeline/    # LangGraph map-reduce
│   ├── state.py          # State schema
│   ├── nodes.py          # Map/Reduce nodes
│   └── graph.py          # Workflow definition
├── rag_storage/          # Vector database operations
│   ├── qdrant_client.py  # Qdrant manager
│   ├── embeddings.py     # Embedding generation
│   └── retrieval.py      # Search functions
├── evaluation/           # RAG evaluation framework
│   ├── evaluation_dataset.py    # Dataset structures
│   ├── retrieval_metrics.py     # Retrieval metrics
│   ├── generation_metrics.py    # Generation metrics
│   ├── evaluation_pipeline.py   # Evaluation orchestrator
│   └── visualizations.py        # Result visualizations
├── scripts/              # Utility scripts
│   ├── run_evaluation.py        # Main evaluation runner
│   └── create_evaluation_dataset.py  # Dataset creator
├── frontend/             # Streamlit UI
│   └── streamlit_app.py  # Main UI
├── data/                 # Uploaded PDFs storage
├── evaluation_datasets/  # Evaluation datasets
└── requirements.txt      # Python dependencies
```

## 🎯 Usage

### 1. Upload a Document

1. Go to **Upload Document** page
2. Select a PDF file
3. Click **Process Document**
4. System will:
   - Extract text from PDF
   - Create small chunks for RAG (stored in Qdrant)
   - Create large chunks for summarization
   - Generate embeddings
   - Return a document ID

### 2. Generate Summary

1. Go to **Summarize** page
2. Select your document
3. Click **Generate Summary**
4. LangGraph will:
   - Distribute chunks to parallel workers
   - Summarize each chunk independently
   - Synthesize all summaries into final summary

### 3. Search Documents

1. Go to **Search Documents** page
2. Enter your query
3. Adjust settings (number of results, similarity threshold)
4. Optionally filter by specific document
5. View results with similarity scores

## 🔧 API Endpoints

### Upload
- `POST /api/v1/upload` - Upload and process PDF
- `GET /api/v1/documents` - List all documents
- `GET /api/v1/documents/{doc_id}` - Get document details

### Summarization
- `POST /api/v1/summarize` - Trigger summarization
- `GET /api/v1/summarize/{doc_id}/status` - Get summary status

### Search
- `POST /api/v1/query` - Semantic search
- `GET /api/v1/collection/info` - Qdrant collection info

## 🧪 Testing

```bash
# Test document processing
python -c "
from document_processor import extract_pdf_text, clean_pages, create_rag_chunks
pages = extract_pdf_text('data/sample.pdf')
cleaned = clean_pages(pages)
chunks = create_rag_chunks(cleaned, document_id='test')
print(f'Created {len(chunks)} chunks')
"

# Test Qdrant connection
python -c "
from rag_storage import QdrantManager
qm = QdrantManager()
print(qm.get_collection_info())
"
```

## 🛠️ Technology Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **PDF Extraction** | PyMuPDF | Fast text extraction |
| **Text Processing** | LangChain | Smart text splitting |
| **Orchestration** | LangGraph | Map-reduce workflow |
| **LLM** | OpenAI GPT-4o-mini | Summarization |
| **Embeddings** | OpenAI text-embedding-3-small | Vector generation |
| **Vector DB** | Qdrant | Semantic search |
| **Backend** | FastAPI | REST API |
| **Frontend** | Streamlit | User interface |

## 📊 Key Features

✅ **Intelligent Chunking**: Dual strategy for RAG vs Summarization
✅ **Parallel Processing**: LangGraph async map-reduce
✅ **Semantic Search**: Vector similarity with Qdrant
✅ **Metadata Tracking**: Page numbers, chunk indices
✅ **Production Ready**: Error handling, validation
✅ **Interactive UI**: Real-time progress tracking

## 🔐 Environment Variables

Required in `.env`:

```bash
OPENAI_API_KEY=sk-...           # Required
QDRANT_HOST=localhost            # Default
QDRANT_PORT=6333                 # Default
```

## 📝 Notes

- **Maximum File Size**: 50MB (configurable in `config.py`)
- **Embedding Dimensions**: 1536 (text-embedding-3-small)
- **RAG Chunk Size**: 1000 characters (~250 tokens)
- **Summary Chunk Size**: 15000 characters (~3750 tokens)
- **Default LLM**: GPT-4o-mini

## 🐛 Troubleshooting

### API won't start
- Check if port 8000 is available
- Verify `.env` file exists with OpenAI API key

### Qdrant connection error
- Ensure Qdrant Docker container is running: `docker ps`
- Check Qdrant dashboard: http://localhost:6333/dashboard

### Summarization fails
- Check OpenAI API key is valid
- Verify document has been uploaded first
- Check backend logs for detailed errors

## 📊 RAG Evaluation

The system includes a comprehensive evaluation framework to measure and analyze RAG performance.

### Evaluation Metrics

#### Retrieval Metrics
- **Precision@K**: Proportion of retrieved documents that are relevant
- **Recall@K**: Proportion of relevant documents that were retrieved
- **MRR (Mean Reciprocal Rank)**: Average reciprocal rank of first relevant result
- **NDCG@K**: Normalized Discounted Cumulative Gain (handles graded relevance)
- **Hit Rate@K**: Whether at least one relevant document appears in top K

#### Generation Metrics (RAGAS)
- **Faithfulness**: Whether generated answer is grounded in retrieved context
- **Answer Relevancy**: How relevant the answer is to the query
- **Context Precision**: How relevant the retrieved contexts are
- **Context Recall**: Whether all relevant information was retrieved

### Creating Evaluation Datasets

Create evaluation datasets with queries and ground truth:

```bash
python scripts/create_evaluation_dataset.py --output evaluation_datasets/my_eval.json
```

Or manually create a JSON file:

```json
{
  "examples": [
    {
      "query": "What is the main topic?",
      "relevant_doc_ids": ["doc1_chunk_0", "doc1_chunk_1"],
      "relevance_scores": {"doc1_chunk_0": 2.0, "doc1_chunk_1": 1.0},
      "ground_truth_answer": "Optional expected answer"
    }
  ]
}
```

### Running Evaluations

**Retrieval-only evaluation**:
```bash
python scripts/run_evaluation.py \
  --dataset evaluation_datasets/sample_eval.json \
  --output evaluation_results/run_001 \
  --k-values 1,3,5,10
```

**End-to-end evaluation** (includes generation quality):
```bash
python scripts/run_evaluation.py \
  --dataset evaluation_datasets/sample_eval.json \
  --output evaluation_results/run_001 \
  --k-values 1,3,5,10 \
  --include-generation
```

### Evaluation Outputs

Results are saved to the specified output directory:
- `results.json` - Complete evaluation results
- `report.md` - Human-readable markdown report
- `metrics_by_k.png` - Visualization of metrics across K values
- `score_distribution.png` - Distribution of scores
- `ragas_metrics.png` - Generation quality metrics (if applicable)
- `results.csv` - Results in CSV format

### Interpreting Results

**Good Retrieval Performance**:
- Precision@5 > 0.6
- Recall@5 > 0.7
- NDCG@5 > 0.7
- MRR > 0.5

**Good Generation Performance**:
- Faithfulness > 0.7
- Answer Relevancy > 0.7
- Context Precision > 0.6



## 🚧 Future Enhancements

- [ ] Add Neo4j GraphRAG integration
- [ ] Implement reranking (Cohere/Cross-encoder)
- [ ] Add PostgreSQL checkpointer for LangGraph
- [ ] Support more file formats (DOCX, TXT)
- [ ] Add authentication and user management
- [ ] Deploy with Docker Compose

## 📄 License

MIT

## 👨‍💻 Author

Built as a demonstration of production-grade RAG and Map-Reduce summarization systems.
