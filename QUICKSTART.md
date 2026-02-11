# Quick Start Guide

## Prerequisites

Before starting, ensure you have:
- ✅ Python 3.9+ installed
- ✅ Qdrant running (confirmed via `docker ps`)
- ⚠️ OpenAI API key (you'll need to add this)

## Setup Steps

### 1. Navigate to Project

```bash
cd /Users/adhilabubacker/Projects/ai-projects/RAG
```

### 2. Create .env File

The `.env` file already exists. Add your OpenAI API key:

```bash
# Edit the .env file
nano .env

# Add your key:
OPENAI_API_KEY=sk-your-key-here

# Save and exit (Ctrl+X, Y, Enter)
```

### 3. Activate Virtual Environment

A virtual environment has been created for you:

```bash
source venv/bin/activate
```

### 4. Install Remaining Dependencies

Some dependencies may still need installation:

```bash
pip install -r requirements.txt
```

### 5. Start the Backend API

In one terminal:

```bash
cd backend
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

You should see:
```
INFO:     Uvicorn running on http://0.0.0.0:8000
INFO:     Application startup complete.
```

Visit http://localhost:8000/docs to see the API documentation.

### 6. Start the Streamlit UI

Open a **new terminal**, then:

```bash
cd /Users/adhilabubacker/Projects/ai-projects/RAG
source venv/bin/activate
cd frontend
streamlit run streamlit_app.py
```

The UI will open automatically at http://localhost:8501

## Testing the System

### Option 1: Use the Sample PDF

A 20-page sample PDF has been created at `data/sample_20page.pdf`.

### Option 2: Create a Different Size PDF

```bash
source venv/bin/activate
python tests/create_sample_pdf.py 50  # Creates 50-page PDF
```

### Test Workflow

1. **Upload Document**
   - Go to http://localhost:8501
   - Select "📤 Upload Document"
   - Upload `data/sample_20page.pdf`
   - Click "Process Document"
   - Copy the Document ID

2. **Generate Summary**
   - Go to "📝 Summarize" page
   - Select your document
   - Click "Generate Summary"
   - Wait for map-reduce process (may take 30-60 seconds)

3. **Search Documents**
   - Go to "🔍 Search Documents"
   - Enter a query like: "What are the key findings?"
   - View semantic search results

## Verify Services

### Check Qdrant

```bash
# Visit Qdrant dashboard
open http://localhost:6333/dashboard

# Or check via API
curl http://localhost:8000/api/v1/collection/info
```

### Check Backend

```bash
curl http://localhost:8000/health
# Should return: {"status":"healthy"}
```

## Common Issues

### "ModuleNotFoundError"
```bash
source venv/bin/activate
pip install -r requirements.txt
```

### "Connection refused" (Backend)
Make sure backend is running:
```bash
cd backend
python -m uvicorn app.main:app --reload
```

### "Qdrant connection error"
Verify Qdrant is running:
```bash
docker ps | grep qdrant
```

If not running:
```bash
docker start graphrag-qdrant
```

### "OpenAI API Error"
Check your `.env` file has a valid API key:
```bash
cat .env | grep OPENAI_API_KEY
```

## Next Steps

- Read the [README.md](file:///Users/adhilabubacker/Projects/ai-projects/RAG/README.md) for full documentation
- Explore the API at http://localhost:8000/docs
- Try uploading your own PDFs
- Experiment with different query types

## Project Structure Quick Reference

```
RAG/
├── backend/app/main.py       # Start backend from here
├── frontend/streamlit_app.py # Start UI from here
├── data/sample_20page.pdf    # Test PDF
├── .env                      # Add your API key here
└── venv/                     # Virtual environment
```

## Stopping the System

1. Press `Ctrl+C` in both terminal windows (backend and frontend)
2. Deactivate virtual environment: `deactivate`
3. Optionally stop Qdrant: `docker stop graphrag-qdrant`
