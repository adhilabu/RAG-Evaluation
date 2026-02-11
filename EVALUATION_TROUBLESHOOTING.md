# Fixing RAG Evaluation Issues

## Issues Identified

### 1. ❌ Document ID Mismatch (Main Issue)

**Problem**: Your evaluation dataset uses semantic chunk IDs like:
- `resume_chunk_experience_1`
- `resume_chunk_skills`
- `resume_chunk_education`

But Qdrant stores documents with **UUID chunk IDs** like:
- `8698a711-f896-48e5-ae4e-2c481c93dee9`
- `5e512a55-51ba-4409-ad47-9287e13c4ab6`

**Result**: All metrics are 0 because no retrieved IDs match the ground truth IDs.

**Solution**: Use the actual UUID chunk IDs from Qdrant in your evaluation dataset.

### 2. ✅ Division by Zero in Visualizations (FIXED)

**Problem**: When all metrics are 0, the visualization code tried to divide by zero.

**Solution**: Added safety checks to skip visualization when all scores are zero.

### 3. ⚠️  RAGAS Compatibility Issue

**Problem**: `AttributeError: 'OpenAIEmbeddings' object has no attribute 'embed_query'`

This is a version compatibility issue between RAGAS and LangChain's OpenAI embeddings.

**Solution**: This doesn't affect retrieval-only evaluation. For generation evaluation, you may need to update RAGAS or use a different version.

---

## How to Fix Your Evaluation Dataset

### Step 1: List Actual Document IDs

Run the helper script to see the actual chunk IDs in Qdrant:

```bash
python scripts/list_document_ids.py
```

This will show you all the document chunks with their actual UUIDs.

### Step 2: Update Your Evaluation Dataset

Replace the semantic IDs in `evaluation_datasets/my_eval.json` with the actual UUIDs from Step 1.

**Before** (incorrect):
```json
{
  "query": "What is Adhil's role?",
  "relevant_doc_ids": ["resume_chunk_experience_1"],
  "relevance_scores": {"resume_chunk_experience_1": 3.0}
}
```

**After** (correct):
```json
{
  "query": "What is Adhil's role?",
  "relevant_doc_ids": ["8698a711-f896-48e5-ae4e-2c481c93dee9"],
  "relevance_scores": {"8698a711-f896-48e5-ae4e-2c481c93dee9": 3.0}
}
```

### Step 3: Re-run Evaluation

```bash
python scripts/run_evaluation.py \
  --dataset evaluation_datasets/my_eval.json \
  --output evaluation_results/run_002 \
  --k-values 1,3,5,10
```

---

## Alternative: Use Document Metadata

If you want to use semantic names, you need to modify how documents are stored in Qdrant to include a custom `chunk_id` field in the metadata. This would require changes to the document upload process.

---

## Current Status

✅ **Fixed**: Visualization division by zero error
✅ **Created**: Helper script to list document IDs (`list_document_ids.py`)
⏳ **Action Needed**: Update your evaluation dataset with correct UUIDs
⚠️  **Optional**: Fix RAGAS compatibility for generation evaluation (not needed for retrieval-only)

---

## Next Steps

1. Run `python scripts/list_document_ids.py` to get actual chunk IDs
2. Update `evaluation_datasets/my_eval.json` with the correct UUIDs
3. Re-run the evaluation
4. You should see non-zero metrics if the documents are relevant to your queries!
