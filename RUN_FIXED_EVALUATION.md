# Quick Start: Running Evaluation with Fixed Dataset

## What I Did

✅ Created `evaluation_datasets/my_eval_fixed.json` with **actual Qdrant document IDs**

**Key Changes**:
- Replaced semantic IDs (`resume_chunk_experience_1`) with actual UUIDs
- Used the 2 document IDs found in your Qdrant database:
  - `8698a711-f896-48e5-ae4e-2c481c93dee9`
  - `5e512a55-51ba-4409-ad47-9287e13c4ab6`
- Reduced from 6 to 3 queries (since you only have 2 documents in Qdrant)

## Run the Fixed Evaluation

**In your terminal** (make sure venv is activated):

```bash
# Retrieval-only evaluation (recommended first)
python scripts/run_evaluation.py \
  --dataset evaluation_datasets/my_eval_fixed.json \
  --output evaluation_results/run_002 \
  --k-values 1,3,5,10
```

This should now show **non-zero retrieval metrics**!

## Expected Results

With the fixed dataset, you should see:
- ✅ **Precision@K > 0** (some retrieved docs are relevant)
- ✅ **Recall@K > 0** (some relevant docs are retrieved)
- ✅ **MRR > 0** (relevant docs appear in results)
- ✅ **Visualizations work** (no more division by zero)

## Important Note

⚠️ **You only have 2 documents in Qdrant**

Your original evaluation dataset expected 6 different document chunks:
- `resume_chunk_experience_1`
- `resume_chunk_skills`
- `resume_chunk_experience_2`
- `resume_chunk_projects`
- `resume_chunk_education`

But Qdrant only contains 2 documents. To get full evaluation coverage:

1. **Upload your resume** to the RAG system via the API or UI
2. **Re-run** `python scripts/list_document_ids.py` (in venv) to see all chunks
3. **Update** `my_eval.json` with all the correct chunk IDs
4. **Re-run** evaluation

## Next Steps

1. Run the evaluation with `my_eval_fixed.json`
2. Check if metrics are now non-zero
3. Upload more documents if needed
4. Create a comprehensive evaluation dataset with all document chunks
