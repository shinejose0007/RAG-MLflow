
# RAG + MLflow Demo

This small demo shows a minimal Retrieval-Augmented-Generation (RAG) pipeline:
- Create embeddings with `sentence-transformers`
- Index them with `faiss`
- Perform retrieval for a query
- Track the embedding model, embeddings, FAISS index and retrievals using `mlflow` (local file-based tracking)

## Contents
- `run_demo.py` — runnable script that creates embeddings, builds FAISS index and logs artifacts/metrics to MLflow.
- `rag_mlflow_demo_notebook.ipynb` — Jupyter notebook with the same flow and explanatory cells.
- `requirements.txt` — Python package versions used.
- `README.md` — this file.

## Quick start (local)
1. Create a virtual environment and install requirements:
```bash
python -m venv venv
source venv/bin/activate    # macOS/Linux
venv\Scripts\activate     # Windows
pip install -r requirements.txt
```

2. Run the demo:
```bash
python run_demo.py
```

3. View MLflow UI:
```bash
mlflow ui --backend-store-uri ./mlruns
```
Then open http://127.0.0.1:5000 in your browser to inspect runs, metrics and artifacts.

## Notes
- This demo uses a local file-based MLflow tracking URI (`./mlruns`). For production, configure a remote tracking server or managed MLflow.
- The notebook and script are minimal by design to be easy to run and adapt. Consider replacing FAISS with a managed vector DB (Pinecone, Weaviate) for scale.
- Ensure you respect license and usage limits for any LLM provider if you extend the demo to call a hosted LLM.



