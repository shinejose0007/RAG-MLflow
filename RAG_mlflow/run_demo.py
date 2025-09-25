"""
RAG + MLflow demo: embeddings -> FAISS index -> retrieval
Tracks embedding model and embedding vectors in MLflow (file-based local tracking).

Run:
    python run_demo.py

Notes:
- Installs required packages from requirements.txt
- By default MLflow uses ./mlruns as the tracking folder.
- This script is intentionally small and runnable locally.
"""

import os
import mlflow
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import json

MLFLOW_TRACKING_URI = os.environ.get('MLFLOW_TRACKING_URI', 'file://' + os.path.abspath('./mlruns'))

def create_documents():
    return [
        "The Port of Hamburg is one of the largest ports in Europe and handles millions of TEU annually.",
        "Container throughput can be forecasted using time-series models and operational features like arrival rates and berth availability.",
        "Predictive maintenance for cranes reduces downtime by combining sensor data with failure logs and scheduled inspections.",
        "RAG systems use dense vector retrieval (embeddings) to fetch supporting documents, then condition an LLM on the retrieved context."
    ]

def embed_and_index(model_name='all-MiniLM-L6-v2'):
    # initialize model
    model = SentenceTransformer(model_name)
    documents = create_documents()
    embeddings = model.encode(documents, convert_to_numpy=True)
    # create faiss index
    d = embeddings.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(embeddings)
    return model, documents, embeddings, index

def retrieve(index, model, documents, query, k=2):
    q_emb = model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(q_emb, k)
    results = []
    for dist, idx in zip(distances[0], indices[0]):
        results.append({'doc': documents[idx], 'score': float(dist)})
    return results

def main():
    print("MLflow tracking URI:", MLFLOW_TRACKING_URI)
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    model_name = 'all-MiniLM-L6-v2'

    # Start MLflow run
    with mlflow.start_run(run_name="rag_embeddings_demo"):
        # Log meta
        mlflow.log_param("embedding_model", model_name)

        # Create embeddings and index
        model, documents, embeddings, index = embed_and_index(model_name=model_name)

        # Save a small JSON of documents as an artifact
        docs_path = "documents.json"
        with open(docs_path, "w", encoding="utf-8") as f:
            json.dump(documents, f, ensure_ascii=False, indent=2)
        mlflow.log_artifact(docs_path, artifact_path="rag_docs")

        # Log embedding array as a numpy artifact
        emb_path = "embeddings.npy"
        np.save(emb_path, embeddings)
        mlflow.log_artifact(emb_path, artifact_path="embeddings")

        # Optionally: save a small faiss index
        faiss_path = "faiss.index"
        faiss.write_index(index, faiss_path)
        mlflow.log_artifact(faiss_path, artifact_path="faiss_index")

        # Log a short note as an artifact
        with open("notes.txt", "w", encoding="utf-8") as f:
            f.write("RAG demo: embeddings logged to MLflow, FAISS index saved as artifact.\n")
        mlflow.log_artifact("notes.txt", artifact_path="notes")

        # Example retrieval and log the results as metric (distance) and artifact
        query = "Wie kann man Container-Durchsatz vorhersagen?"
        results = retrieve(index, model, documents, query, k=2)
        # log distances as metrics
        for i, r in enumerate(results):
            mlflow.log_metric(f"dist_{i}", r['score'])
        # save retrieval to artifact
        with open("retrieval.json", "w", encoding="utf-8") as f:
            json.dump({"query": query, "results": results}, f, ensure_ascii=False, indent=2)
        mlflow.log_artifact("retrieval.json", artifact_path="retrievals")

        print("Run finished. Artifacts and metrics logged to MLflow.")

if __name__ == '__main__':
    main()
