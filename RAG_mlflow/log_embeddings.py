
"""log_embeddings.py
Simple script to encode documents, log embeddings and a dummy evaluation metric to MLflow.
Run: python log_embeddings.py
"""
from sentence_transformers import SentenceTransformer
import numpy as np
import mlflow, os, uuid

documents = [
    "The Port of Hamburg is one of the largest ports in Europe and handles millions of TEU annually.",
    "Container throughput can be forecasted using time-series models and operational features like arrival rates and berth availability.",
    "Predictive maintenance for cranes reduces downtime by combining sensor data with failure logs and scheduled inspections.",
    "RAG systems use dense vector retrieval (embeddings) to fetch supporting documents, then condition an LLM on the retrieved context."
]

model_name = 'all-MiniLM-L6-v2'
model = SentenceTransformer(model_name)
embeddings = model.encode(documents, convert_to_numpy=True)

mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI', 'file://' + os.path.abspath('./mlruns')))
experiment_name = 'rag_embeddings_demo'
mlflow.set_experiment(experiment_name)

with mlflow.start_run(run_name='emb_run_'+str(uuid.uuid4())[:8]) as run:
    run_id = run.info.run_id
    mlflow.log_param('model_name', model_name)
    mlflow.log_param('num_documents', len(documents))
    np.save('embeddings.npy', embeddings)
    mlflow.log_artifact('embeddings.npy', artifact_path='embeddings')
    # Save and log model files
    model_dir = 'sent_transformer_saved'
    model.save(model_dir)
    mlflow.log_artifact(model_dir, artifact_path='embedding_model')
    # Dummy evaluation metric: mean pairwise distance (illustrative)
    from sklearn.metrics.pairwise import euclidean_distances
    dists = euclidean_distances(embeddings)
    mean_dist = float(dists.mean())
    mlflow.log_metric('mean_pairwise_distance', mean_dist)
    print(f'Logged run {run_id} with mean_pairwise_distance={mean_dist}')
