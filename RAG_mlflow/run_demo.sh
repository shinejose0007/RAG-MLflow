
#!/usr/bin/env bash
# Simple helper: start mlflow ui in background and print info
echo "Starting mlflow UI (sqlite backend) ..."
mlflow ui --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns &
echo "MLflow UI started at http://127.0.0.1:5000 (if not open, wait a few seconds)."
echo "Open rag_mlflow_demo_notebook.ipynb in Jupyter or run in Colab."
