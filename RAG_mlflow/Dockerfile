
FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy app and notebook
COPY . /app

EXPOSE 8501 5000

# Start MLflow UI and Streamlit when container runs
# MLflow UI on port 5000, Streamlit on 8501
CMD mlflow ui --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns & \
    streamlit run streamlit_app.py --server.port 8501 --server.address 0.0.0.0
