
import streamlit as st
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import mlflow
import os

st.set_page_config(page_title='RAG + MLflow Demo', layout='wide')

st.title('RAG + MLflow Demo — Retrieval & Embeddings')

st.sidebar.header('Settings')
model_name = st.sidebar.selectbox('Embedding model', ['all-MiniLM-L6-v2'], index=0)
k = st.sidebar.slider('Top-k retrieval', 1, 5, 2)

@st.cache(allow_output_mutation=True)
def load_model(name):
    return SentenceTransformer(name)

model = load_model(model_name)

# Sample docs
documents = [
    "The Port of Hamburg is one of the largest ports in Europe and handles millions of TEU annually.",
    "Container throughput can be forecasted using time-series models and operational features like arrival rates and berth availability.",
    "Predictive maintenance for cranes reduces downtime by combining sensor data with failure logs and scheduled inspections.",
    "RAG systems use dense vector retrieval (embeddings) to fetch supporting documents, then condition an LLM on the retrieved context."
]

embeddings = model.encode(documents, convert_to_numpy=True)
d = embeddings.shape[1]
index = faiss.IndexFlatL2(d)
index.add(embeddings)

st.subheader('Query the corpus')
query = st.text_input('Frage an das System', 'Welche Methoden eignen sich zur Vorhersage von Ankunftszeiten im Hafen?')

if st.button('Retrieve'):
    q_emb = model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(q_emb, k)
    st.write('Ergebnisse:')
    for dist, idx in zip(distances[0], indices[0]):
        st.markdown(f"- **Score:** {dist:.3f}  
  {documents[idx]}")
    # Compose prompt (no LLM call)
    context = '\n\n'.join([documents[i] for i in indices[0]])
    prompt = f"""Verwende den folgenden Kontext, um die Frage zu beantworten:\n\n{context}\n\nFrage:\n{query}"""
    st.text_area('Prompt (zum Senden an ein LLM)', prompt, height=200)

st.sidebar.markdown('---')
st.sidebar.markdown('MLflow runs in ./mlruns by default. Start MLflow UI on port 5000 to inspect runs.')
