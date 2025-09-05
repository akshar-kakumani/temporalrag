import json, pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

with open("ragapp/docs.json") as f:
    docs = json.load(f)

model = SentenceTransformer("all-MiniLM-L6-v2")
texts = [doc["text"] for doc in docs]
embeddings = model.encode(texts)

with open("ragapp/embeddings.pkl", "wb") as f:
    pickle.dump((docs, embeddings), f)

index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(np.array(embeddings))
faiss.write_index(index, "ragapp/index.faiss")
