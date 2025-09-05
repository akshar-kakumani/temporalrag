# Temporal-Aware RAG (Django)

This is a simple Retrieval-Augmented Generation (RAG) project built using Django. It enhances document retrieval based on both semantic similarity and temporal relevance (i.e., recent or old documents depending on the query context).

---

## 📁 Project Structure

```
temporalrag/
├── manage.py
├── requirements.txt
├── ragapp/
│   ├── docs.json
│   ├── ingest.py
│   ├── utils.py
│   ├── views.py
│   ├── embeddings.pkl (created after running ingest.py)
│   ├── index.faiss (created after running ingest.py)
│   ├── templates/
│   │   └── ragapp/
│   │       └── home.html
│   ├── static/
│   │   └── ragapp/
│   │       └── style.css
```

---

## 🚀 How to Run the Project

### 1. Clone or Download the Repository

Unzip the project folder or clone it from your repository.

### 2. Install Dependencies

Make sure Python is installed, then install required packages:

```bash
pip install -r requirements.txt
```

### 3. Create Embedding Index

Run this script to encode documents and create the FAISS index:

```bash
python ragapp/ingest.py
```

This will generate:
- `embeddings.pkl`
- `index.faiss`

### 4. Set Your OpenAI API Key

In `views.py`, replace:

```python
openai.api_key = "your-openai-key"
```

with your actual [OpenAI API key](https://platform.openai.com/account/api-keys).

### 5. Run the Django Server

```bash
python manage.py runserver
```

Visit [http://127.0.0.1:8000](http://127.0.0.1:8000) in your browser.

---

## 💡 What You Can Do

- Ask time-sensitive questions like:
  - "What are the latest advances in AI?"
  - "What is the history of transformers in NLP?"
  - "How has cloud security evolved in the last five years?"
  - "What were the major cyber attacks in 2024?"
  - "What are the recent innovations in blockchain technology?"
  - "How has telemedicine adoption changed since 2020?"
  - "What are the trends in robotics automation over the past decade?"
  - "What regulatory changes have impacted AI in 2023 and 2024?"
- The app retrieves and ranks documents using:
  - Sentence similarity
  - Temporal bias (recent or old depending on query)

---

## 📚 Technologies Used

- Django (web framework)
- Sentence-Transformers (semantic embeddings)
- FAISS (fast document indexing)
- OpenAI (text generation)
- HTML/CSS for frontend

---

## ✅ Next Steps

- Add upload feature to inject new documents
- Log past queries and responses
- Deploy on Render or Railway

---

Built for learning and demonstrating how RAG can be customized with temporal intelligence.
