# TemporalRAG (Django)

A modern Retrieval-Augmented Generation (RAG) application prototype built with Django featuring real-time streaming responses, multiple answer modes, and temporal-aware document retrieval. The app combines semantic similarity with temporal relevance to provide contextually appropriate answers.

## ✨ Features

- **🚀 Live Streaming**: Real-time word-by-word answer generation
- **💬 Multiple Modes**: Regular, Conversational, and Web Answer modes
- **⏰ Temporal Intelligence**: Time-aware document ranking and filtering
- **🌐 Web Integration**: Direct web search capabilities via Gemini API
- **🎨 Modern UI**: Dark mode, responsive design, and smooth animations
- **🔒 Secure**: Environment-based API key management
- **☁️ Production Ready**: Optimized for AWS Elastic Beanstalk deployment

---

## 📁 Project Structure

```
temporalrag/
├── manage.py
├── requirements.txt
├── Procfile                    # For Heroku-style deployments
├── .env                        # Environment variables (create from .env.example)
├── .env.example               # Template for environment variables
├── .gitignore                 # Git ignore rules
├── ragapp/
│   ├── docs.json              # Knowledge base (400+ AI/tech documents)
│   ├── ingest.py              # Data processing script
│   ├── utils.py               # Utility functions
│   ├── views.py               # Main application logic
│   ├── urls.py                # URL routing
│   ├── embeddings.pkl         # Processed embeddings (created after ingest.py)
│   ├── index.faiss            # FAISS search index (created after ingest.py)
│   ├── templates/
│   │   └── ragapp/
│   │       └── home.html      # Main UI template
│   └── static/
│       └── ragapp/
│           ├── style.css      # Modern CSS with dark mode
│           └── favicon.svg    # App favicon
└── temporalrag/
    ├── settings.py            # Django settings with environment variables
    ├── urls.py                # Main URL configuration
    └── wsgi.py                # WSGI application
```

---

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone <your-repo-url>
cd temporalrag
```

### 2. Set Up Environment

Create a virtual environment and install dependencies:

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Create Environment Variables

Copy the example environment file and add your API keys:

```bash
cp .env.example .env
```

Edit `.env` file with your API keys:

```env
DJANGO_SECRET_KEY=your-secret-key-here
DEBUG=True
ALLOWED_HOSTS=localhost,127.0.0.1
GROQ_API_KEY=your-groq-api-key
GEMINI_API_KEY=your-gemini-api-key
```

**Get API Keys:**
- **Groq API**: [console.groq.com](https://console.groq.com) (for LLM responses)
- **Gemini API**: [makersuite.google.com](https://makersuite.google.com) (for web search)

### 4. Create Embedding Index

Run the data processing script:

```bash
python ragapp/ingest.py
```

This generates:
- `ragapp/embeddings.pkl` - Document embeddings
- `ragapp/index.faiss` - FAISS search index

### 5. Run the Application

```bash
# Run database migrations
python manage.py migrate

# Collect static files
python manage.py collectstatic --noinput

# Start the development server
python manage.py runserver
```

Visit [http://127.0.0.1:8000](http://127.0.0.1:8000) in your browser.

## 🎯 Usage

### Answer Modes

1. **📝 Regular Mode**: Standard RAG with temporal awareness
2. **💬 Conversational Mode**: Multi-message responses for complex topics
3. **🌐 Web Answer Mode**: Direct web search using Gemini API

### Streaming Feature

Enable "⚡ Live Streaming" for real-time, word-by-word answer generation.

### Temporal Filtering

Use the time range dropdown to focus on:
- All time
- Past year
- Past 5 years
- Custom range

---

## 💡 Example Queries

Try these time-sensitive questions:

- **Recent AI Developments**: "What are the latest advances in AI in 2024?"
- **Historical Context**: "What is the history of transformers in NLP?"
- **Evolution Analysis**: "How has cloud security evolved in the last five years?"
- **Specific Timeframes**: "What were the major cyber attacks in 2024?"
- **Technology Trends**: "What are the recent innovations in blockchain technology?"
- **Industry Changes**: "How has telemedicine adoption changed since 2020?"
- **Long-term Trends**: "What are the trends in robotics automation over the past decade?"
- **Regulatory Updates**: "What regulatory changes have impacted AI in 2023 and 2024?"

## 🔧 Technical Architecture

The app uses a sophisticated retrieval system:

1. **Document Processing**: 400+ AI/tech documents with timestamps
2. **Embedding Generation**: Sentence-Transformers (all-MiniLM-L6-v2)
3. **Vector Search**: FAISS for fast similarity search
4. **Temporal Scoring**: Time-aware document ranking
5. **LLM Integration**: Groq API for fast response generation
6. **Streaming**: Server-Sent Events for real-time updates

---

## 📚 Technologies Used

- **Backend**: Django, Python
- **AI/ML**: Sentence-Transformers, FAISS, Groq API, Gemini API
- **Frontend**: HTML5, CSS3, JavaScript (ES6+)
- **Deployment**: AWS Elastic Beanstalk, WhiteNoise
- **Data**: JSON, Pickle, NumPy

---

## 🚀 Deployment

### AWS Elastic Beanstalk

1. **Prepare for deployment**:
   ```bash
   # Install dependencies
   pip install -r requirements.txt
   
   # Collect static files
   python manage.py collectstatic --noinput
   
   # Create deployment package
   zip -r temporalrag.zip . -x "venv/*" ".git/*" "*.pyc" "__pycache__/*"
   ```

2. **Set environment variables in EB**:
   - `DJANGO_SECRET_KEY`
   - `DEBUG=False`
   - `ALLOWED_HOSTS=your-domain.elasticbeanstalk.com`
   - `GROQ_API_KEY`
   - `GEMINI_API_KEY`

3. **Deploy**: Upload the zip file to Elastic Beanstalk

### Other Platforms

- **Heroku**: Use the included `Procfile`
- **Railway/Render**: Follow standard Django deployment guides

---

## 🔒 Security Features

- Environment-based API key management
- CSRF protection enabled
- Secure static file serving with WhiteNoise
- Non-root user in production containers
- Health check endpoints for monitoring

---

## 📈 Performance Optimizations

- Lazy loading of ML models and embeddings
- FAISS vector search for fast retrieval
- Streaming responses for better UX
- Compressed static files with cache headers
- Optimized database queries

---

Built for learning and demonstrating modern RAG applications with temporal intelligence and real-time streaming capabilities.
