from django.shortcuts import render
from django.http import StreamingHttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt
from .utils import temporal_score, detect_time_bias
import pickle, faiss, numpy as np
from sentence_transformers import SentenceTransformer
import requests
import json
import re
import time
import os

# Import Django settings to get API keys
from django.conf import settings

# API URLs
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"

# Get API keys from Django settings
def get_groq_api_key():
    return getattr(settings, 'GROQ_API_KEY', None)

def get_gemini_api_key():
    return getattr(settings, 'GEMINI_API_KEY', None)

# Lazy loading of models and data
_model = None
_index = None
_docs = None
_embeddings = None

def get_model():
    """Lazy load the sentence transformer model"""
    global _model
    if _model is None:
        try:
            _model = SentenceTransformer("all-MiniLM-L6-v2")
        except Exception as e:
            raise Exception(f"Failed to load sentence transformer model: {str(e)}. Please check your internet connection.")
    return _model

def get_index_and_data():
    """Lazy load the FAISS index and embeddings"""
    global _index, _docs, _embeddings
    if _index is None:
        try:
            _index = faiss.read_index("ragapp/index.faiss")
            with open("ragapp/embeddings.pkl", "rb") as f:
                _docs, _embeddings = pickle.load(f)
        except Exception as e:
            raise Exception(f"Failed to load FAISS index or embeddings: {str(e)}. Please run 'python ragapp/ingest.py' first.")
    return _index, _docs, _embeddings

def get_top_docs(query, k=10):  
    q_vec = get_model().encode([query])
    _, I = get_index_and_data()[0].search(q_vec, k*3)
    query_type = detect_time_bias(query)

    scored = []
    for idx in I[0]:
        doc = get_index_and_data()[1][idx]
        score = np.dot(q_vec[0], get_index_and_data()[2][idx])
        boosted = score * temporal_score(doc["timestamp"], query_type)
        scored.append((boosted, doc))

    # Sort by score only (first element of tuple), ignore document for sorting
    return sorted(scored, key=lambda x: x[0], reverse=True)[:k]

def generate_conversational_answer(query, context_docs):
    """Generate a conversational response broken into multiple messages"""
    # Enhanced context with document metadata
    context_parts = []
    for i, (_, doc) in enumerate(context_docs):
        context_parts.append(f"[Document {i+1} - {doc['timestamp']}]: {doc['text']}")
    context = "\n\n".join(context_parts)
    
    # Prompt for conversational mode
    prompt = f"""Based on the following context, provide a conversational answer to the question. 
Break your response into 3-5 short, engaging messages (2-3 sentences each) that build upon each other.

Each message should:
- Be conversational and friendly
- Explain one key point or concept
- Build on the previous message
- Use simple, clear language
- Include specific details from the context when relevant

IMPORTANT: You must respond with ONLY a valid JSON array. Do not include any other text before or after the JSON.

Example format:
[
    {{"role": "assistant", "content": "First message here..."}},
    {{"role": "assistant", "content": "Second message here..."}},
    {{"role": "assistant", "content": "Third message here..."}}
]

Context:
{context}

Question: {query}

Respond with ONLY the JSON array:"""

    try:
        groq_key = get_groq_api_key()
        if not groq_key:
            return [{"role": "assistant", "content": "Error: GROQ API key not configured"}]
            
        headers = {
            "Authorization": f"Bearer {groq_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": "llama-3.1-8b-instant",  
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 1200,  
            "temperature": 0.7
        }
        
        response = requests.post(GROQ_API_URL, headers=headers, json=payload)
        response.raise_for_status()
        
        result = response.json()
        content = result['choices'][0]['message']['content'].strip()
        
        # Try to parse JSON response - handle various formats
        try:
            # First, try to find JSON array in the response
            json_start = content.find('[')
            json_end = content.rfind(']') + 1
            
            if json_start != -1 and json_end > json_start:
                json_content = content[json_start:json_end]
                messages = json.loads(json_content)
                
                if isinstance(messages, list) and all(isinstance(msg, dict) and 'content' in msg for msg in messages):
                    return messages
            else:
                # Try parsing the entire content as JSON
                messages = json.loads(content)
                if isinstance(messages, list) and all(isinstance(msg, dict) and 'content' in msg for msg in messages):
                    return messages
                    
        except json.JSONDecodeError as e:
            print(f"JSON parsing failed: {e}")
            print(f"Content received: {content}")
        
        # Fallback: split content into messages if JSON parsing fails
        # Clean up the content first
        content = content.replace('{"role": "assistant", "content": "', '').replace('"}', '')
        content = content.replace('{"role":"assistant","content":"', '').replace('"}', '')
        
        # Split by common sentence endings
        sentences = re.split(r'[.!?]+', content)
        messages = []
        current_message = ""
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            if len(current_message + sentence) < 150:  # Keep messages short
                current_message += sentence + ". "
            else:
                if current_message.strip():
                    messages.append({"role": "assistant", "content": current_message.strip()})
                current_message = sentence + ". "
        
        if current_message.strip():
            messages.append({"role": "assistant", "content": current_message.strip()})
        
        # If we still don't have messages, create one from the original content
        if not messages:
            # Clean up any remaining JSON artifacts
            clean_content = content.replace('"', '').replace('{', '').replace('}', '').replace('[', '').replace(']', '')
            clean_content = re.sub(r'role:\s*assistant,\s*content:\s*', '', clean_content)
            clean_content = clean_content.strip()
            
            if clean_content:
                messages.append({"role": "assistant", "content": clean_content})
            else:
                messages.append({"role": "assistant", "content": content})
        
        return messages
            
    except Exception as e:
        return [{"role": "assistant", "content": f"Error generating answer: {str(e)}"}]

def generate_streaming_answer(query, context_docs):
    """Generate a streaming response for real-time updates"""
    # Enhanced context with document metadata
    context_parts = []
    for i, (_, doc) in enumerate(context_docs):
        context_parts.append(f"[Document {i+1} - {doc['timestamp']}]: {doc['text']}")
    context = "\n\n".join(context_parts)
    
    prompt = f"""Based on the following context, provide a comprehensive answer to the question. 
Include specific details, examples, and temporal relevance when applicable.

Context:
{context}

Question: {query}

Please provide a detailed answer with:
- Key points from the context
- Temporal relevance (if applicable)
- Specific examples or data mentioned
- Additional insights based on the information provided

Answer:"""

    try:
        groq_key = get_groq_api_key()
        if not groq_key:
            yield f"data: {json.dumps({'error': 'GROQ API key not configured'})}\n\n"
            return
            
        headers = {
            "Authorization": f"Bearer {groq_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": "llama-3.1-8b-instant",  
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 1000,  
            "temperature": 0.7,
            "stream": True  # Enable streaming
        }
        
        response = requests.post(GROQ_API_URL, headers=headers, json=payload, stream=True)
        response.raise_for_status()
        
        for line in response.iter_lines():
            if line:
                line = line.decode('utf-8')
                if line.startswith('data: '):
                    data = line[6:]  # Remove 'data: ' prefix
                    if data == '[DONE]':
                        break
                    try:
                        json_data = json.loads(data)
                        if 'choices' in json_data and len(json_data['choices']) > 0:
                            delta = json_data['choices'][0].get('delta', {})
                            if 'content' in delta:
                                yield f"data: {json.dumps({'content': delta['content']})}\n\n"
                    except json.JSONDecodeError:
                        continue
                        
    except Exception as e:
        yield f"data: {json.dumps({'error': str(e)})}\n\n"

def generate_streaming_conversational_answer(query, context_docs):
    """Generate a streaming conversational response"""
    # Enhanced context with document metadata
    context_parts = []
    for i, (_, doc) in enumerate(context_docs):
        context_parts.append(f"[Document {i+1} - {doc['timestamp']}]: {doc['text']}")
    context = "\n\n".join(context_parts)
    
    prompt = f"""Based on the following context, provide a conversational answer to the question. 
Break your response into 3-5 short, engaging messages (2-3 sentences each) that build upon each other.

Each message should:
- Be conversational and friendly
- Explain one key point or concept
- Build on the previous message
- Use simple, clear language
- Include specific details from the context when relevant

IMPORTANT: You must respond with ONLY a valid JSON array. Do not include any other text before or after the JSON.

Example format:
[
    {{"role": "assistant", "content": "First message here..."}},
    {{"role": "assistant", "content": "Second message here..."}},
    {{"role": "assistant", "content": "Third message here..."}}
]

Context:
{context}

Question: {query}

Respond with ONLY the JSON array:"""

    try:
        groq_key = get_groq_api_key()
        if not groq_key:
            yield f"data: {json.dumps({'error': 'GROQ API key not configured'})}\n\n"
            return
            
        headers = {
            "Authorization": f"Bearer {groq_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": "llama-3.1-8b-instant",  
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 1200,  
            "temperature": 0.7,
            "stream": True  # Enable streaming
        }
        
        response = requests.post(GROQ_API_URL, headers=headers, json=payload, stream=True)
        response.raise_for_status()
        
        for line in response.iter_lines():
            if line:
                line = line.decode('utf-8')
                if line.startswith('data: '):
                    data = line[6:]  # Remove 'data: ' prefix
                    if data == '[DONE]':
                        break
                    try:
                        json_data = json.loads(data)
                        if 'choices' in json_data and len(json_data['choices']) > 0:
                            delta = json_data['choices'][0].get('delta', {})
                            if 'content' in delta:
                                yield f"data: {json.dumps({'content': delta['content']})}\n\n"
                    except json.JSONDecodeError:
                        continue
                        
    except Exception as e:
        yield f"data: {json.dumps({'error': str(e)})}\n\n"

@csrf_exempt
def health_check(request):
    """Health check endpoint to verify system components"""
    try:
        # Test model loading
        model = get_model()
        
        # Test index and data loading
        index, docs, embeddings = get_index_and_data()
        
        # Test basic functionality
        test_query = "test"
        q_vec = model.encode([test_query])
        _, I = index.search(q_vec, 1)
        
        return JsonResponse({
            "status": "healthy",
            "components": {
                "sentence_transformer": "loaded",
                "faiss_index": "loaded",
                "embeddings": "loaded",
                "documents_count": len(docs),
                "test_query": "successful"
            }
        })
        
    except Exception as e:
        return JsonResponse({
            "status": "unhealthy",
            "error": str(e)
        }, status=500)

@csrf_exempt
def stream_answer(request):
    """Streaming endpoint for real-time answer generation"""
    if request.method == "POST":
        try:
            data = json.loads(request.body)
            query = data.get("query")
            mode = data.get("mode", "regular")
            
            if not query:
                return JsonResponse({"error": "Query is required"}, status=400)
            
            # Get relevant documents
            context_docs = get_top_docs(query)
            
            # Choose streaming function based on mode
            if mode == "conversational":
                stream_generator = generate_streaming_conversational_answer(query, context_docs)
            else:
                stream_generator = generate_streaming_answer(query, context_docs)
            
            response = StreamingHttpResponse(
                stream_generator,
                content_type='text/event-stream'
            )
            response['Cache-Control'] = 'no-cache'
            response['X-Accel-Buffering'] = 'no'
            return response
            
        except Exception as e:
            return JsonResponse({"error": str(e)}, status=500)
    
    return JsonResponse({"error": "Method not allowed"}, status=405)

@csrf_exempt
def gemini_web_answer(request):
    """Backend endpoint for Gemini API calls"""
    if request.method == "POST":
        try:
            data = json.loads(request.body)
            query = data.get("query")
            
            if not query:
                return JsonResponse({"error": "Query is required"}, status=400)
            
            gemini_key = get_gemini_api_key()
            if not gemini_key:
                return JsonResponse({"error": "Gemini API key not configured"}, status=500)
            
            headers = {
                "Content-Type": "application/json"
            }
            
            payload = {
                "contents": [{"parts": [{"text": query}]}]
            }
            
            response = requests.post(
                f"{GEMINI_API_URL}?key={gemini_key}",
                headers=headers,
                json=payload
            )
            
            if not response.ok:
                return JsonResponse({"error": f"API error: {response.status}"}, status=response.status_code)
            
            result = response.json()
            content = 'No answer received.'
            
            if (result.get('candidates') and 
                result['candidates'][0] and 
                result['candidates'][0].get('content') and 
                result['candidates'][0]['content'].get('parts') and 
                result['candidates'][0]['content']['parts'][0].get('text')):
                
                content = result['candidates'][0]['content']['parts'][0]['text']
                
                # Format: Remove Markdown asterisks and convert to HTML
                import re
                content = re.sub(r'\*\*([^*]+)\*\*', r'<b>\1</b>', content)  # bold
                if re.search(r'^\s*\*', content, re.MULTILINE):
                    # Only format as list if there are bullets
                    content = re.sub(r'\n\s*\*\s*', '</li><li>', content)  # bullet points
                    content = re.sub(r'^\*\s*', '<li>', content, flags=re.MULTILINE)  # bullet at start of line
                    content = re.sub(r'<li>', '<ul><li>', content, count=1)  # first bullet
                    content = re.sub(r'(</li>)(?![\s\S]*<li>)', r'\1</ul>', content)  # close ul after last li
                # Remove any stray asterisks
                content = content.replace('*', '')
            
            return JsonResponse({"content": content})
            
        except Exception as e:
            return JsonResponse({"error": str(e)}, status=500)
    
    return JsonResponse({"error": "Method not allowed"}, status=405)

def home(request):
    if request.method == "POST":
        query = request.POST.get("query")
        mode = request.POST.get("mode", "regular")  # Default to regular mode
        context_docs = get_top_docs(query)
        
        # For non-streaming requests, use the original functions
        if mode == "conversational":
            answer = generate_conversational_answer(query, context_docs)
        else:
            answer = generate_answer(query, context_docs)
        
        return render(request, "ragapp/home.html", {
            "query": query,
            "answer": answer,
            "context_docs": context_docs,
            "mode": mode
        })
    return render(request, "ragapp/home.html")

# Keep the original generate_answer function for non-streaming requests
def generate_answer(query, context_docs):
    # Enhanced context with document metadata
    context_parts = []
    for i, (_, doc) in enumerate(context_docs):
        context_parts.append(f"[Document {i+1} - {doc['timestamp']}]: {doc['text']}")
    context = "\n\n".join(context_parts)
    
    prompt = f"""Based on the following context, provide a comprehensive answer to the question. 
Include specific details, examples, and temporal relevance when applicable.

Context:
{context}

Question: {query}

Please provide a detailed answer with:
- Key points from the context
- Temporal relevance (if applicable)
- Specific examples or data mentioned
- Additional insights based on the information provided

Answer:"""

    try:
        groq_key = get_groq_api_key()
        if not groq_key:
            return "Error: GROQ API key not configured"
        
        # Clean the API key (remove any whitespace)
        groq_key = groq_key.strip()
        
        # Check prompt length
        if len(prompt) > 32000:  # Groq has a limit
            return "Error: Prompt too long for Groq API"
            
        headers = {
            "Authorization": f"Bearer {groq_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": "llama-3.1-8b-instant",  
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 1000,  
            "temperature": 0.7
        }
        
        response = requests.post(GROQ_API_URL, headers=headers, json=payload)
        
        if not response.ok:
            error_detail = f"HTTP {response.status_code}"
            try:
                error_response = response.json()
                if 'error' in error_response:
                    error_detail += f": {error_response['error'].get('message', 'Unknown error')}"
                elif 'message' in error_response:
                    error_detail += f": {error_response['message']}"
            except:
                error_detail += f": {response.text[:200]}"
            return f"Error generating answer: {error_detail}"
        
        result = response.json()
        return result['choices'][0]['message']['content']
            
    except Exception as e:
        return f"Error generating answer: {str(e)}"
