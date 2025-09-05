from datetime import datetime

def temporal_score(doc_timestamp, query_type="neutral"):
    doc_date = datetime.strptime(doc_timestamp, "%Y-%m-%d")
    age_days = (datetime.now() - doc_date).days

    if query_type == "recent":
        return max(0.1, 1.0 - (age_days / 3650))
    elif query_type == "old":
        return min(2.0, 1.0 + (age_days / 3650))
    return 1.0

def detect_time_bias(query):
    query = query.lower()
    if "recent" in query or "latest" in query:
        return "recent"
    elif "history" in query or "origin" in query:
        return "old"
    return "neutral"
