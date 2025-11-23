"""
Fallback live web search using Bing API when documents lack needed info.
Provides extra context for improving LLM answers.
"""

import requests
from typing import List, Dict
from config.config import BING_SEARCH_API_KEY, BING_SEARCH_API_URL

def bing_search(query: str, top_k: int = 3) -> List[Dict]:
    """
    Simple wrapper around Bing Web Search API v7.
    Returns list of {"name", "url", "snippet"}.
    """
    if not BING_SEARCH_API_KEY:
        return []

    endpoint = BING_SEARCH_API_URL
    headers = {"Ocp-Apim-Subscription-Key": BING_SEARCH_API_KEY}
    params = {"q": query, "count": top_k}
    r = requests.get(endpoint, headers=headers, params=params, timeout=10)
    r.raise_for_status()
    data = r.json()
    results = []
    for item in data.get("webPages", {}).get("value", []):
        results.append({
            "name": item.get("name"),
            "url": item.get("url"),
            "snippet": item.get("snippet")
        })
    return results

def web_search_fallback(query: str, top_k: int = 5) -> List[Dict]:
    try:
        return bing_search(query, top_k)
    except Exception:
        return []
