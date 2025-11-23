"""
LLM interaction module using Groq streaming API.
Sends prompts and streams token-by-token responses back to the UI.
"""

import os
import json
import requests
from typing import Optional
from config.config import GROQ_API_KEY, GROQ_API_URL, user_prompt, system_prompt    

# NOTE: Groq's exact REST API path & request schema may vary.
# This wrapper uses a generic POST to GROQ_API_URL + "/completions" and expects a text response.
# If Groq's API requires a different path or JSON schema, adjust GROQ_API_URL and the payload accordingly.

HEADERS = {
    "Authorization": f"Bearer {GROQ_API_KEY}" if GROQ_API_KEY else "",
    "Content-Type": "application/json"
}

user_prompt = user_prompt
system_prompt = system_prompt





def stream_groq_chat(system_prompt: str, user_prompt: str,
                     max_tokens: int = 1000, temperature: float = 0.25):
    """
    Streaming generator for Groq responses.
    Yields tokens as they come in (for Streamlit live updates).
    """
    payload = {
        "model": "llama-3.1-8b-instant",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": True
    }

    response = requests.post(GROQ_API_URL, headers=HEADERS,
                             json=payload, stream=True, timeout=30)

    if response.status_code != 200:
        raise RuntimeError(f"Groq API error {response.status_code}: {response.text}")

    for line in response.iter_lines():
        if line:
            try:
                if line.startswith(b"data: "):
                    line = line.replace(b"data: ", b"")
                data = json.loads(line.decode("utf-8"))
                delta = data["choices"][0]["delta"]
                if "content" in delta:
                    yield delta["content"]
            except:
                continue
