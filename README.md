

A conversational AI chatbot designed to assist users in educational queries. The system leverages the UV package for enhanced data retrieval and response generation, combining Retrieval-Augmented Generation (RAG) with LLM streaming and web search fallback.

---



## Project Overview
EduTutor is an AI-powered chatbot built to answer educational queries with high accuracy. By leveraging the **UV package**, it integrates knowledge from PDFs and web sources, providing real-time, contextual answers.  

This project combines:
- RAG workflow for contextual retrieval
- LLM streaming responses for interactive conversations
- Web search fallback for queries outside available documents

---

## Features
- Interactive chat interface using **Streamlit**
- Supports PDF ingestion for custom document knowledge
- Web search fallback to handle unseen queries
- Real-time streaming responses from the LLM
- Personalized explanations and recommendations

---

## Tech Stack
- **Language:** Python 3.13
- **Libraries/Packages:** Streamlit, UV, Groq API, Hugging Face embeddings
- **Modeling:** LLMs for chat, Retrieval-Augmented Generation

---

## Installation

1. Clone the repository:
```bash
git clone [https://github.com/yourusername/edututor-chatbot.git](https://github.com/aijulhussain/NeoStats-Analytics.git)
uv venv
.venv\Scripts\activate
uv add -r requirements.txt
Set your API keys: GROQ_API_KEY, BING_SEARCH_API_KEY
streamlit run app.py

