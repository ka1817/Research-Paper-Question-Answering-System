## 📄 Research Paper QA System
A powerful, AI-driven Question Answering (QA) system built to extract and answer questions from research papers!
Built with LangChain, Streamlit, FastAPI, FAISS, and more — designed for both interactive web apps and robust backend APIs.

🚀 Features

📚 Upload Research Papers (PDFs) easily

🧠 Smart Question Answering over the uploaded papers

🔎 BM25 Ranking + Dense Vector Search (FAISS) hybrid retrieval

🧩 LangChain Integration for LLM-powered reasoning

🖥️ Streamlit Frontend for easy interaction

⚡ FastAPI Backend for scalable APIs

🛡️ Environment variable support via .env

🧪 Testable using pytest

🛠️ Tech Stack

LangChain

FastAPI

Streamlit

FAISS

BM25 (rank_bm25)

Hugging Face Embeddings

Groq LLM API (via langchain_groq)

PDF Processing (pypdf)

Environment Management (python-dotenv)

Testing (pytest)

📚 How It Works

PDF Parsing: Extract text from uploaded papers using pypdf.

Indexing: Create hybrid indexes (BM25 + FAISS) over the extracted text.

Embedding: Use HuggingFace or Groq LLMs for text embeddings.

Retrieval: Fetch the most relevant sections using hybrid search.

Answer Generation: Query the LLM through LangChain to generate answers based on the context.

