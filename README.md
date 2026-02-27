# Enterprise RAG Chatbot Platform

A production-ready Retrieval-Augmented Generation (RAG) pipeline designed for context-aware document interaction. This system allows users to chat with private documentation securely using local vector storage and state-of-the-art LLMs.

## Features

* **Modular Ingestion**: Extracts and chunks text from PDF documentation.
* **Local Vector Storage**: Uses FAISS for high-performance, in-memory semantic search.
* **Context-Aware AI**: Integrated with Google Gemini for accurate, grounded responses.
* **Conversation Memory**: Maintains chat history for seamless follow-up questions.
* **Confidence Guardrails**: Implements distance-based thresholds to prevent hallucinations and out-of-scope replies.

---

## Architectural Decisions

### 1. Document Chunking Strategy
The system uses a **Sliding Window** approach with a chunk size of 800 characters and a 150-character overlap.  
**Reasoning**: This size is large enough to capture complete semantic concepts while the overlap ensures that sentences spanning two chunks are not lost, maintaining continuity for the retriever.

### 2. Embedding Model Choice
I selected the `all-MiniLM-L6-v2` model from Sentence-Transformers.  
**Reasoning**: It provides a perfect balance of speed and accuracy. It is lightweight enough to run locally on CPU-only machines while producing high-quality 384-dimensional embeddings optimized for semantic similarity.

### 3. Handling Irrelevant Queries
A dual-layer guardrail system is implemented:  
* **Layer 1 (Vector Threshold)**: If the FAISS search returns a distance score higher than 1.5, the system identifies the query as "Out-of-Distribution" and triggers a polite fallback message immediately, saving API costs.  
* **Layer 2 (System Prompting)**: The LLM is strictly instructed to respond only based on provided context and to decline answering if the information is missing.

---

## Setup Instructions

1. **Environment**: Create a `.env` file and add your `GOOGLE_API_KEY`.
2. **Install**: `pip install -r requirements.txt`
3. **Ingest**: Place your PDF in the `/data` folder and run `python src/ingest.py`.
4. **Launch**: Run `python src/app.py` to start the web interface at `localhost:7860`.

---
