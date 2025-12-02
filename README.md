📊 RAG Policy Chatbot Using Vector Search & LLM
📘 Project Overview

This project implements a Retrieval-Augmented Generation (RAG) based Question Answering system that enables users to ask natural language questions on internal company policy documents (such as Refund Policy and Employee Leave Policy). The system retrieves the most relevant document sections using semantic search with vector embeddings and generates accurate answers using a language model, exposed through a FastAPI backend.

The primary goal is to eliminate manual document searching and provide instant, reliable, document-grounded answers through an API.

🗃️ Data Sources

The system uses internally created policy documents:

Company Refund & Return Policy (Doc1.txt)

Employee Leave Policy (Doc2.txt)

Each document contains structured textual rules and conditions used for retrieval.
The documents are stored as TXT files, which are parsed and chunked for vectorization.

⚙️ Methodology
1️⃣ Document Ingestion

Loaded all TXT files from the input directory.

Each document was read and stored with metadata (source filename).

2️⃣ Text Chunking

Documents were split into overlapping chunks (250 words with 30-word overlap).

This improves semantic retrieval and avoids loss of context during embedding.

3️⃣ Embedding Generation

Each chunk was converted into a dense vector using:

Sentence Transformers – all-MiniLM-L6-v2

These embeddings capture the semantic meaning of text instead of just keywords.

4️⃣ Vector Indexing with FAISS

All embeddings were stored in a FAISS vector index.

This enables fast similarity search for user queries.

5️⃣ Query Retrieval

User questions are converted into embeddings.

FAISS retrieves the Top-K most relevant document chunks based on vector similarity.

6️⃣ Answer Generation (RAG Pipeline)

Retrieved text chunks are passed as context to a language model.

The model generates an answer strictly from the retrieved document content.

Output is cleaned to remove prompt leakage and hallucination.

7️⃣ API Deployment

The complete pipeline is exposed using FastAPI.

Users interact with the system via a REST API endpoint.

API is tested using Swagger UI (/docs).

🧠 Technologies Used

Language: Python

Embeddings: Sentence Transformers (MiniLM)

Vector Search: FAISS

LLM: DistilGPT-2 / GPT-2 (demo purpose)

Backend: FastAPI, Uvicorn

Prototyping: Jupyter Notebook

Data Storage: JSON, Text files

📈 Key Functional Results

Successfully answers policy-related questions using only internal documents.

Provides accurate, document-grounded answers without hallucination.

Supports multiple document sources in a single knowledge base.

Delivers responses via a production-style REST API.

Demonstrates a complete RAG pipeline from ingestion to deployment.

Example:

Question: “How many paid leaves do employees get per year?”

Answer: “Employees are entitled to 24 paid leaves per year.”

💡 Practical Applications

HR policy chatbots

Customer support automation

Internal company knowledge bases

Compliance document search

Enterprise document intelligence systems

This system significantly reduces manual search time and improves organizational productivity.

🚀 Future Enhancements

Replace demo LLM with instruction-tuned models (LLaMA, GPT-4, Claude).

Add PDF ingestion and OCR support.

Implement user authentication and logging.

Deploy using Docker & Cloud (AWS/GCP).

Add a frontend UI (Streamlit / React) for chat-based interaction.

Integrate real-time document updates and re-indexing.

✅ How to Run the Project
pip install -r requirements.txt
python -m uvicorn app:app --port 8001


Open in browser:

http://localhost:8001/docs
