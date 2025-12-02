# 📊 RAG Policy Chatbot Using Vector Search & LLM

## 🔄 System Flow (End-to-End RAG Pipeline)

```text
START
  │
  │  User Sends Question via API
  ▼
[ FastAPI Backend ]
  │
  │  Convert Query → Vector using SentenceTransformer
  ▼
[ Query Embedding ]
  │
  │  FAISS Similarity Search
  ▼
[ Top-K Chunk Retrieval ]
  │
  │  Merge Chunks → Build Prompt Context
  ▼
[ Context Builder ]
  │
  │  Send Prompt to Language Model
  ▼
[ GPT Model ]
  │
  │  Generate Answer from Context
  ▼
[ Answer Cleaning ]
  │
  │  Attach Source Metadata
  ▼
[ API JSON Response ]
  │
  ▼
END
```
## 📘 Project Overview

This project implements a **Retrieval-Augmented Generation (RAG) based Question Answering system** that enables users to ask natural language questions on internal company policy documents such as **Refund Policy** and **Employee Leave Policy**.

Instead of manually searching long documents, users can interact with a REST API that:

- Retrieves the most relevant document sections using **semantic vector search**
- Generates accurate answers using a **language model**
- Returns the **source documents** used to generate the answer

The primary goal is to **eliminate manual document searching** and provide **instant, reliable, document-grounded answers**.

---

## 🗃️ Data Sources

The system uses internally created policy documents:

- `Doc1.txt` – Company Refund & Return Policy  
- `Doc2.txt` – Employee Leave Policy  

Each document contains structured policy rules and conditions used for retrieval.  
The documents are stored as **TXT files**, parsed and chunked for vectorization.

---

## ⚙️ Methodology

### 1. Document Ingestion
- Loaded all TXT files from the input directory.
- Each document is read and stored with metadata (source filename).

### 2. Text Chunking
- Documents are split into overlapping chunks (250 words with 30-word overlap).
- This improves semantic retrieval and avoids loss of context during embedding.

### 3. Embedding Generation
- Each chunk is converted into a dense vector using:
  - `sentence-transformers/all-MiniLM-L6-v2`
- These embeddings capture the semantic meaning of text.

### 4. Vector Indexing with FAISS
- All embeddings are stored in a FAISS vector index.
- Enables fast similarity search.

### 5. Query Retrieval
- User questions are converted into embeddings.
- FAISS retrieves the **Top-K most relevant document chunks**.

### 6. Answer Generation (RAG Pipeline)
- Retrieved chunks are passed as context to the language model.
- The model generates an answer strictly from document content.
- Output is cleaned to remove hallucinations and prompt leakage.

### 7. API Deployment
- The full pipeline is deployed using **FastAPI**.
- Users interact via a REST API endpoint.
- API testing is done using **Swagger UI**.

---

## 🧠 Technologies Used

- **Programming Language:** Python  
- **Embeddings:** Sentence Transformers (MiniLM)  
- **Vector Database:** FAISS  
- **Language Model:** GPT-2 / DistilGPT-2  
- **Backend:** FastAPI, Uvicorn  
- **Prototyping:** Jupyter Notebook  
- **Storage:** JSON, TXT files  

---

## 📈 Key Functional Results

- Answers policy-related questions using only internal documents.
- Prevents hallucination by grounding answers in retrieved content.
- Supports multi-document knowledge base.
- Exposed through a production-style REST API.
- Demonstrates a full end-to-end RAG implementation.

### ✅ Example

**Question:**  
How many paid leaves do employees get per year?

**Answer:**  
Employees are entitled to 24 paid leaves per year.

---

## 💡 Practical Applications

- HR policy chatbots  
- Customer support automation  
- Internal company knowledge bases  
- Legal & compliance document retrieval  
- Enterprise document intelligence systems  

This system significantly reduces manual search time and improves productivity.

---

## 🚀 Future Enhancements

- Replace GPT-2 with instruction-tuned models (LLaMA, GPT-4, Claude)
- Add PDF ingestion and OCR support
- Add authentication and activity logging
- Deploy using Docker & Cloud (AWS / GCP)
- Add a frontend chat UI (Streamlit or React)
- Enable automatic document re-indexing

---

## ✅ How to Run the Project

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```
### Step 2: Run the FastAPI Server
python -m uvicorn app:app --port 8001

### Step 3: Open in Browser
http://localhost:8001/docs
