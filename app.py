from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import faiss
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM


# ---------- Load FAISS index and chunks ----------

index = faiss.read_index("faiss_index.bin")

with open("chunks.json", "r", encoding="utf-8") as f:
    all_chunks = json.load(f)


# ---------- Load Embedding Model ----------

embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


# ---------- Load LLM Model (GPT-2 for demo) ----------

#llm_model = "gpt2"
#tokenizer = AutoTokenizer.from_pretrained(llm_model)
#model = AutoModelForCausalLM.from_pretrained(llm_model)
#tokenizer.pad_token = tokenizer.eos_token

llm_model = "distilgpt2"   # much smaller & RAM-safe
tokenizer = AutoTokenizer.from_pretrained(llm_model)
model = AutoModelForCausalLM.from_pretrained(
    llm_model,
    low_cpu_mem_usage=True
)
tokenizer.pad_token = tokenizer.eos_token


# ---------- Helper Functions ----------

def retrieve(query: str, k: int = 2):
    q_emb = embed_model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(q_emb, k)

    results = []
    for idx in indices[0]:
        results.append(all_chunks[idx])
    return results


def generate_answer(query: str):
    retrieved_chunks = retrieve(query, k=2)
    context = "\n\n".join([c["chunk"] for c in retrieved_chunks])

    prompt = f"""
You are a helpful assistant. Use ONLY the context below to answer the question.
If the answer is not in the context, say "I don't know".

Context:
{context}

Question: {query}
Answer:
"""

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        padding=True,
        truncation=True
    )

    outputs = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=80,
        num_beams=2,
        no_repeat_ngram_size=2,
        early_stopping=True,
        pad_token_id=tokenizer.eos_token_id
    )

    # ✅ CLEAN OUTPUT (ONLY FINAL ANSWER)
    full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)

    if "Answer:" in full_output:
        clean_answer = full_output.split("Answer:")[-1].strip()
    else:
        clean_answer = full_output.strip()

    sources = list({c["source"] for c in retrieved_chunks})
    return clean_answer, sources


# ---------- FastAPI App ----------

app = FastAPI(title="RAG Policy Chatbot API")


class QueryRequest(BaseModel):
    query: str


class QueryResponse(BaseModel):
    query: str
    answer: str
    sources: List[str]


@app.post("/query", response_model=QueryResponse)
def query_rag(request: QueryRequest):
    answer, sources = generate_answer(request.query)
    return QueryResponse(
        query=request.query,
        answer=answer,
        sources=sources
    )

