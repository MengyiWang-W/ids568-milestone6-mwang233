import os
import time
import json
import glob
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Set

import faiss
import numpy as np
import requests
from pypdf import PdfReader
from docx import Document
from sentence_transformers import SentenceTransformer


# =========================
# Configuration
# =========================
DATA_DIR = "data"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
OLLAMA_MODEL = "qwen2.5:7b"
OLLAMA_URL = "http://localhost:11434/api/generate"

TOP_K = 3

# Final run settings
DEBUG_MODE = False
RUN_CHUNKING_EXPERIMENTS = False

DEFAULT_CHUNK_SIZE = 512
DEFAULT_CHUNK_OVERLAP = 80

RAG_OUTPUT_RESULTS_FILE = "rag_eval_results.json"
CHUNKING_OUTPUT_RESULTS_FILE = "chunking_experiment_results.json"


@dataclass
class Chunk:
    chunk_id: str
    source: str
    text: str


# =========================
# File Readers
# =========================
def read_txt(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def read_pdf(path: str) -> str:
    reader = PdfReader(path)
    pages = []
    for page in reader.pages:
        pages.append(page.extract_text() or "")
    return "\n".join(pages)


def read_docx(path: str) -> str:
    doc = Document(path)
    return "\n".join([p.text for p in doc.paragraphs])


def load_documents(data_dir: str) -> List[Dict[str, str]]:
    docs = []

    for path in glob.glob(os.path.join(data_dir, "*")):
        if path.endswith(".txt"):
            text = read_txt(path)
        elif path.endswith(".pdf"):
            text = read_pdf(path)
        elif path.endswith(".docx"):
            text = read_docx(path)
        else:
            continue

        text = text.strip()
        if not text:
            print(f"Warning: empty or unreadable file skipped -> {os.path.basename(path)}")
            continue

        docs.append({
            "source": os.path.basename(path),
            "text": text
        })

    return docs


# =========================
# Chunking
# =========================
def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end]).strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(words):
            break

        start = max(0, end - overlap)

    return chunks


def build_chunks(documents: List[Dict[str, str]], chunk_size: int, overlap: int) -> List[Chunk]:
    all_chunks = []
    idx = 0

    for doc in documents:
        pieces = chunk_text(doc["text"], chunk_size, overlap)
        for piece in pieces:
            all_chunks.append(
                Chunk(
                    chunk_id=f"chunk_{idx}",
                    source=doc["source"],
                    text=piece
                )
            )
            idx += 1

    return all_chunks


# =========================
# Embedding + FAISS
# =========================
def embed_chunks(model: SentenceTransformer, chunks: List[Chunk]) -> np.ndarray:
    texts = [c.text for c in chunks]
    vectors = model.encode(
        texts,
        convert_to_numpy=True,
        show_progress_bar=True
    )
    return vectors.astype("float32")


def build_faiss_index(vectors: np.ndarray) -> faiss.Index:
    dim = vectors.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(vectors)
    return index


def retrieve(
    query: str,
    model: SentenceTransformer,
    index: faiss.Index,
    chunks: List[Chunk],
    top_k: int = 3
) -> List[Dict[str, Any]]:
    qvec = model.encode([query], convert_to_numpy=True).astype("float32")
    distances, indices = index.search(qvec, top_k)

    results = []
    for rank, idx in enumerate(indices[0]):
        if idx < 0 or idx >= len(chunks):
            continue

        c = chunks[idx]
        results.append({
            "rank": rank + 1,
            "chunk_id": c.chunk_id,
            "source": c.source,
            "text": c.text,
            "distance": float(distances[0][rank])
        })

    return results


# =========================
# LLM Generation
# =========================
def generate_answer(query: str, retrieved_chunks: List[Dict[str, Any]]) -> str:
    context = "\n\n".join(
        f"[{item['chunk_id']} | {item['source']}]\n{item['text']}"
        for item in retrieved_chunks
    )

    prompt = f"""
You are a grounded RAG assistant.

Answer ONLY using the retrieved context below.
If the context is insufficient, say exactly:
Insufficient evidence in retrieved context.

Rules:
- Use only the retrieved context.
- Do not add outside knowledge.
- Keep the answer under 60 words.
- End with a complete sentence.
- Cite chunk IDs explicitly in parentheses.
- If the retrieved context contains conflicting claims, mention the conflict clearly.

Question:
{query}

Retrieved Context:
{context}

Answer:
""".strip()

    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "num_predict": 80,
                    "temperature": 0
                }
            },
            timeout=600
        )
        response.raise_for_status()
        data = response.json()
        return data.get("response", "").strip()

    except Exception as e:
        return f"LLM_ERROR: {str(e)}"


# =========================
# Metrics
# =========================
def precision_at_k(retrieved_sources: List[str], relevant_sources: Set[str], k: int) -> float:
    top_k = retrieved_sources[:k]
    if k == 0:
        return 0.0
    hits = sum(1 for src in top_k if src in relevant_sources)
    return hits / k


def recall_at_k(retrieved_sources: List[str], relevant_sources: Set[str], k: int) -> float:
    if not relevant_sources:
        return 0.0
    top_k = retrieved_sources[:k]
    hits = sum(1 for src in top_k if src in relevant_sources)
    return hits / len(relevant_sources)


# =========================
# Evaluation Queries
# Update filenames if your actual names differ
# =========================
def build_evaluation_queries() -> List[Dict[str, Any]]:
    all_queries = [
        {
            "query": "What is retrieval-augmented generation?",
            "relevant_sources": {"module7-slides.pdf", "RAG Overview.txt"}
        },
        {
            "query": "What are the main components of a RAG system?",
            "relevant_sources": {"module7-slides.pdf", "RAG Overview.txt"}
        },
        {
            "query": "How does a retriever work in RAG?",
            "relevant_sources": {"module7-slides.pdf", "RAG Overview.txt"}
        },
        {
            "query": "What is the role of embeddings in RAG?",
            "relevant_sources": {"module7-slides.pdf", "RAG Overview.txt"}
        },
        {
            "query": "What are the advantages of RAG over fine-tuning?",
            "relevant_sources": {"RAG Overview.txt", "module7-slides.pdf"}
        },
        {
            "query": "What are the limitations of RAG systems?",
            "relevant_sources": {"RAG Overview.txt", "Conflicting Views on RAG Effectiveness.txt"}
        },
        {
            "query": "Why can RAG fail even when relevant information exists?",
            "relevant_sources": {
                "Homework 7.1 - Retrieval Stress-Test & Failure Analysis.pdf",
                "Conflicting Views on RAG Effectiveness.txt",
                "RAG Overview.txt"
            }
        },
        {
            "query": "What are common retrieval failure modes in RAG?",
            "relevant_sources": {
                "Homework 7.1 - Retrieval Stress-Test & Failure Analysis.pdf",
                "module7-slides.pdf"
            }
        },
        {
            "query": "How do architecture choices affect groundedness in RAG?",
            "relevant_sources": {
                "Homework 7.2 - RAG Architecture Comparison.pdf",
                "module7-slides.pdf",
                "Conflicting Views on RAG Effectiveness.txt"
            }
        },
        {
            "query": "How should an agent use retrieval in a multi-step workflow?",
            "relevant_sources": {
                "Homework 7.3 - Agentic Task Decomposition.pdf",
                "module7-slides.pdf"
            }
        }
    ]

    if DEBUG_MODE:
        return all_queries[:3]

    return all_queries


# =========================
# Evaluation Pipeline
# =========================
def evaluate_pipeline(
    eval_queries: List[Dict[str, Any]],
    embed_model: SentenceTransformer,
    index: faiss.Index,
    chunks: List[Chunk],
    top_k: int = 3
) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
    results = []

    for item in eval_queries:
        query = item["query"]
        relevant_sources = set(item["relevant_sources"])

        retrieval_start = time.perf_counter()
        retrieved = retrieve(query, embed_model, index, chunks, top_k=top_k)
        retrieval_end = time.perf_counter()

        generation_start = time.perf_counter()
        answer = generate_answer(query, retrieved)
        generation_end = time.perf_counter()

        retrieved_sources = [r["source"] for r in retrieved]
        p_at_k = precision_at_k(retrieved_sources, relevant_sources, top_k)
        r_at_k = recall_at_k(retrieved_sources, relevant_sources, top_k)

        failure_type = "none"
        if answer.startswith("LLM_ERROR:"):
            failure_type = "generation_error"
        elif "Insufficient evidence in retrieved context." in answer and len(retrieved) > 0:
            failure_type = "possible_generation_or_grounding_failure"
        elif p_at_k == 0.0:
            failure_type = "retrieval_failure"

        results.append({
            "query": query,
            "relevant_sources": sorted(list(relevant_sources)),
            "retrieved": retrieved,
            "answer": answer,
            "precision_at_k": round(p_at_k, 4),
            "recall_at_k": round(r_at_k, 4),
            "retrieval_latency_sec": round(retrieval_end - retrieval_start, 4),
            "generation_latency_sec": round(generation_end - generation_start, 4),
            "end_to_end_latency_sec": round(
                (retrieval_end - retrieval_start) + (generation_end - generation_start), 4
            ),
            "failure_type": failure_type
        })

    summary = {
        "avg_precision_at_k": round(float(np.mean([r["precision_at_k"] for r in results])), 4) if results else 0.0,
        "avg_recall_at_k": round(float(np.mean([r["recall_at_k"] for r in results])), 4) if results else 0.0,
        "avg_retrieval_latency_sec": round(float(np.mean([r["retrieval_latency_sec"] for r in results])), 4) if results else 0.0,
        "avg_generation_latency_sec": round(float(np.mean([r["generation_latency_sec"] for r in results])), 4) if results else 0.0,
        "avg_end_to_end_latency_sec": round(float(np.mean([r["end_to_end_latency_sec"] for r in results])), 4) if results else 0.0
    }

    return results, summary


# =========================
# Save Results
# =========================
def save_rag_results(
    results: List[Dict[str, Any]],
    summary: Dict[str, float],
    path: str,
    chunk_size: int,
    overlap: int
) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "config": {
                    "data_dir": DATA_DIR,
                    "embedding_model": EMBED_MODEL_NAME,
                    "ollama_model": OLLAMA_MODEL,
                    "chunk_size": chunk_size,
                    "chunk_overlap": overlap,
                    "top_k": TOP_K,
                    "debug_mode": DEBUG_MODE
                },
                "summary": summary,
                "results": results
            },
            f,
            indent=2,
            ensure_ascii=False
        )


# =========================
# One Standard Pipeline Run
# =========================
def run_single_pipeline(chunk_size: int, overlap: int) -> Dict[str, Any]:
    print("=== RAG Pipeline Start ===")

    print("\n[1] Loading documents...")
    documents = load_documents(DATA_DIR)
    print(f"Loaded {len(documents)} documents")
    if not documents:
        raise RuntimeError("No readable documents found in the data/ folder.")

    print("\n[2] Chunking documents...")
    chunks = build_chunks(documents, chunk_size, overlap)
    print(f"Built {len(chunks)} chunks")
    if not chunks:
        raise RuntimeError("No chunks were created. Check your document contents.")

    print("\n[3] Loading embedding model...")
    embed_model = SentenceTransformer(EMBED_MODEL_NAME)

    print("\n[4] Embedding chunks...")
    t0 = time.perf_counter()
    vectors = embed_chunks(embed_model, chunks)
    t1 = time.perf_counter()
    print(f"Embedding time: {t1 - t0:.4f}s")

    print("\n[5] Building FAISS index...")
    t2 = time.perf_counter()
    index = build_faiss_index(vectors)
    t3 = time.perf_counter()
    print(f"Index build time: {t3 - t2:.4f}s")

    print("\n[6] Running one sample query...")
    sample_query = "What is retrieval-augmented generation?"
    sample_retrieved = retrieve(sample_query, embed_model, index, chunks, TOP_K)
    sample_answer = generate_answer(sample_query, sample_retrieved)

    print(f"\nSample Query: {sample_query}")
    print("\nRetrieved Chunks:")
    for item in sample_retrieved:
        print(
            f"- rank={item['rank']} | {item['chunk_id']} | "
            f"{item['source']} | distance={item['distance']:.4f}"
        )

    print("\nSample Answer:")
    print(sample_answer)

    print(f"\n[7] Running evaluation on {'3' if DEBUG_MODE else '10'} handcrafted queries...")
    eval_queries = build_evaluation_queries()
    results, summary = evaluate_pipeline(eval_queries, embed_model, index, chunks, top_k=TOP_K)

    print("\n=== Evaluation Summary ===")
    for k, v in summary.items():
        print(f"{k}: {v}")

    print(f"\n[8] Saving results to {RAG_OUTPUT_RESULTS_FILE} ...")
    save_rag_results(results, summary, RAG_OUTPUT_RESULTS_FILE, chunk_size, overlap)
    print("Done.")

    print("\n=== RAG Pipeline Complete ===")

    return {
        "chunk_size": chunk_size,
        "overlap": overlap,
        **summary
    }


# =========================
# Chunking Experiments
# =========================
def run_chunking_experiments() -> None:
    print("=== Chunking Experiment Start ===")

    configs = [
        {"chunk_size": 256, "overlap": 50},
        {"chunk_size": 512, "overlap": 80},
        {"chunk_size": 1024, "overlap": 120}
    ]

    all_results = []

    for cfg in configs:
        print("\n" + "=" * 60)
        print(f"Running config: chunk_size={cfg['chunk_size']}, overlap={cfg['overlap']}")
        print("=" * 60)

        summary = run_single_pipeline(
            chunk_size=cfg["chunk_size"],
            overlap=cfg["overlap"]
        )

        all_results.append(summary)

    with open(CHUNKING_OUTPUT_RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print("\n=== Chunking Comparison Summary ===")
    for item in all_results:
        print(item)

    print(f"\nSaved chunking experiment results to {CHUNKING_OUTPUT_RESULTS_FILE}")


# =========================
# Entry Point
# =========================
if __name__ == "__main__":
    if RUN_CHUNKING_EXPERIMENTS:
        run_chunking_experiments()
    else:
        run_single_pipeline(
            chunk_size=DEFAULT_CHUNK_SIZE,
            overlap=DEFAULT_CHUNK_OVERLAP
        )