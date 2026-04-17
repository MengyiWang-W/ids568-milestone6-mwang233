import os
import json
import time
from typing import List, Dict, Any

import requests
from sentence_transformers import SentenceTransformer

from rag_pipeline import (
    DATA_DIR,
    EMBED_MODEL_NAME,
    OLLAMA_MODEL,
    OLLAMA_URL,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CHUNK_OVERLAP,
    TOP_K,
    load_documents,
    build_chunks,
    embed_chunks,
    build_faiss_index,
    retrieve
)

TRACE_DIR = "agent_traces"
SUMMARY_FILE = "agent_run_summary.json"


# =========================
# Utility
# =========================
def ensure_trace_dir() -> None:
    os.makedirs(TRACE_DIR, exist_ok=True)


def call_ollama(prompt: str, num_predict: int = 60, timeout: int = 600) -> str:
    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "num_predict": num_predict,
                    "temperature": 0
                }
            },
            timeout=timeout
        )
        response.raise_for_status()
        data = response.json()
        return data.get("response", "").strip()
    except Exception as e:
        return f"LLM_ERROR: {str(e)}"


# =========================
# Shared Retrieval State
# =========================
def initialize_retrieval_state() -> Dict[str, Any]:
    print("[Init] Loading documents...")
    documents = load_documents(DATA_DIR)
    if not documents:
        raise RuntimeError("No readable documents found in data/")

    print("[Init] Building chunks...")
    chunks = build_chunks(documents, DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP)
    if not chunks:
        raise RuntimeError("No chunks created from documents.")

    print("[Init] Loading embedding model...")
    embed_model = SentenceTransformer(EMBED_MODEL_NAME)

    print("[Init] Embedding chunks...")
    vectors = embed_chunks(embed_model, chunks)

    print("[Init] Building FAISS index...")
    index = build_faiss_index(vectors)

    return {
        "documents": documents,
        "chunks": chunks,
        "embed_model": embed_model,
        "index": index
    }


# =========================
# Tools
# =========================
def retriever_tool(query: str, state: Dict[str, Any], top_k: int = TOP_K) -> List[Dict[str, Any]]:
    return retrieve(
        query=query,
        model=state["embed_model"],
        index=state["index"],
        chunks=state["chunks"],
        top_k=top_k
    )


def summarizer_tool(task: str, retrieved_chunks: List[Dict[str, Any]]) -> str:
    context = "\n\n".join(
        f"[{c['chunk_id']} | {c['source']}]\n{c['text']}"
        for c in retrieved_chunks
    )

    prompt = f"""
You are a grounded summarization tool inside an agent pipeline.

Task:
{task}

Retrieved Context:
{context}

Instructions:
- Answer ONLY using the retrieved context.
- Keep the answer under 60 words.
- Cite chunk IDs in parentheses.
- If evidence is insufficient, say exactly:
Insufficient evidence in retrieved context.
- If there are conflicting claims, explicitly say there is a conflict.
- Do not use outside knowledge.

Answer:
""".strip()

    return call_ollama(prompt, num_predict=70, timeout=600)


def direct_reasoning_tool(task: str) -> str:
    prompt = f"""
You are a lightweight reasoning tool inside an agent pipeline.

Task:
{task}

Instructions:
- Keep the answer under 25 words.
- Only use this for meta-level planning or classification.
- If the task requires document evidence, say exactly:
Use retriever first.

Answer:
""".strip()

    return call_ollama(prompt, num_predict=35, timeout=300)


# =========================
# Safer Tool Selection
# =========================
def choose_tool(task: str) -> Dict[str, str]:
    """
    Safer routing policy:
    - Always use retrieval for document-grounded tasks
    - Avoid direct reasoning unless strictly necessary
    """

    task_lower = task.lower()

    retrieval_keywords = [
        "explain",
        "summarize",
        "identify",
        "describe",
        "compare",
        "what",
        "why",
        "how",
        "role",
        "failure",
        "grounded",
        "retrieval",
        "rag",
        "architecture",
        "components",
        "hallucination",
        "workflow",
        "embeddings"
    ]

    if any(k in task_lower for k in retrieval_keywords):
        return {
            "tool": "retriever_tool",
            "reason": "Requires document-grounded information via retrieval."
        }

    return {
        "tool": "retriever_tool",
        "reason": "Defaulting to retrieval for safety and traceable grounding."
    }
# =========================
# Agent Core
# =========================
def run_agent_task(task_id: int, task: str, state: Dict[str, Any]) -> Dict[str, Any]:
    trace: Dict[str, Any] = {
        "task_id": task_id,
        "task": task
    }

    total_start = time.perf_counter()

    route_start = time.perf_counter()
    route_decision = choose_tool(task)
    route_end = time.perf_counter()

    trace["routing"] = {
        "selected_tool": route_decision["tool"],
        "reason": route_decision["reason"],
        "latency_sec": round(route_end - route_start, 4)
    }

    retrieved_chunks: List[Dict[str, Any]] = []
    final_answer = ""
    failure_type = "none"
    tool_calls: List[Dict[str, Any]] = []

    if route_decision["tool"] == "direct_reasoning_tool":
        direct_start = time.perf_counter()
        direct_answer = direct_reasoning_tool(task)
        direct_end = time.perf_counter()

        tool_calls.append({
            "tool": "direct_reasoning_tool",
            "input": task,
            "output": direct_answer,
            "latency_sec": round(direct_end - direct_start, 4)
        })

        if direct_answer.strip() == "Use retriever first.":
            retrieval_start = time.perf_counter()
            retrieved_chunks = retriever_tool(task, state, top_k=TOP_K)
            retrieval_end = time.perf_counter()

            summary_start = time.perf_counter()
            final_answer = summarizer_tool(task, retrieved_chunks)
            summary_end = time.perf_counter()

            tool_calls.append({
                "tool": "retriever_tool",
                "input": task,
                "retrieved_sources": [c["source"] for c in retrieved_chunks],
                "retrieved_chunk_ids": [c["chunk_id"] for c in retrieved_chunks],
                "latency_sec": round(retrieval_end - retrieval_start, 4)
            })
            tool_calls.append({
                "tool": "summarizer_tool",
                "input": task,
                "output": final_answer,
                "latency_sec": round(summary_end - summary_start, 4)
            })
        else:
            final_answer = direct_answer

    else:
        retrieval_start = time.perf_counter()
        retrieved_chunks = retriever_tool(task, state, top_k=TOP_K)
        retrieval_end = time.perf_counter()

        summary_start = time.perf_counter()
        final_answer = summarizer_tool(task, retrieved_chunks)
        summary_end = time.perf_counter()

        tool_calls.append({
            "tool": "retriever_tool",
            "input": task,
            "retrieved_sources": [c["source"] for c in retrieved_chunks],
            "retrieved_chunk_ids": [c["chunk_id"] for c in retrieved_chunks],
            "latency_sec": round(retrieval_end - retrieval_start, 4)
        })
        tool_calls.append({
            "tool": "summarizer_tool",
            "input": task,
            "output": final_answer,
            "latency_sec": round(summary_end - summary_start, 4)
        })

    if isinstance(final_answer, str) and final_answer.startswith("LLM_ERROR:"):
        failure_type = "generation_error"
    elif final_answer.strip() == "Insufficient evidence in retrieved context.":
        failure_type = "possible_retrieval_or_grounding_failure"

    total_end = time.perf_counter()

    trace["tool_calls"] = tool_calls
    trace["final_answer"] = final_answer
    trace["failure_type"] = failure_type
    trace["total_latency_sec"] = round(total_end - total_start, 4)

    trace_path = os.path.join(TRACE_DIR, f"task_{task_id:02d}.json")
    with open(trace_path, "w", encoding="utf-8") as f:
        json.dump(trace, f, indent=2, ensure_ascii=False)

    return trace


# =========================
# Tasks
# =========================
def build_agent_tasks() -> List[str]:
    return [
        "Explain what retrieval-augmented generation is and summarize its main benefit.",
        "Summarize the main components of a RAG system.",
        "Explain the role of embeddings in RAG.",
        "Compare the advantages and limitations of RAG.",
        "Identify common retrieval failure modes in RAG.",
        "Explain why RAG can still fail even when relevant information exists.",
        "Summarize the conflicting views on whether RAG eliminates hallucination.",
        "Explain how architecture choices affect groundedness in RAG.",
        "Describe how an agent should use retrieval in a multi-step workflow.",
        "Give a short evidence-grounded explanation of why retrieval quality matters for generation quality."
    ]


# =========================
# Main
# =========================
def main() -> None:
    ensure_trace_dir()
    state = initialize_retrieval_state()
    tasks = build_agent_tasks()

    print("\n=== Agent Controller Start ===")
    print(f"Running {len(tasks)} tasks...")

    all_summaries = []

    for i, task in enumerate(tasks, start=1):
        print(f"\n--- Task {i:02d} ---")
        print(task)

        trace = run_agent_task(i, task, state)

        print("Selected tool:", trace["routing"]["selected_tool"])
        print("Final answer:", trace["final_answer"][:300] if isinstance(trace["final_answer"], str) else trace["final_answer"])
        print("Saved trace:", os.path.join(TRACE_DIR, f"task_{i:02d}.json"))

        all_summaries.append({
            "task_id": i,
            "task": task,
            "selected_tool": trace["routing"]["selected_tool"],
            "failure_type": trace["failure_type"],
            "total_latency_sec": trace["total_latency_sec"]
        })

    with open(SUMMARY_FILE, "w", encoding="utf-8") as f:
        json.dump(all_summaries, f, indent=2, ensure_ascii=False)

    print("\n=== Agent Controller Complete ===")
    print(f"Saved {SUMMARY_FILE}")
    print(f"Saved traces in {TRACE_DIR}/")


if __name__ == "__main__":
    main()