# RAG Evaluation Report

## 1. System Setup

This project implements a complete Retrieval-Augmented Generation (RAG) pipeline using:

- Document ingestion from local files in `data/`
- Chunking with configurable chunk size and overlap
- Sentence embeddings from `sentence-transformers/all-MiniLM-L6-v2`
- FAISS as the vector index
- A real open-weight 7B model (`qwen2.5:7b`) served via Ollama

The system performs retrieval and grounded generation using a local LLM, without any mock or template-based outputs.

---

## 2. Corpus Description

The corpus contains six documents related to RAG and agent systems:

- module7-slides.pdf
- module6-slides.pdf
- Homework 7.1 - Retrieval Stress-Test & Failure Analysis.pdf
- Homework 7.2 - RAG Architecture Comparison.pdf
- Conflicting Views on RAG Effectiveness.txt
- RAG Overview.txt

These documents include definitions, architecture descriptions, failure modes, and conflicting perspectives, enabling both factual retrieval and grounding evaluation.

---

## 3. Chunking and Indexing Design

Three chunking configurations were evaluated:

| Chunk Size | Overlap | Precision@k | Recall@k | Retrieval Latency | Generation Latency | End-to-End |
|-----------|--------:|------------:|---------:|------------------:|-------------------:|-----------:|
| 256       | 50      | 0.3333      | 0.5000   | 0.0226s           | 73.0063s           | 73.0288s   |
| 512       | 80      | 0.4444      | 0.6667   | 0.3566s           | 434.9039s          | 435.2605s  |
| 1024      | 120     | 0.3333      | 0.5000   | 0.1947s           | 318.8697s          | 319.0643s  |

### Interpretation

- **256/50**: Too small → fragmented context → lower recall  
- **1024/120**: Too large → noisy context → reduced precision  
- **512/80**: Best balance → highest precision and recall  

👉 Final configuration selected: **512 / 80**

---

## 4. Retrieval Evaluation (10 Queries)

Final results:

- Precision@k: **0.5667**
- Recall@k: **0.7667**
- Retrieval latency: **0.0207s**
- Generation latency: **177.9963s**
- End-to-end latency: **178.017s**

### Interpretation

- High recall indicates relevant documents are usually retrieved  
- Lower precision shows some irrelevant chunks are included  
- Retrieval is fast and not a bottleneck  
- Generation dominates runtime

---

## 5. Grounding Analysis

The system generally produces grounded responses using retrieved context.

- Answers include chunk citations
- Responses follow the constraint of using only retrieved information

However, some limitations are observed:

- The model sometimes produces **overly conservative answers**
- Phrases like *“Insufficient evidence in retrieved context”* appear even when partial information exists
- Conflicting documents require the model to explicitly mention disagreement

👉 Overall:  
The system **generally uses retrieved evidence**, but grounding quality still depends on retrieval quality and prompt constraints.

---

## 6. Error Attribution

### Retrieval Failures

- Occur when relevant chunks are not ranked in top-k
- More common in smaller chunk sizes (256)

### Generation / Grounding Issues

- Overly cautious responses due to strict prompting
- Occasional incomplete synthesis of retrieved information

### Mixed Cases

- When documents contain conflicting claims, answers depend on:
  - correct retrieval
  - correct conflict interpretation

---

## 7. Latency Analysis

- Retrieval latency: **~0.02s**
- Generation latency: **~178s**
- End-to-end latency ≈ generation latency

👉 Conclusion:
- FAISS retrieval is efficient
- Local LLM inference is the main bottleneck

---

## 8. Final Design Choice

Final configuration:

- Chunk size: 512
- Overlap: 80
- Embedding model: all-MiniLM-L6-v2
- Vector index: FAISS
- Generator: qwen2.5:7b (Ollama)

This configuration provides the best trade-off between retrieval quality and coherence.

---

## 9. Limitations

- Local 7B inference is slow
- No reranking step
- Small corpus size
- Source-level evaluation (not chunk-level)
- Strict prompting may reduce answer completeness

---

## 10. Summary

This RAG pipeline demonstrates:

- Full ingestion → retrieval → generation workflow
- Quantitative evaluation using precision/recall
- Grounded responses with citation
- Clear trade-offs between chunking strategies and performance

The system is fully reproducible and uses a real open-weight LLM.