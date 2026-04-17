# Agent Report

## 1. Overview

This project implements a multi-tool agent controller that integrates retrieval into a grounded reasoning workflow.

The agent uses a real open-weight 7B model (`qwen2.5:7b`) via Ollama and coordinates multiple tools to solve tasks.

---

## 2. Tools

### 2.1 Retriever Tool

- Uses FAISS index from Part 1
- Returns top-k relevant chunks
- Provides source and chunk IDs

### 2.2 Summarizer Tool

- Uses LLM to generate grounded answers
- Constrained to:
  - use only retrieved context
  - cite chunk IDs
  - detect insufficient evidence
  - identify conflicts

---

## 3. Tool Selection Policy

The agent uses a **retrieval-first routing policy**:

- All document-grounded tasks are routed to `retriever_tool`
- Retrieved context is passed to `summarizer_tool`
- Final answers are generated using only retrieved evidence

👉 Important design choice:

Although multiple tools exist, **all evaluation tasks require document-grounded responses**, so the agent consistently selects retrieval.

This is a deliberate design decision to ensure grounding reliability and traceability.

---

## 4. Retrieval Integration

Workflow for each task:

1. Task received
2. Tool selection (retrieval-first)
3. Retrieve top-k chunks
4. Generate grounded answer
5. Save trace

Each trace includes:

- selected tool
- reasoning
- retrieved chunks
- final answer
- latency

---

## 5. Evaluation Tasks

10 tasks covering:

- definitions
- system components
- embeddings
- failure modes
- conflicting claims
- architecture reasoning
- agent workflows

---

## 6. Results

| Task | Tool | Failure | Latency (s) |
|------|------|--------|------------|
| 1 | retriever | generation_error | 602.0896 |
| 2 | retriever | none | 470.0886 |
| 3 | retriever | none | 184.3872 |
| 4 | retriever | none | 133.9302 |
| 5 | retriever | none | 134.0634 |
| 6 | retriever | none | 137.3224 |
| 7 | retriever | none | 137.9220 |
| 8 | retriever | none | 189.3721 |
| 9 | retriever | none | 310.7505 |
|10 | retriever | none | 110.1321 |

### Summary

- 10/10 tasks used retrieval
- 1/10 task had generation error (timeout)
- Average latency ≈ 241s

---

## 7. Performance Analysis

- Retrieval is fast and consistent
- Generation is slow and dominates latency
- Larger prompts increase runtime significantly

The retrieval-first policy improves grounding consistency but increases latency.

---

## 8. Failure Analysis

### Generation Failure

- Task 1 failed due to **LLM timeout (600s)**
- Retrieval succeeded but generation did not complete

### Retrieval Dependence

- Answer quality depends heavily on retrieved chunks
- Noisy retrieval leads to weaker answers

### Conservative Outputs

- Some responses are overly cautious due to strict grounding rules

---

## 9. Model Impact

Using a local 7B model:

Pros:
- Fully reproducible
- No external API dependency
- Good grounding behavior

Cons:
- Slow inference
- Occasional instability (timeouts)

---

## 10. Limitations

- No advanced planning or memory
- No reranking
- Static routing policy
- High latency due to local inference

---

## 11. Conclusion

The agent successfully demonstrates:

- Multi-tool workflow
- Retrieval as a decision-triggered tool
- Transparent reasoning through traces
- Grounded answer generation

The system prioritizes correctness and traceability over speed, aligning with the requirements of document-grounded agent systems.