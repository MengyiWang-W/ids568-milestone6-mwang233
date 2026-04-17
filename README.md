# IDS568 Milestone 6 - RAG & Agentic Pipeline

## 1. Overview

This repository implements:

- **Part 1:** A complete Retrieval-Augmented Generation (RAG) pipeline
- **Part 2:** A multi-tool agent controller integrating retrieval

All outputs are generated from **real local execution** using a **7B open-weight model (`qwen2.5:7b`)** served through Ollama.

## 2. Environment

### Runtime
- OS: Windows
- Shell: PowerShell
- Python: 3.12

### Models
- Embedding model: `sentence-transformers/all-MiniLM-L6-v2`
- Generator model: `qwen2.5:7b`

### Serving Stack
- Ollama local server (`http://localhost:11434`)
- FAISS vector index

## 3. Setup Instructions
### Step 1 — Install dependencies
```bash
pip install -r requirements.txt
```
### Step 2 — Install Ollama

Download from: https://ollama.com

### Step 3 — Pull model
```bash
ollama pull qwen2.5:7b
```
### Step 4 — Start server
```bash
ollama serve
```
## 4. Usage: RAG Pipeline
Run the final RAG pipeline:
```bash
python rag_pipeline.py
```
Generated outputs:
rag_eval_results.json
chunking_experiment_results.json

## 5. Final RAG Configuration

Chunk size: 512
Overlap: 80
Top-k: 3
Vector index: FAISS
Generator: qwen2.5:7b

## 6. Part 1 Results (Actual Run)
Precision@k: 0.5667
Recall@k: 0.7667
Retrieval latency: 0.0207s
Generation latency: 177.9963s
End-to-end latency: 178.017s

Key Observations
Retrieval is fast and not a bottleneck
Generation dominates latency
Recall > Precision, so retrieval introduces some noise

## 7. Usage: Agent Controller

Run the agent controller:
```bash
python agent_controller.py
```
Generated outputs:

agent_run_summary.json
agent_traces/task_01.json ... agent_traces/task_10.json

Each trace includes:

task description
selected tool
routing reason
retrieved chunk IDs
final answer
latency

## 8. Part 2 Results (Actual Run)
Tool Usage
10/10 tasks used retriever_tool
Failures
1/10 tasks had generation_error (timeout)
Latency
Fastest: 110.1321s
Slowest: 602.0896s
Average: ~241.01s

## 9. Architecture Overview
RAG Pipeline

Documents
→ Chunking
→ Embedding
→ FAISS
→ Retrieval
→ LLM
→ Grounded Answer

Agent Workflow

Task
→ Tool Selection
→ Retrieval
→ Summarization
→ Answer
→ Trace Logging

## 10. Repository Structure

rag_pipeline.py

rag_eval_results.json

chunking_experiment_results.json

rag_evaluation_report.md

rag_pipeline_diagram.md

agent_controller.py

agent_run_summary.json

agent_report.md

agent_traces/

data/

requirements.txt

README.md

.gitignore

## 11. Model Deployment Notes

Model: qwen2.5:7b
Size: 7B
Serving: Ollama local server
Inference: local CPU

Observed Performance:

Typical generation latency: 100–600 seconds
Performance varies depending on prompt length and retrieved context size

## 12. Limitations
Local LLM inference is slow and unstable
No reranking after retrieval
Small course-specific corpus
Simple agent routing policy
Evaluation uses source-level labels

## 13. Failure Analysis Summary
One task failed due to LLM generation timeout (600s)
Retrieval succeeded but generation did not complete
This demonstrates the difference between retrieval success and generation failure

## 14. Reproducibility
pip install -r requirements.txt
ollama pull qwen2.5:7b
ollama serve
python rag_pipeline.py
python agent_controller.py

All outputs can be reproduced locally.

## 15. Notes
No mock outputs are used
No proprietary APIs are used
All results come from real execution
Retrieval is always used for document-grounded tasks