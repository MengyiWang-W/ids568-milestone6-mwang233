
```md
# RAG Pipeline Diagram

## End-to-End Data Flow

```text
Documents (data/)
    |
    v
[Document Loaders]
(txt / pdf / docx)
    |
    v
[Chunking]
(chunk_size, overlap)
    |
    v
[Embedding Model]
(all-MiniLM-L6-v2)
    |
    v
[FAISS Index]
(vector store)
    |
    v
[Retriever]
(top-k similarity search)
    |
    v
[Prompt Builder]
(query + retrieved chunks)
    |
    v
[Ollama LLM]
(qwen2.5:7b)
    |
    v
[Grounded Answer]
(with chunk citations)