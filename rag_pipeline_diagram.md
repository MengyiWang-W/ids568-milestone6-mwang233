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
```

### Key Decision Points
Choose chunk size and overlap
Choose top-k retrieval size
Decide whether retrieved evidence is sufficient for grounded generation


### Data Transformations
Raw documents → text
Text → chunks
Chunks → embeddings
Query → query embedding
Query embedding + FAISS → retrieved chunks
Retrieved chunks + query → grounded prompt
Prompt → final answer