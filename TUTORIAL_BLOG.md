# Building an Industry-Standard RAG System for Resume Selection

In this tutorial, we will walk through building a production-ready Retrieval-Augmented Generation (RAG) system for selecting resumes. We'll cover advanced techniques like **Hybrid Search**, **Reranking**, and **Agentic Workflows** using LangChain, Pinecone, and Google Gemini (or OpenAI).

## Overview

The goal is to create a system that can:
1. **Ingest** PDF resumes and index them efficiently.
2. **Retrieve** the most relevant candidates using semantic understanding *and* keyword matching.
3. **Reason** about the candidates using an LLM Agent to answer complex queries.

## Architecture

1. **Data Ingestion**: 
   - Load PDFs -> Chunk Text -> Generate Embeddings (Dense + Sparse).
   - Store in **Pinecone** (Vector DB).
2. **Retrieval Engine**:
   - **Hybrid Search**: Combines Dense Vectors (semantic) + Sparse Vectors (BM25 keywords).
   - **Reranking**: Uses a Cross-Encoder to re-score top results for high precision.
3. **Agentic Layer**:
   - **LangChain Agent**: Orchestrates the process, deciding when to search and how to answer user questions.

---

## Step 1: Data Ingestion & Chunking

We use `PyPDFLoader` to read resumes and `RecursiveCharacterTextSplitter` to chunk them. Chunking is critical to ensure we capture enough context without exceeding token limits.

**Key Concept: Hybrid Embeddings**
We generate two types of vectors for each chunk:
- **Dense Vectors**: (e.g., `all-MiniLM-L6-v2` or OpenAI/Gemini) capture *meaning* (e.g., "ML" is related to "AI").
- **Sparse Vectors**: (BM25) capture *exact keywords* (e.g., "Python", "C++").

This ensures we don't miss candidates who have the exact skills listed, even if the semantic context is slightly different.

```python
# src/ingestion.py snippet
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
bm25 = BM25Encoder()
bm25.fit(chunk_texts)

# Upsert to Pinecone with both vectors
index.upsert(vectors=[{
    "id": id, 
    "values": dense_vec, 
    "sparse_values": sparse_vec, 
    "metadata": metadata
}])
```

## Step 2: Advanced Retrieval

Retrieving the right documents is the heart of RAG. We use a multi-stage approach.

### 1. Query Enrichment
Users often write short queries like "python dev". We use an LLM to expand this into:
- "Python developer with Django experience"
- "Backend engineer Python"
- "Software developer Python"

### 2. Hybrid Search
We query Pinecone using both dense and sparse vectors. This gives us a broad set of potentially relevant candidates.

### 3. Reranking
The top results from the vector DB might still be noisy. We use a **Cross-Encoder** (e.g., `ms-marco-MiniLM`) to read the query and the document *together* and output a relevance score. This is slower but much more accurate.

```python
# src/retrieval.py snippet
reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
scores = reranker.predict(pairs)
```

## Step 3: The Agentic Framework

Instead of a simple Q&A chain, we build an **Agent**. An agent has access to "Tools" (like our retrieval function) and can decide how to use them.

We use LangChain's `create_react_agent`. The agent follows a "Thought -> Action -> Observation" loop.

**Example Flow:**
1. **User**: "Find a senior accountant with audit experience."
2. **Agent Thought**: "I need to search for resumes matching this description."
3. **Action**: Calls `SearchResumes("senior accountant audit")`.
4. **Observation**: Gets back 5 candidate snippets.
5. **Agent Thought**: "I have the candidates. I will summarize their skills."
6. **Final Answer**: "Here are the top candidates..."

## Conclusion

By combining Hybrid Search, Reranking, and Agents, we move beyond simple vector search to a robust, industry-standard RAG system. This architecture ensures high recall (finding all relevant docs) and high precision (showing only the best ones).

### Future Improvements
- **Metadata Filtering**: Filter by years of experience or location before searching.
- **Evaluation**: Use RAGAS to strictly evaluate retrieval quality.
- **Multimodal**: Process candidate profile pictures or portfolio links.
