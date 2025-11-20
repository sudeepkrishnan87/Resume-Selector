import os
import time
from typing import List

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from pinecone import Pinecone, ServerlessSpec
from pinecone_text.sparse import BM25Encoder

# Load environment variables
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("INDEX_NAME", "resume-index")

if not PINECONE_API_KEY:
    print("WARNING: PINECONE_API_KEY is missing")

def load_documents(directory: str) -> List:
    """Loads all PDF documents from the specified directory."""
    documents = []
    for filename in os.listdir(directory):
        if filename.endswith(".pdf"):
            filepath = os.path.join(directory, filename)
            print(f"Loading {filename}...")
            loader = PyPDFLoader(filepath)
            documents.extend(loader.load())
    print(f"Loaded {len(documents)} pages from {len(os.listdir(directory))} files.")
    return documents

def chunk_documents(documents: List) -> List:
    """Chunks documents into smaller pieces."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks.")
    return chunks

def setup_pinecone_index(pc: Pinecone, index_name: str):
    """Creates the Pinecone index if it doesn't exist."""
    existing_indexes = [i.name for i in pc.list_indexes()]
    
    if index_name in existing_indexes:
        print(f"Deleting existing index '{index_name}' to ensure correct dimensions...")
        pc.delete_index(index_name)
        time.sleep(5) # Wait for deletion

    print(f"Creating index '{index_name}'...")
    pc.create_index(
        name=index_name,
        dimension=384,  # all-MiniLM-L6-v2 dimension
        metric="dotproduct", # Recommended for hybrid search
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )
    # Wait for index to be ready
    while not pc.describe_index(index_name).status['ready']:
        time.sleep(1)
    print("Index created.")

def ingest_data():
    """Main ingestion function."""
    # 1. Load Documents
    resume_dir = os.path.abspath("resume_dir")
    if not os.path.exists(resume_dir):
        print(f"Directory {resume_dir} not found.")
        return

    docs = load_documents(resume_dir)
    
    # 2. Chunk Documents
    chunks = chunk_documents(docs)
    
    # 3. Initialize Embeddings & Pinecone
    # Using Local HuggingFace Embeddings (all-MiniLM-L6-v2)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    bm25 = BM25Encoder()
    
    # Fit BM25 on the corpus
    chunk_texts = [chunk.page_content for chunk in chunks]
    bm25.fit(chunk_texts)
    
    # Dump BM25 params to a file for later use in retrieval
    bm25.dump("bm25_values.json")
    print("BM25 parameters saved to bm25_values.json")

    pc = Pinecone(api_key=PINECONE_API_KEY)
    setup_pinecone_index(pc, INDEX_NAME)
    index = pc.Index(INDEX_NAME)

    # 4. Generate Embeddings and Upsert
    batch_size = 100
    print("Starting upsert...")
    
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i+batch_size]
        ids = [f"doc_{i+j}" for j in range(len(batch))]
        texts = [chunk.page_content for chunk in batch]
        
        # Dense Embeddings
        dense_vectors = embeddings.embed_documents(texts)
        
        # Sparse Embeddings
        sparse_vectors = bm25.encode_documents(texts)
        
        vectors_to_upsert = []
        for j, chunk in enumerate(batch):
            metadata = {
                "text": chunk.page_content,
                "source": chunk.metadata.get("source", ""),
                "page": chunk.metadata.get("page", 0)
            }
            
            vectors_to_upsert.append({
                "id": ids[j],
                "values": dense_vectors[j],
                "sparse_values": sparse_vectors[j],
                "metadata": metadata
            })
            
        index.upsert(vectors=vectors_to_upsert)
        print(f"Upserted batch {i//batch_size + 1}/{(len(chunks)-1)//batch_size + 1}")

    print("Ingestion complete.")

if __name__ == "__main__":
    ingest_data()
