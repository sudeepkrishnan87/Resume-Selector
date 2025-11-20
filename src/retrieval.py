import os
from typing import List, Dict, Any

from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from pinecone import Pinecone
from pinecone_text.sparse import BM25Encoder
from sentence_transformers import CrossEncoder

# Load environment variables
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("INDEX_NAME", "resume-index")

class ResumeRetriever:
    def __init__(self):
        self.pc = Pinecone(api_key=PINECONE_API_KEY)
        self.index = self.pc.Index(INDEX_NAME)
        
        # Load Embeddings
        print("Loading embedding model...")
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        # Load BM25
        print("Loading BM25 encoder...")
        self.bm25 = BM25Encoder()
        if os.path.exists("bm25_values.json"):
            self.bm25.load("bm25_values.json")
        else:
            print("WARNING: bm25_values.json not found. Sparse retrieval may be inaccurate.")
            self.bm25.fit(["dummy text"]) # Fallback

        # Load Reranker
        print("Loading reranker...")
        self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

        # Load LLM for Query Enrichment
        self.llm = self._setup_llm()

    def _setup_llm(self):
        """Sets up the LLM, preferring Google, falling back to OpenAI."""
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            if os.getenv("GOOGLE_API_KEY"):
                return ChatGoogleGenerativeAI(model="gemini-1.0-pro", convert_system_message_to_human=True)
        except ImportError:
            pass
        
        try:
            from langchain_openai import ChatOpenAI
            if os.getenv("OPENAI_API_KEY"):
                return ChatOpenAI(model="gpt-3.5-turbo")
        except ImportError:
            pass
            
        print("WARNING: No LLM available for Query Enrichment. HyDE and Expansion will be disabled.")
        return None

    def enrich_query(self, query: str) -> List[str]:
        """Generates multiple variations of the query (Query Expansion)."""
        if not self.llm:
            return [query]
            
        prompt = f"Provide 3 alternative search queries for the following user query to improve retrieval from a resume database. Return only the queries, one per line.\nUser Query: {query}"
        try:
            response = self.llm.invoke(prompt)
            queries = response.content.split('\n')
            queries = [q.strip() for q in queries if q.strip()]
            return [query] + queries
        except Exception as e:
            print(f"Query expansion failed: {e}")
            return [query]

    def generate_hyde(self, query: str) -> str:
        """Generates a hypothetical resume snippet (HyDE)."""
        if not self.llm:
            return query
            
        prompt = f"Write a hypothetical resume snippet that would be the perfect match for this query: '{query}'. Include skills, experience, and keywords."
        try:
            response = self.llm.invoke(prompt)
            return response.content
        except Exception as e:
            print(f"HyDE generation failed: {e}")
            return query

    def hybrid_search(self, query: str, top_k: int = 10, alpha: float = 0.5) -> List[Dict[str, Any]]:
        """
        Performs hybrid search (Dense + Sparse).
        alpha: 0.0 = pure sparse, 1.0 = pure dense
        """
        # Generate vectors
        dense_vec = self.embeddings.embed_query(query)
        sparse_vec = self.bm25.encode_queries(query)
        
        # Scale vectors based on alpha
        if alpha < 0 or alpha > 1:
            raise ValueError("Alpha must be between 0 and 1")

        # Scale dense vector
        dense_vec = [v * alpha for v in dense_vec]
        
        # Scale sparse vector
        sparse_vec['values'] = [v * (1 - alpha) for v in sparse_vec['values']]

        results = self.index.query(
            vector=dense_vec,
            sparse_vector=sparse_vec,
            top_k=top_k,
            include_metadata=True
        )
        
        return results['matches']

    def rerank(self, query: str, matches: List[Dict[str, Any]], top_n: int = 5) -> List[Dict[str, Any]]:
        """Reranks the initial retrieval results using a CrossEncoder."""
        if not matches:
            return []
            
        # Prepare pairs for reranking
        pairs = []
        for match in matches:
            text = match['metadata'].get('text', '')
            pairs.append([query, text])
            
        # Get scores
        scores = self.reranker.predict(pairs)
        
        # Attach scores to matches
        for i, match in enumerate(matches):
            match['rerank_score'] = float(scores[i])
            
        # Sort by rerank score
        sorted_matches = sorted(matches, key=lambda x: x['rerank_score'], reverse=True)
        
        return sorted_matches[:top_n]

    def retrieve(self, query: str, top_k: int = 10, top_n: int = 5, use_enrichment: bool = True) -> List[Dict[str, Any]]:
        print(f"Searching for: '{query}'")
        
        search_queries = [query]
        if use_enrichment and self.llm:
            print("Enriching query...")
            # 1. Query Expansion
            expanded_queries = self.enrich_query(query)
            print(f"Expanded queries: {expanded_queries}")
            search_queries.extend(expanded_queries)
            
            # 2. HyDE (Optional - can be used as another query or to augment)
            # For simplicity, we'll just stick to expansion for now to avoid too many calls
            # hyde_doc = self.generate_hyde(query)
            # search_queries.append(hyde_doc)

        # De-duplicate
        search_queries = list(set(search_queries))
        
        all_matches = []
        for q in search_queries:
            matches = self.hybrid_search(q, top_k=top_k)
            all_matches.extend(matches)
            
        # De-duplicate matches by ID
        unique_matches = {m['id']: m for m in all_matches}.values()
        initial_matches = list(unique_matches)
        
        print(f"Found {len(initial_matches)} unique matches from Pinecone across {len(search_queries)} queries.")
        
        # 2. Reranking
        reranked_matches = self.rerank(query, initial_matches, top_n=top_n)
        print(f"Reranked top {len(reranked_matches)} results.")
        
        return reranked_matches

if __name__ == "__main__":
    retriever = ResumeRetriever()
    
    # Test Query
    query = "Senior Accountant with audit experience"
    results = retriever.retrieve(query)
    
    print("\n--- Top Results ---")
    for i, res in enumerate(results):
        print(f"{i+1}. Score: {res['rerank_score']:.4f} (Pinecone: {res['score']:.4f})")
        print(f"   Source: {res['metadata'].get('source')}")
        print(f"   Snippet: {res['metadata'].get('text')[:150]}...\n")
