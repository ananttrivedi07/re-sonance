import chromadb
from sentence_transformers import SentenceTransformer

class RAGMemoryBackbone:
    def __init__(self, db_path="./nostalgic_memory_db", collection_name="patient_history"):
        print("Initializing local Vector Database Backbone...")
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.chroma_client = chromadb.PersistentClient(path=db_path)
        self.collection = self.chroma_client.get_or_create_collection(name=collection_name)

    def add_memory(self, document_text: str, doc_id: str):
        """Saves historical entries, summaries, or caretaker logs to disk index."""
        vector = self.embedding_model.encode(document_text).tolist()
        self.collection.add(
            embeddings=[vector],
            documents=[document_text],
            ids=[doc_id]
        )

    def query_memory(self, query_text: str, max_results=2) -> str:
        """Queries database using user text embeddings to retrieve contextual summaries."""
        if self.collection.count() == 0:
            return ""
            
        query_vector = self.embedding_model.encode(query_text).tolist()
        results = self.collection.query(query_embeddings=[query_vector], n_results=max_results)
        
        retrieved_docs = results['documents'][0] if results['documents'] else []
        return "\n".join(retrieved_docs)