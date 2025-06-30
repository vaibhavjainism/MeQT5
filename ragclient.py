# client.py
from pathway.xpacks.llm.vector_store import VectorStoreClient

class RAGClient:
    def __init__(self, host="localhost", port=8080):
        self.host = host
        self.port = port
        self.client = None
        self.setup_client()

    def setup_client(self):
        """Initialize the vector store client"""
        self.client = VectorStoreClient(
            host=self.host,
            port=self.port,
            timeout=30
        )

    def search_documents(self, query, k=3):
        """Search documents using the vector store"""
        try:
            results = self.client.query(
                query=query,
                k=k,
            )
            return results
        except Exception as e:
            print(f"Search error: {str(e)}")
            return None

def main():
    # Initialize the client
    client = RAGClient(
        host="localhost",  # Update if server is on different host
        port=8080
    )
    
    try:
        # Example search
        query = "What is this document about?"
        results = client.search_documents(query, k=3)
        
        if results:
            print(f"\nSearch Results for '{query}':")
            for idx, result in enumerate(results):
                print(f"\n--- Result {idx + 1} ---")
                print(f"Content: {result.get('content', '')[:200]}...")
                print(f"Metadata: {result.get('metadata', {})}")
        else:
            print("No results found")
            
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
