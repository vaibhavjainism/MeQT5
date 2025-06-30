# server.py
import pathway as pw
from pathway.xpacks.llm.vector_store import VectorStoreServer
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pathlib import Path

class RAGServer:
    def __init__(self, docs_directory, host="0.0.0.0", port=8080):
        self.docs_directory = Path(docs_directory)
        self.host = host
        self.port = port
        self.server = None
        
        # Initialize embeddings model
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=200
        )

    def read_documents(self):
        """Read documents from the specified directory"""
        return pw.io.fs.read(
            self.docs_directory,
            format='binary',
            mode='streaming',
            with_metadata=True,
        )

    def setup_server(self):
        """Setup the vector store server"""
        try:
            docs = self.read_documents()
            self.server = VectorStoreServer.from_langchain_components(
                docs,
                embedder=self.embeddings,
                splitter=self.text_splitter
            )
            return True
        except Exception as e:
            print(f"Error setting up server: {str(e)}")
            return False

    def start(self, threaded=True):
        """Start the vector store server"""
        if not self.server and not self.setup_server():
            raise RuntimeError("Failed to setup server")
        
        print(f"Starting RAG API server on http://{self.host}:{self.port}")
        return self.server.run_server(
            host=self.host,
            port=self.port,
            threaded=threaded,
            with_cache=True
        )

def main():
    # Initialize and start the server
    server = RAGServer(
        docs_directory="/workspace/codesearchnet-python-raw-457k/",  # Update this path
        host="0.0.0.0",
        port=8080
    )
    server2 = RAGServer(
        docs_directory="workspace/db_folder/files.txt",  # Update this path
        host="0.0.0.0",
        port=8081
    )
    try:
        server_thread = server.start(threaded=True)
        server_thread.join()
        server_thread_2 = server2.start(threaded=True)
        server_thread_2.join()
    except KeyboardInterrupt:
        print("\nShutting down server...")
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
