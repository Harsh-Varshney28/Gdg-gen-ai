"""
Vector store operations using ChromaDB for document embeddings and similarity search
"""
import logging
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import uuid

from config import Config

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VectorStore:
    """Handle vector database operations using ChromaDB"""
    
    def __init__(self):
        self.config = Config()
        self.db_path = Config.CHROMA_DB_PATH
        self.collection_name = Config.COLLECTION_NAME
        self.embedding_model_name = Config.EMBEDDING_MODEL
        
        # Initialize ChromaDB client
        self.client = None
        self.collection = None
        self.embedding_model = None
        
        self._initialize_db()
        self._load_embedding_model()
    
    def _initialize_db(self):
        """Initialize ChromaDB client and collection"""
        try:
            # Create ChromaDB client
            self.client = chromadb.PersistentClient(path=self.db_path)
            
            # Get or create collection
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}  # Use cosine similarity
            )
            
            logger.info(f"ChromaDB initialized successfully at {self.db_path}")
            logger.info(f"Collection '{self.collection_name}' ready")
            
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
            raise
    
    def _load_embedding_model(self):
        """Load the sentence transformer model for embeddings"""
        try:
            self.embedding_model = SentenceTransformer(self.embedding_model_name)
            logger.info(f"Embedding model '{self.embedding_model_name}' loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            List of embedding vectors
        """
        try:
            embeddings = self.embedding_model.encode(texts, convert_to_tensor=False)
            return embeddings.tolist()
            
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}")
            raise
    
    def add_documents(self, chunks: List[str], metadata: List[Dict[str, Any]]) -> bool:
        """
        Add document chunks to the vector store
        
        Args:
            chunks: List of text chunks
            metadata: List of metadata dictionaries for each chunk
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if len(chunks) != len(metadata):
                raise ValueError("Number of chunks must match number of metadata entries")
            
            if not chunks:
                logger.warning("No chunks provided to add")
                return True
            
            # Generate embeddings for all chunks
            logger.info(f"Generating embeddings for {len(chunks)} chunks...")
            embeddings = self.generate_embeddings(chunks)
            
            # Generate unique IDs for each chunk
            ids = [str(uuid.uuid4()) for _ in chunks]
            
            # Add to collection
            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=chunks,
                metadatas=metadata
            )
            
            logger.info(f"Successfully added {len(chunks)} chunks to vector store")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add documents to vector store: {e}")
            return False
    
    def search_similar(self, query: str, n_results: int = None) -> Dict[str, Any]:
        """
        Search for similar documents using query
        
        Args:
            query: Search query string
            n_results: Number of results to return (uses config default if None)
            
        Returns:
            Dictionary containing search results
        """
        try:
            n_results = n_results or Config.MAX_RETRIEVED_CHUNKS
            
            # Generate embedding for query
            query_embedding = self.generate_embeddings([query])[0]
            
            # Search in collection
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results
            )
            
            # Format results
            formatted_results = {
                "documents": results.get("documents", [[]])[0],
                "metadata": results.get("metadatas", [[]])[0],
                "distances": results.get("distances", [[]])[0],
                "ids": results.get("ids", [[]])[0]
            }
            
            logger.info(f"Found {len(formatted_results['documents'])} similar documents")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Failed to search similar documents: {e}")
            return {"documents": [], "metadata": [], "distances": [], "ids": []}
    
    def get_collection_info(self) -> Dict[str, Any]:
        """
        Get information about the current collection
        
        Returns:
            Dictionary with collection statistics
        """
        try:
            count = self.collection.count()
            
            return {
                "collection_name": self.collection_name,
                "document_count": count,
                "embedding_model": self.embedding_model_name,
                "db_path": self.db_path
            }
            
        except Exception as e:
            logger.error(f"Failed to get collection info: {e}")
            return {}
    
    def clear_collection(self) -> bool:
        """
        Clear all documents from the collection
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Delete the collection and recreate it
            self.client.delete_collection(name=self.collection_name)
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            
            logger.info(f"Collection '{self.collection_name}' cleared successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to clear collection: {e}")
            return False
    
    def delete_documents_by_source(self, source_name: str) -> bool:
        """
        Delete all documents from a specific source
        
        Args:
            source_name: Name of the source document to delete
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Query for documents from this source
            results = self.collection.get(
                where={"source": source_name}
            )
            
            if results["ids"]:
                # Delete documents
                self.collection.delete(ids=results["ids"])
                logger.info(f"Deleted {len(results['ids'])} chunks from source '{source_name}'")
            else:
                logger.info(f"No documents found for source '{source_name}'")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete documents from source '{source_name}': {e}")
            return False
    
    def update_document_metadata(self, document_id: str, new_metadata: Dict[str, Any]) -> bool:
        """
        Update metadata for a specific document
        
        Args:
            document_id: ID of the document to update
            new_metadata: New metadata dictionary
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self.collection.update(
                ids=[document_id],
                metadatas=[new_metadata]
            )
            
            logger.info(f"Updated metadata for document {document_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update metadata for document {document_id}: {e}")
            return False
    
    def get_sources(self) -> List[str]:
        """
        Get list of all unique sources in the collection
        
        Returns:
            List of source names
        """
        try:
            # Get all documents
            results = self.collection.get()
            
            # Extract unique sources
            sources = set()
            for metadata in results.get("metadatas", []):
                if "source" in metadata:
                    sources.add(metadata["source"])
            
            return list(sources)
            
        except Exception as e:
            logger.error(f"Failed to get sources: {e}")
            return []
