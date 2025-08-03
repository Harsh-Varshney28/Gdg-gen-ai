"""
Main RAG (Retrieval-Augmented Generation) pipeline that orchestrates all components
"""
import logging
from typing import List, Dict, Any, Optional
import time
from pathlib import Path

from config import Config
from document_processor import DocumentProcessor
from vector_store import VectorStore
from llm_client import LLMClient

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGPipeline:
    """Main RAG pipeline that orchestrates document processing, retrieval, and generation"""
    
    def __init__(self):
        self.config = Config()
        
        # Initialize components
        logger.info("Initializing RAG Pipeline components...")
        
        try:
            self.document_processor = DocumentProcessor()
            logger.info("✓ Document processor initialized")
            
            self.vector_store = VectorStore()
            logger.info("✓ Vector store initialized")
            
            self.llm_client = LLMClient()
            logger.info("✓ LLM client initialized")
            
            logger.info("RAG Pipeline initialization complete!")
            
        except Exception as e:
            logger.error(f"Failed to initialize RAG Pipeline: {e}")
            raise
    
    def ingest_documents(self, document_paths: List[str] = None, clear_existing: bool = False) -> Dict[str, Any]:
        """
        Ingest PDF documents into the vector store
        
        Args:
            document_paths: Optional list of specific document paths to process
            clear_existing: Whether to clear existing documents before ingesting new ones
            
        Returns:
            Dictionary with ingestion results
        """
        try:
            logger.info("Starting document ingestion...")
            start_time = time.time()
            
            # Clear existing documents if requested
            if clear_existing:
                logger.info("Clearing existing documents...")
                self.vector_store.clear_collection()
            
            # Process documents
            processed_data = self.document_processor.process_documents(document_paths)
            
            if not processed_data["chunks"]:
                logger.warning("No document chunks to ingest")
                return {
                    "success": False,
                    "message": "No documents were processed",
                    "documents_processed": 0,
                    "chunks_added": 0
                }
            
            # Add to vector store
            success = self.vector_store.add_documents(
                chunks=processed_data["chunks"],
                metadata=processed_data["metadata"]
            )
            
            processing_time = time.time() - start_time
            
            if success:
                result = {
                    "success": True,
                    "message": "Documents ingested successfully",
                    "documents_processed": len(processed_data["documents"]),
                    "chunks_added": len(processed_data["chunks"]),
                    "processing_time": round(processing_time, 2),
                    "documents": processed_data["documents"]
                }
                
                logger.info(f"Document ingestion completed in {processing_time:.2f} seconds")
                logger.info(f"Processed {len(processed_data['documents'])} documents, {len(processed_data['chunks'])} chunks")
                
                return result
            else:
                return {
                    "success": False,
                    "message": "Failed to add documents to vector store",
                    "documents_processed": len(processed_data["documents"]),
                    "chunks_added": 0
                }
                
        except Exception as e:
            logger.error(f"Document ingestion failed: {e}")
            return {
                "success": False,
                "message": f"Document ingestion failed: {str(e)}",
                "documents_processed": 0,
                "chunks_added": 0
            }
    
    def query(self, question: str, n_results: int = None, use_groq: bool = True) -> Dict[str, Any]:
        """
        Query the RAG system with a question
        
        Args:
            question: User's question
            n_results: Number of similar chunks to retrieve
            use_groq: Whether to use Groq API (True) or HuggingFace (False)
            
        Returns:
            Dictionary with query results
        """
        try:
            logger.info(f"Processing query: {question[:100]}...")
            start_time = time.time()
            
            # Retrieve similar documents
            search_results = self.vector_store.search_similar(
                query=question,
                n_results=n_results
            )
            
            # Build context from retrieved documents
            context = self._build_context(search_results)
            
            # Generate response using LLM
            response = self.llm_client.generate_response(
                prompt=question,
                context=context,
                use_groq=use_groq
            )
            
            processing_time = time.time() - start_time
            
            result = {
                "success": True,
                "question": question,
                "answer": response,
                "context": context,
                "retrieved_chunks": len(search_results["documents"]),
                "sources": self._extract_sources(search_results),
                "processing_time": round(processing_time, 2),
                "search_results": search_results
            }
            
            logger.info(f"Query processed in {processing_time:.2f} seconds")
            return result
            
        except Exception as e:
            logger.error(f"Query processing failed: {e}")
            return {
                "success": False,
                "question": question,
                "answer": f"Error processing query: {str(e)}",
                "context": "",
                "retrieved_chunks": 0,
                "sources": [],
                "processing_time": 0
            }
    
    def _build_context(self, search_results: Dict[str, Any]) -> str:
        """
        Build context string from search results
        
        Args:
            search_results: Results from vector similarity search
            
        Returns:
            Formatted context string
        """
        if not search_results["documents"]:
            return ""
        
        context_parts = []
        
        for i, (doc, metadata) in enumerate(zip(search_results["documents"], search_results["metadata"])):
            source = metadata.get("source", "Unknown")
            chunk_id = metadata.get("chunk_id", 0)
            
            context_parts.append(f"[Source: {source}, Chunk {chunk_id + 1}]\n{doc}")
        
        return "\n\n".join(context_parts)
    
    def _extract_sources(self, search_results: Dict[str, Any]) -> List[str]:
        """
        Extract unique source names from search results
        
        Args:
            search_results: Results from vector similarity search
            
        Returns:
            List of unique source names
        """
        sources = set()
        for metadata in search_results["metadata"]:
            source = metadata.get("source", "Unknown")
            sources.add(source)
        
        return list(sources)
    
    def get_system_status(self) -> Dict[str, Any]:
        """
        Get current system status and statistics
        
        Returns:
            Dictionary with system status information
        """
        try:
            # Get vector store info
            collection_info = self.vector_store.get_collection_info()
            
            # Get LLM availability
            llm_availability = self.llm_client.is_available()
            
            # Get sources
            sources = self.vector_store.get_sources()
            
            return {
                "status": "operational",
                "vector_store": collection_info,
                "llm_services": llm_availability,
                "available_sources": sources,
                "config": self.config.get_model_info()
            }
            
        except Exception as e:
            logger.error(f"Failed to get system status: {e}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    def add_single_document(self, file_path: str) -> Dict[str, Any]:
        """
        Add a single document to the vector store
        
        Args:
            file_path: Path to the document file
            
        Returns:
            Dictionary with operation result
        """
        try:
            if not Path(file_path).exists():
                return {
                    "success": False,
                    "message": f"File not found: {file_path}"
                }
            
            return self.ingest_documents(document_paths=[file_path])
            
        except Exception as e:
            logger.error(f"Failed to add single document: {e}")
            return {
                "success": False,
                "message": f"Failed to add document: {str(e)}"
            }
    
    def remove_document(self, source_name: str) -> Dict[str, Any]:
        """
        Remove all chunks from a specific document
        
        Args:
            source_name: Name of the source document to remove
            
        Returns:
            Dictionary with operation result
        """
        try:
            success = self.vector_store.delete_documents_by_source(source_name)
            
            if success:
                return {
                    "success": True,
                    "message": f"Document '{source_name}' removed successfully"
                }
            else:
                return {
                    "success": False,
                    "message": f"Failed to remove document '{source_name}'"
                }
                
        except Exception as e:
            logger.error(f"Failed to remove document: {e}")
            return {
                "success": False,
                "message": f"Failed to remove document: {str(e)}"
            }
    
    def clear_all_documents(self) -> Dict[str, Any]:
        """
        Clear all documents from the vector store
        
        Returns:
            Dictionary with operation result
        """
        try:
            success = self.vector_store.clear_collection()
            
            if success:
                return {
                    "success": True,
                    "message": "All documents cleared successfully"
                }
            else:
                return {
                    "success": False,
                    "message": "Failed to clear documents"
                }
                
        except Exception as e:
            logger.error(f"Failed to clear documents: {e}")
            return {
                "success": False,
                "message": f"Failed to clear documents: {str(e)}"
            }
    
    def get_document_summary(self, source_name: str) -> str:
        """
        Generate a summary of a specific document
        
        Args:
            source_name: Name of the source document
            
        Returns:
            Summary of the document
        """
        try:
            # Get all chunks from this source
            results = self.vector_store.collection.get(
                where={"source": source_name}
            )
            
            if not results["documents"]:
                return f"No document found with source name: {source_name}"
            
            # Combine all chunks
            full_text = " ".join(results["documents"])
            
            # Generate summary
            summary = self.llm_client.summarize_text(full_text)
            
            return summary
            
        except Exception as e:
            logger.error(f"Failed to generate document summary: {e}")
            return f"Failed to generate summary: {str(e)}"
