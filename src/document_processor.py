"""
Document processor for PDF ingestion and text extraction
"""
import os
import logging
from typing import List, Dict, Any
import PyPDF2
import fitz  # PyMuPDF
from pathlib import Path

from config import Config

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentProcessor:
    """Handle PDF document processing and text extraction"""
    
    def __init__(self):
        self.documents_path = Path(Config.DOCUMENTS_PATH)
        self.supported_extensions = Config.SUPPORTED_EXTENSIONS
        self.max_file_size = Config.MAX_FILE_SIZE
        
        # Create documents directory if it doesn't exist
        self.documents_path.mkdir(exist_ok=True)
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """
        Extract text from PDF using PyMuPDF (more robust than PyPDF2)
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Extracted text content
        """
        try:
            text = ""
            
            # Try with PyMuPDF first (more reliable)
            try:
                doc = fitz.open(pdf_path)
                for page_num in range(len(doc)):
                    page = doc.load_page(page_num)
                    text += page.get_text()
                doc.close()
                logger.info(f"Successfully extracted text from {pdf_path} using PyMuPDF")
                
            except Exception as e:
                logger.warning(f"PyMuPDF failed for {pdf_path}: {e}. Trying PyPDF2...")
                
                # Fallback to PyPDF2
                with open(pdf_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    for page in pdf_reader.pages:
                        text += page.extract_text()
                logger.info(f"Successfully extracted text from {pdf_path} using PyPDF2")
            
            # Clean the text
            text = self._clean_text(text)
            
            if not text.strip():
                logger.warning(f"No text extracted from {pdf_path}")
                return ""
                
            return text
            
        except Exception as e:
            logger.error(f"Error extracting text from {pdf_path}: {e}")
            return ""
    
    def _clean_text(self, text: str) -> str:
        """
        Clean extracted text by removing extra whitespace and formatting issues
        
        Args:
            text: Raw extracted text
            
        Returns:
            Cleaned text
        """
        # Remove extra whitespace and normalize line breaks
        text = ' '.join(text.split())
        
        # Remove common PDF artifacts
        text = text.replace('\uf0b7', '•')  # Replace bullet points
        text = text.replace('\u2022', '•')  # Replace bullet points
        text = text.replace('\n\n', '\n')  # Reduce multiple line breaks
        
        return text.strip()
    
    def chunk_text(self, text: str, chunk_size: int = None, overlap: int = None) -> List[str]:
        """
        Split text into chunks for vector storage
        
        Args:
            text: Text to chunk
            chunk_size: Size of each chunk (uses config default if None)
            overlap: Overlap between chunks (uses config default if None)
            
        Returns:
            List of text chunks
        """
        chunk_size = chunk_size or Config.CHUNK_SIZE
        overlap = overlap or Config.CHUNK_OVERLAP
        
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            
            # Try to break at sentence boundary
            if end < len(text):
                # Look for sentence endings within the last 100 characters
                sentence_end = text.rfind('.', end - 100, end)
                if sentence_end != -1:
                    end = sentence_end + 1
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            # Calculate next start position with overlap
            start = end - overlap
            
            # Avoid infinite loop
            if start >= len(text):
                break
        
        logger.info(f"Split text into {len(chunks)} chunks")
        return chunks
    
    def process_documents(self, document_paths: List[str] = None) -> Dict[str, Any]:
        """
        Process all PDF documents in the documents directory
        
        Args:
            document_paths: Optional list of specific document paths to process
            
        Returns:
            Dictionary containing processed documents data
        """
        if document_paths is None:
            # Get all PDF files from documents directory
            document_paths = [
                str(f) for f in self.documents_path.glob("*.pdf")
                if f.is_file()
            ]
        
        if not document_paths:
            logger.warning("No PDF documents found to process")
            return {"documents": [], "chunks": [], "metadata": []}
        
        all_chunks = []
        all_metadata = []
        documents_processed = []
        
        for doc_path in document_paths:
            logger.info(f"Processing document: {doc_path}")
            
            # Validate file
            if not self._validate_file(doc_path):
                continue
            
            # Extract text
            text = self.extract_text_from_pdf(doc_path)
            if not text:
                logger.warning(f"Skipping {doc_path} - no text extracted")
                continue
            
            # Chunk text
            chunks = self.chunk_text(text)
            
            # Create metadata for each chunk
            doc_name = Path(doc_path).name
            for i, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                all_metadata.append({
                    "source": doc_name,
                    "chunk_id": i,
                    "total_chunks": len(chunks),
                    "file_path": doc_path
                })
            
            documents_processed.append({
                "path": doc_path,
                "name": doc_name,
                "chunks_count": len(chunks),
                "text_length": len(text)
            })
            
            logger.info(f"Processed {doc_name}: {len(chunks)} chunks, {len(text)} characters")
        
        result = {
            "documents": documents_processed,
            "chunks": all_chunks,
            "metadata": all_metadata
        }
        
        logger.info(f"Total documents processed: {len(documents_processed)}")
        logger.info(f"Total chunks created: {len(all_chunks)}")
        
        return result
    
    def _validate_file(self, file_path: str) -> bool:
        """
        Validate if file is suitable for processing
        
        Args:
            file_path: Path to the file
            
        Returns:
            True if file is valid, False otherwise
        """
        path = Path(file_path)
        
        # Check if file exists
        if not path.exists():
            logger.error(f"File does not exist: {file_path}")
            return False
        
        # Check extension
        if path.suffix.lower() not in self.supported_extensions:
            logger.error(f"Unsupported file extension: {path.suffix}")
            return False
        
        # Check file size
        if path.stat().st_size > self.max_file_size:
            logger.error(f"File too large: {path.stat().st_size} bytes")
            return False
        
        return True

