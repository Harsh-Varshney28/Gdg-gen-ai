"""
Demo script for RAG QA Assistant
This script demonstrates the core functionality of the system
"""
import sys
import os
sys.path.append('src')

from rag_pipeline import RAGPipeline
from config import Config
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Main demo function"""
    print("=" * 60)
    print("🤖 RAG Question-Answering Assistant Demo")
    print("=" * 60)
    
    try:
        # Initialize the RAG pipeline
        print("\n📊 Initializing RAG Pipeline...")
        rag = RAGPipeline()
        
        # Get system status
        print("\n📈 System Status:")
        status = rag.get_system_status()
        print(f"   • Status: {status['status']}")
        print(f"   • Vector Store: {status['vector_store']['document_count']} documents")
        print(f"   • Embedding Model: {status['config']['embedding_model']}")
        print(f"   • LLM Model: {status['config']['llm_model']}")
        
        # Check for documents
        print(f"\n📄 Checking for documents in ./documents/ directory...")
        
        # If no documents, show how to add them
        if status['vector_store']['document_count'] == 0:
            print("   ⚠️  No documents found in vector store.")
            print("   📝 To test the system:")
            print("      1. Add PDF files to the ./documents/ directory")
            print("      2. Run: python demo.py")
            print("      3. The system will automatically process them")
            
            # Try to ingest any documents that might be in the folder
            print("\n🔄 Attempting to ingest documents...")
            result = rag.ingest_documents()
            
            if result['success']:
                print(f"   ✅ Successfully processed {result['documents_processed']} documents")
                print(f"   📊 Created {result['chunks_added']} text chunks")
            else:
                print(f"   ❌ {result['message']}")
                print("\n💡 Demo Tips:")
                print("   • Place sample PDF files in the ./documents/ folder")
                print("   • Ensure your .env file has GROQ_API_KEY set")
                print("   • Run the demo again after adding documents")
                return
        
        # Demonstrate querying
        print(f"\n🤔 Testing Question-Answering...")
        
        # Sample questions to test
        sample_questions = [
            "What is the main topic of the documents?",
            "Can you summarize the key points?",
            "What are the important details mentioned?"
        ]
        
        for i, question in enumerate(sample_questions, 1):
            print(f"\n   Question {i}: {question}")
            
            try:
                response = rag.query(question)
                
                if response['success']:
                    print(f"   ✅ Answer: {response['answer'][:200]}...")
                    print(f"   📊 Retrieved {response['retrieved_chunks']} relevant chunks")
                    print(f"   📚 Sources: {response['sources']}")
                    print(f"   ⏱️  Processing time: {response['processing_time']}s")
                else:
                    print(f"   ❌ Error: {response['answer']}")
                    
            except Exception as e:
                print(f"   ❌ Query failed: {str(e)}")
                
            if i < len(sample_questions):
                print("   " + "-" * 50)
        
        # Show available sources
        sources = rag.vector_store.get_sources()
        if sources:
            print(f"\n📚 Available Documents:")
            for source in sources:
                print(f"   • {source}")
        
        print(f"\n🎉 Demo completed successfully!")
        print(f"📝 Your RAG QA Assistant is working and ready for questions!")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {str(e)}")
        print(f"\n🔧 Troubleshooting:")
        print(f"   • Make sure all dependencies are installed: pip install -r requirements.txt")
        print(f"   • Check that your .env file has the required API keys")
        print(f"   • Ensure PDF documents are in the ./documents/ directory")

def interactive_mode():
    """Interactive questioning mode"""
    print("\n🎯 Interactive Mode - Ask your own questions!")
    print("Type 'quit' to exit")
    
    try:
        rag = RAGPipeline()
        
        while True:
            print("\n" + "="*50)
            question = input("❓ Your question: ").strip()
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
                
            if not question:
                print("Please enter a question.")
                continue
                
            print("\n🔍 Processing your question...")
            
            try:
                response = rag.query(question)
                
                if response['success']:
                    print(f"\n💡 Answer:")
                    print(f"   {response['answer']}")
                    print(f"\n📊 Details:")
                    print(f"   • Retrieved chunks: {response['retrieved_chunks']}")
                    print(f"   • Sources: {', '.join(response['sources'])}")
                    print(f"   • Processing time: {response['processing_time']}s")
                else:
                    print(f"\n❌ Error: {response['answer']}")
                    
            except Exception as e:
                print(f"\n❌ Query failed: {str(e)}")
                
    except Exception as e:
        print(f"❌ Interactive mode failed: {str(e)}")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
        interactive_mode()
    else:
        main()
        
        # Ask if user wants interactive mode
        response = input("\n🎯 Would you like to try interactive mode? (y/n): ").strip().lower()
        if response in ['y', 'yes']:
            interactive_mode()
