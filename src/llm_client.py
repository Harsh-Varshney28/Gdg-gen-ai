"""
LLM client for Groq API and HuggingFace integration
"""
import logging
from typing import List, Dict, Any, Optional
import os
from groq import Groq
import requests

from config import Config

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LLMClient:
    """Handle LLM operations using Groq API and HuggingFace"""
    
    def __init__(self):
        self.config = Config()
        self.groq_api_key = Config.GROQ_API_KEY
        self.hf_api_key = Config.HUGGINGFACE_API_KEY
        self.model_name = Config.LLM_MODEL
        self.max_tokens = Config.MAX_TOKENS
        self.temperature = Config.TEMPERATURE
        
        # Initialize Groq client
        self.groq_client = None
        self._initialize_groq()
    
    def _initialize_groq(self):
        """Initialize Groq client"""
        try:
            if self.groq_api_key:
                self.groq_client = Groq(api_key=self.groq_api_key)
                logger.info("Groq client initialized successfully")
            else:
                logger.warning("Groq API key not provided. Groq functionality disabled.")
                
        except Exception as e:
            logger.error(f"Failed to initialize Groq client: {e}")
    
    def generate_response_groq(self, prompt: str, context: str = "") -> str:
        """
        Generate response using Groq API
        
        Args:
            prompt: User's question or prompt
            context: Retrieved context from documents
            
        Returns:
            Generated response
        """
        try:
            if not self.groq_client:
                return "Groq client not available. Please check your API key."
            
            # Construct system message
            system_message = self._build_system_message()
            
            # Construct user message with context
            user_message = self._build_user_message(prompt, context)
            
            # Make API call
            response = self.groq_client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message}
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                top_p=0.9,
                stream=False
            )
            
            generated_text = response.choices[0].message.content
            logger.info("Response generated successfully using Groq")
            
            return generated_text
            
        except Exception as e:
            logger.error(f"Failed to generate response with Groq: {e}")
            return f"Error generating response: {str(e)}"
    
    def generate_response_huggingface(self, prompt: str, context: str = "", model_id: str = "microsoft/DialoGPT-medium") -> str:
        """
        Generate response using HuggingFace Inference API
        
        Args:
            prompt: User's question or prompt
            context: Retrieved context from documents
            model_id: HuggingFace model ID
            
        Returns:
            Generated response
        """
        try:
            if not self.hf_api_key:
                return "HuggingFace API key not available."
            
            # Construct the input text
            input_text = self._build_user_message(prompt, context)
            
            # HuggingFace API endpoint
            api_url = f"https://api-inference.huggingface.co/models/{model_id}"
            headers = {"Authorization": f"Bearer {self.hf_api_key}"}
            
            # Prepare payload
            payload = {
                "inputs": input_text,
                "parameters": {
                    "max_new_tokens": self.max_tokens,
                    "temperature": self.temperature,
                    "do_sample": True,
                    "top_p": 0.9
                }
            }
            
            # Make API call
            response = requests.post(api_url, headers=headers, json=payload)
            response.raise_for_status()
            
            result = response.json()
            
            if isinstance(result, list) and len(result) > 0:
                generated_text = result[0].get("generated_text", "")
                # Remove the input prompt from the generated text
                if generated_text.startswith(input_text):
                    generated_text = generated_text[len(input_text):].strip()
                
                logger.info("Response generated successfully using HuggingFace")
                return generated_text
            else:
                return "No response generated"
            
        except Exception as e:
            logger.error(f"Failed to generate response with HuggingFace: {e}")
            return f"Error generating response: {str(e)}"
    
    def generate_response(self, prompt: str, context: str = "", use_groq: bool = True) -> str:
        """
        Generate response using the preferred LLM service
        
        Args:
            prompt: User's question or prompt
            context: Retrieved context from documents
            use_groq: Whether to use Groq (True) or HuggingFace (False)
            
        Returns:
            Generated response
        """
        if use_groq and self.groq_client:
            return self.generate_response_groq(prompt, context)
        else:
            return self.generate_response_huggingface(prompt, context)
    
    def _build_system_message(self) -> str:
        """Build system message for the LLM"""
        return """You are a helpful AI assistant that answers questions based on provided context from documents. 

Instructions:
1. Use only the information provided in the context to answer questions
2. If the context doesn't contain enough information to answer the question, say so clearly
3. Be concise but comprehensive in your responses
4. Cite specific parts of the context when relevant
5. If asked about something not in the context, politely explain that you can only answer based on the provided documents

Always be accurate and helpful while staying within the bounds of the provided information."""
    
    def _build_user_message(self, prompt: str, context: str) -> str:
        """
        Build user message with context and prompt
        
        Args:
            prompt: User's question
            context: Retrieved context
            
        Returns:
            Formatted user message
        """
        if context:
            return f"""Context from documents:
{context}

Question: {prompt}

Please answer the question based on the context provided above."""
        else:
            return f"""Question: {prompt}

Note: No relevant context was found in the documents. Please let me know if you'd like me to help with something else."""
    
    def summarize_text(self, text: str, max_length: int = 200) -> str:
        """
        Generate a summary of the provided text
        
        Args:
            text: Text to summarize
            max_length: Maximum length of summary
            
        Returns:
            Summary of the text
        """
        try:
            prompt = f"Please provide a concise summary of the following text in no more than {max_length} words:\n\n{text}"
            
            return self.generate_response(prompt, use_groq=True)
            
        except Exception as e:
            logger.error(f"Failed to generate summary: {e}")
            return "Failed to generate summary"
    
    def extract_keywords(self, text: str, num_keywords: int = 10) -> List[str]:
        """
        Extract key terms from the text
        
        Args:
            text: Text to extract keywords from
            num_keywords: Number of keywords to extract
            
        Returns:
            List of extracted keywords
        """
        try:
            prompt = f"Extract the {num_keywords} most important keywords or key phrases from the following text. Return only the keywords, separated by commas:\n\n{text}"
            
            response = self.generate_response(prompt, use_groq=True)
            
            # Parse keywords from response
            keywords = [kw.strip() for kw in response.split(',')]
            return keywords[:num_keywords]
            
        except Exception as e:
            logger.error(f"Failed to extract keywords: {e}")
            return []
    
    def is_available(self) -> Dict[str, bool]:
        """
        Check availability of different LLM services
        
        Returns:
            Dictionary showing availability of each service
        """
        return {
            "groq": bool(self.groq_client and self.groq_api_key),
            "huggingface": bool(self.hf_api_key)
        }
