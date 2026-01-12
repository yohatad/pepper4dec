"""
Simplified RAG ROS 2 Node for Pepper Robot Lab Assistant

Clean implementation without intent classification.
The application controls when to call the RAG service.

Environment Variables:
    LLM_BASE_URL: LLM API endpoint
    LLM_API_KEY: API key for LLM service
    LLM_MODEL: Model name
    CHROMA_PATH: Path for ChromaDB storage
    EMBEDDING_MODEL: Sentence transformer model
    RAG_VERBOSE: Set to "true" for debug logging
"""

import os
import rclpy
from pathlib import Path
from typing import List, Dict
from rclpy.node import Node
from ament_index_python.packages import get_package_share_directory
from cssr_interface.srv import LanguageModelCreateCollection, LanguageModelPrompt
from simple_rag import (
    get_config,
    get_collection,
    setup_collection,
    handle_query,
)

PACKAGE_PATH = Path(get_package_share_directory('language_model'))


class SimpleRAGNode(Node):
    """
    Simple RAG Node for Pepper robot.
    
    Services:
        - rag_prompt: Answer questions using RAG
        - create_collection: Initialize knowledge base
    
    Configuration via environment variables:
        export LLM_API_KEY="your-api-key"
        export LLM_BASE_URL="https://api.groq.com/openai/v1"
        export LLM_MODEL="llama-3.1-8b-instant"
    """
    
    def __init__(self):
        super().__init__('rag_node')
        
        # Parameters (can override env vars)
        self.declare_parameter('collection_name', 'upanzi_knowledge')
        self.declare_parameter('verbose', False)
        
        # Log configuration (hide API key)
        config = get_config()
        self.get_logger().info(f"LLM URL: {config.llm_base_url}")
        self.get_logger().info(f"LLM Model: {config.llm_model}")
        self.get_logger().info(f"LLM API Key: {'***' + config.llm_api_key[-4:] if len(config.llm_api_key) > 4 else '(default)'}")
        self.get_logger().info(f"ChromaDB Path: {config.chroma_path}")
        self.get_logger().info(f"Embedding Model: {config.embedding_model}")
        
        # State
        self._collection = None
        self._conversation_history: List[Dict] = []
        
        # Try to load existing collection
        collection_name = self.get_parameter('collection_name').get_parameter_value().string_value
        self._load_collection(collection_name)
        
        # Services - use the correct imported service types
        self.create_service(LanguageModelPrompt, 'rag_prompt', self._prompt_callback)
        self.create_service(LanguageModelCreateCollection, 'create_collection', self._create_collection_callback)
        
        status = f"Collection: {collection_name}" if self._collection else "Collection: Not loaded"
        self.get_logger().info(f"RAG Node ready. {status}")
    
    def _load_collection(self, name: str):
        """Load existing collection"""
        try:
            self._collection = get_collection(name)
            if self._collection:
                count = self._collection.count()
                self.get_logger().info(f"Loaded collection '{name}' with {count} documents")
        except Exception as e:
            self.get_logger().warn(f"Could not load collection: {e}")
    
    @property
    def verbose(self) -> bool:
        env_verbose = os.getenv("RAG_VERBOSE", "false").lower() == "true"
        param_verbose = self.get_parameter('verbose').get_parameter_value().bool_value
        return env_verbose or param_verbose
    
    def _prompt_callback(self, request, response):
        """Handle RAG query"""
        
        # Check collection
        if self._collection is None:
            response.response = "Knowledge base not initialized. Please set up the collection first."
            return response
        
        query = request.prompt.strip() if request.prompt else ""
        if not query:
            response.response = "I didn't catch that. Could you repeat?"
            return response
        
        if self.verbose:
            self.get_logger().info(f"Query: {query}")
        
        try:
            # Handle query
            result = handle_query(
                collection=self._collection,
                query=query,
                conversation_history=self._conversation_history
            )
            
            # Update history
            self._conversation_history.append({
                'query': query,
                'response': result['response']
            })
            
            # Keep history manageable
            if len(self._conversation_history) > 5:
                self._conversation_history = self._conversation_history[-5:]
            
            if self.verbose:
                self.get_logger().info(f"Response: {result['response'][:100]}...")
                self.get_logger().info(f"Sources: {result['sources']}")
            
            response.response = result['response']
            
        except Exception as e:
            self.get_logger().error(f"Query error: {e}")
            response.response = "Something went wrong. Please try again."
        
        return response
    
    def _create_collection_callback(self, request, response):
        """Create and populate collection"""
        
        name = request.name.strip() if request.name else ""
        data_path = request.datafile_path.strip() if request.datafile_path else ""
        description = request.description.strip() if request.description else ""
        
        if not name:
            response.success = 0
            response.message = "Collection name required"
            return response
        
        if not data_path:
            response.success = 0
            response.message = "Data file path required"
            return response
        
        if not Path(data_path).exists():
            response.success = 0
            response.message = f"File not found: {data_path}"
            return response
        
        if self.verbose:
            self.get_logger().info(f"Creating collection: {name} from {data_path}")
        
        try:
            collection = setup_collection(name, data_path, description)
            
            if collection:
                self._collection = collection
                self._conversation_history = []
                
                count = collection.count()
                response.success = 1
                response.message = f"Collection '{name}' ready with {count} documents"
                self.get_logger().info(response.message)
            else:
                response.success = 0
                response.message = "Failed to create collection"
                
        except Exception as e:
            response.success = 0
            response.message = f"Error: {e}"
            self.get_logger().error(response.message)
        
        return response
    
    def clear_history(self):
        """Clear conversation history"""
        self._conversation_history = []


def main(args=None):
    rclpy.init(args=args)
    
    node = None
    try:
        node = SimpleRAGNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node:
            node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()