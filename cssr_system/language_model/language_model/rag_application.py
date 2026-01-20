#!/usr/bin/env python3
"""
Simplified RAG ROS 2 Node for Pepper Robot Lab Assistant

Environment Variables:
    LLM_API_KEY: API key for LLM service (MUST be exported)
    LLM_BASE_URL: LLM API endpoint
    LLM_MODEL: Model name
    CHROMA_PATH: Path for ChromaDB storage
    EMBEDDING_MODEL: Sentence transformer model
    RAG_VERBOSE: Set to "true" for debug logging

Configuration:
    All non-API key settings loaded from config/rag_system_configuration.yaml
"""

import rclpy
from pathlib import Path
from typing import List, Dict
from rclpy.node import Node
from ament_index_python.packages import get_package_share_directory
from cssr_interfaces.srv import LanguageModelCreateCollection, LanguageModelPrompt
from .rag_implementation import (
    get_config,
    get_collection,
    setup_collection,
    handle_query,
    apply_config_file,
)

PACKAGE_PATH = Path(get_package_share_directory('language_model'))

# Static config options (not set via config file)
NODE_NAME = "conversation_management"
GET_INTENT_SERVICE = "/conversationManagement/get_intent"
PROMPT_SERVICE = "/conversationManagement/prompt"
CREATE_COLLECTION = "/conversationManagement/create_collection"
HEARTBEAT_MSG_PERIOD = 10  # seconds

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
        
        # Load configuration from YAML file
        config_file = PACKAGE_PATH / 'config' / 'rag_system_configuration.yaml'
        if config_file.exists():
            success, messages = apply_config_file(str(config_file))
            if success:
                self.get_logger().info(f"Configuration loaded from {config_file}")
                for msg in messages:
                    self.get_logger().info(f"Config note: {msg}")
            else:
                self.get_logger().error(f"Failed to load configuration: {messages}")
        else:
            self.get_logger().warn(f"Configuration file not found: {config_file}")
        
        # Log verbose status
        config = get_config()
        if config.verbose:
            self.get_logger().info("Verbose mode is ENABLED - detailed logging will be displayed")
        else:
            self.get_logger().info("Verbose mode is DISABLED - only essential logs will be shown")
        
        # Parameters (can override env vars)
        self.declare_parameter('collection_name', 'upanzi_knowledge')
        self.declare_parameter('verbose', False)
        
        # Log configuration (hide API key)
        self.get_logger().info(f"LLM URL: {config.llm_base_url}")
        self.get_logger().info(f"LLM Model: {config.llm_model}")
        self.get_logger().info(f"LLM API Key: {'***' + config.llm_api_key[-4:] if len(config.llm_api_key) > 4 else '(default)'}")
        self.get_logger().info(f"Embedding Model: {config.embedding_model}")
        self.get_logger().info(f"Data Default Path: {config.data_default_path}")
        
        # State
        self.collection = None
        self.conversation_history: List[Dict] = []
        
        # Try to load existing collection
        collection_name = self.get_parameter('collection_name').get_parameter_value().string_value
        self.load_collection(collection_name)
        
        # Services - use the correct imported service types
        self.create_service(LanguageModelPrompt, PROMPT_SERVICE, self.prompt_callback)
        self.create_service(LanguageModelPrompt, GET_INTENT_SERVICE, self.get_intent_callback)
        self.create_service(LanguageModelCreateCollection, CREATE_COLLECTION, self.create_collection_callback)
        
        status = f"Collection: {collection_name}" if self.collection else "Collection: Not loaded"
        self.get_logger().info(f"RAG Node ready. {status}")
    
    def load_collection(self, name: str):
        """Load existing collection"""
        try:
            self.log_verbose(f"Attempting to load collection: {name}")
            self.collection = get_collection(name)
            if self.collection:
                count = self.collection.count()
                self.get_logger().info(f"Loaded collection '{name}' with {count} documents")
                self.log_verbose(f"Collection metadata: {self.collection.metadata}")
            else:
                self.log_verbose(f"Collection '{name}' not found")
        except Exception as e:
            self.get_logger().warn(f"Could not load collection: {e}")
            self.log_verbose(f"Collection load error details: {str(e)}")
    
    @property
    def verbose(self) -> bool:
        config = get_config()
        config_verbose = config.verbose
        param_verbose = self.get_parameter('verbose').get_parameter_value().bool_value
        verbose_mode = config_verbose or param_verbose
        return verbose_mode
    
    def log_verbose(self, message: str) -> None:
        """Log message only if verbose mode is enabled"""
        if self.verbose:
            self.get_logger().info(f"[VERBOSE] {message}")
    
    def prompt_callback(self, request, response):
        """Handle RAG query"""
        
        self.log_verbose(f"Prompt callback invoked with request: {request}")
        
        # Check collection
        if self.collection is None:
            self.log_verbose("Collection is None - knowledge base not initialized")
            response.response = "Knowledge base not initialized. Please set up the collection first."
            return response
        
        query = request.prompt.strip()
        if not query:
            self.log_verbose("Empty query received")
            response.response = "I didn't catch that. Could you please repeat it?"
            return response
        
        self.log_verbose(f"Processing query: '{query}'")
        self.log_verbose(f"Current conversation history length: {len(self.conversation_history)}")
        
        try:
            # Handle query
            self.log_verbose("Calling handle_query...")
            result = handle_query(
                collection=self.collection,
                query=query,
                conversation_history=self.conversation_history
            )
            
            # Update history
            self.conversation_history.append({"role": "user", "content": request.prompt})
            self.conversation_history.append({"role": "assistant", "content": result['response']})
            
            config = get_config()
            # Keep history manageable
            if len(self.conversation_history) > config.max_history_turns:
                self.log_verbose(f"Truncating conversation history from {len(self.conversation_history)} to {config.max_history_turns} turns")
                self.conversation_history = self.conversation_history[-config.max_history_turns:]
            
            self.log_verbose(f"Response generated: {result['response'][:200]}...")
            self.log_verbose(f"Sources used: {result['sources']}")
            self.log_verbose(f"Updated conversation history length: {len(self.conversation_history)}")
            
            response.response = result['response']
            
        except Exception as e:
            self.get_logger().error(f"Query error: {e}")
            self.log_verbose(f"Detailed error in prompt_callback: {str(e)}")
            response.response = "Something went wrong. Please try again."
        
        self.log_verbose("Prompt callback completed")
        return response
    
    def get_intent_callback(self, request, response):
        """Handle intent extraction"""
        
        self.log_verbose(f"Get intent callback invoked with request: {request}")
        
        # Check collection
        if self.collection is None:
            self.log_verbose("Collection is None - knowledge base not initialized")
            response.response = "Knowledge base not initialized. Please set up the collection first."
            return response
        
        query = request.prompt.strip()
        if not query:
            self.log_verbose("Empty query received for intent extraction")
            response.response = "I didn't catch that. Could you please repeat it?"
            return response
        
        self.log_verbose(f"Processing intent extraction for query: '{query}'")
        
        try:
            # Handle query for intent
            self.log_verbose("Calling handle_query for intent extraction...")
            result = handle_query(
                collection=self.collection,
                query=query,
                get_intent=True
            )
            
            self.log_verbose(f"Intent extracted: {result['response']}")
            response.response = result['response']
            
        except Exception as e:
            self.get_logger().error(f"Intent extraction error: {e}")
            self.log_verbose(f"Detailed error in get_intent_callback: {str(e)}")
            response.response = "Something went wrong during intent extraction. Please try again."
        
        self.log_verbose("Get intent callback completed")
        return response

    def create_collection_callback(self, request, response):
        """Create and populate collection"""
        
        self.log_verbose(f"Create collection callback invoked: name='{request.name}', datafile_path='{request.datafile_path}'")
        
        name = request.name.strip() if request.name else ""
        data_path = request.datafile_path.strip() if request.datafile_path else ""
        description = request.description.strip() if request.description else ""
        
        if not name:
            self.log_verbose("Collection name not provided")
            response.success = 0
            response.message = "Collection name required"
            return response
        
        self.log_verbose(f"Processing collection creation: name='{name}', description='{description}'")
        
        # Use default data path from configuration if not provided
        if not data_path:
            config = get_config()
            data_path = config.data_default_path
            self.get_logger().info(f"Using default data path from config: {data_path}")
            self.log_verbose(f"Default data path from config: {data_path}")
        
        # Convert to absolute path if relative
        data_path_obj = Path(data_path)
        if not data_path_obj.is_absolute():
            data_path_obj = PACKAGE_PATH / data_path_obj
            self.log_verbose(f"Converted relative path to absolute: {data_path_obj}")
        
        if not data_path_obj.exists():
            self.log_verbose(f"Data file not found: {data_path_obj}")
            response.success = 0
            response.message = f"File not found: {data_path_obj}"
            return response
        
        self.log_verbose(f"Data file exists: {data_path_obj}")
        self.log_verbose(f"Starting collection creation: {name} from {data_path_obj}")
        
        try:
            collection = setup_collection(name, str(data_path_obj), description)
            
            if collection:
                self.collection = collection
                self.conversation_history = []
                
                count = collection.count()
                response.success = 1
                response.message = f"Collection '{name}' ready with {count} documents"
                self.get_logger().info(response.message)
                self.log_verbose(f"Collection created successfully with {count} documents")
                self.log_verbose(f"Collection metadata: {collection.metadata}")
            else:
                self.log_verbose("Collection creation returned None")
                response.success = 0
                response.message = "Failed to create collection"
                
        except Exception as e:
            self.get_logger().error(f"Error creating collection: {e}")
            self.log_verbose(f"Detailed error in create_collection_callback: {str(e)}")
            response.success = 0
            response.message = f"Error: {e}"
        
        self.log_verbose("Create collection callback completed")
        return response
    
    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []

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
