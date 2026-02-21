#!/usr/bin/env python3
"""
ROS2 node developed for conversation management for Pepper robot, utilizing Retrieval-Augmented Generation 
(RAG) techniques.

The application controls when to call the RAG service.

Environment Variables:
    LLM_API_KEY: API key for LLM service (MUST be exported as environment variable)

Configuration (config/converation_manager_configuration.yaml):
    llm:         base_url, model
    embedding:   model
    search:      similarity_threshold, top_k
    conversation: max_history_turns, context_turns
    data:        default_path (knowledge base JSON file)
    debug:       verbose

ROS Parameters:
    collection_name (str, default: 'upanzi_knowledge'): ChromaDB collection name
    verbose        (bool, default: False): Enable verbose logging

Services:
    prompt (cssr_interfaces/srv/ConversationManagerPrompt): Answer questions using RAG
"""

import rclpy
from pathlib import Path
from typing import List, Dict
from rclpy.node import Node
from ament_index_python.packages import get_package_share_directory
from cssr_interfaces.srv import ConversationManagerPrompt
from .conversation_manager_implementation import (
    get_config,
    get_collection,
    setup_collection,
    handle_query,
    apply_config_file,
)

PACKAGE_PATH = Path(get_package_share_directory('conversation_manager'))

class conversationManager(Node):
    
    def __init__(self):
        super().__init__('conversation_manager')
        
        # Load configuration from YAML file
        config_file = PACKAGE_PATH / 'config' / 'converation_manager_configuration.yaml'
        if config_file.exists():
            success, messages = apply_config_file(str(config_file))
            if success:
                self.get_logger().info(f"Configuration loaded from {config_file}")
                for msg in messages:
                    self.get_logger().info(f"Config note: {msg}")
            else:
                self.get_logger().error(f"Failed to load configuration: {messages}")
                raise RuntimeError(f"Configuration error: {messages}")
        else:
            self.get_logger().error(f"Configuration file not found: {config_file}")
            raise FileNotFoundError(f"Required config file not found: {config_file}")
        
        # Log verbose status
        config = get_config()
        if config.verbose:
            self.get_logger().info("Verbose mode is ENABLED - detailed logging will be displayed")
        else:
            self.get_logger().info("Verbose mode is DISABLED - only essential logs will be shown")
        
        # Parameters
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
        
        # Load collection from configuration
        collection_name = self.get_parameter('collection_name').get_parameter_value().string_value
        self.initialize_collection(collection_name)
        
        # Service
        self.create_service(ConversationManagerPrompt, 'prompt', self.prompt_callback)
        
        status = f"Collection: {collection_name} ({self.collection.count()} docs)" if self.collection else "Collection: Failed to load"
        self.get_logger().info(f"RAG Node ready. {status}")
    
    def initialize_collection(self, name: str):
        """Initialize collection from configuration"""
        config = get_config()
        
        self.log_verbose(f"Initializing collection: {name}")
        
        # First, try to load existing collection
        try:
            self.collection = get_collection(name)
            if self.collection:
                count = self.collection.count()
                self.get_logger().info(f"Loaded existing collection '{name}' with {count} documents")
                self.log_verbose(f"Collection metadata: {self.collection.metadata}")
                return
        except Exception as e:
            self.log_verbose(f"Could not load existing collection: {e}")
        
        # If collection doesn't exist, create it from data file
        self.get_logger().info(f"Collection '{name}' not found, creating from data file...")
        
        data_path = config.data_default_path
        self.log_verbose(f"Using data path from config: {data_path}")
        
        # Convert to absolute path if relative
        data_path_obj = Path(data_path)
        if not data_path_obj.is_absolute():
            data_path_obj = PACKAGE_PATH / data_path_obj
            self.log_verbose(f"Converted relative path to absolute: {data_path_obj}")
        
        if not data_path_obj.exists():
            self.get_logger().error(f"Data file not found: {data_path_obj}")
            raise FileNotFoundError(f"Required data file not found: {data_path_obj}")
        
        self.log_verbose(f"Data file exists: {data_path_obj}")
        
        try:
            # Create and populate collection
            self.collection = setup_collection(
                name=name,
                json_path=str(data_path_obj),
                description="Upanzi Network Knowledge Base"
            )
            
            if self.collection:
                count = self.collection.count()
                self.get_logger().info(f"Created collection '{name}' with {count} documents")
                self.log_verbose(f"Collection metadata: {self.collection.metadata}")
            else:
                raise RuntimeError("setup_collection returned None")
                
        except Exception as e:
            self.get_logger().error(f"Failed to create collection: {e}")
            self.log_verbose(f"Detailed error: {str(e)}")
            raise
    
    @property
    def verbose(self) -> bool:
        return get_config().verbose or self.get_parameter('verbose').get_parameter_value().bool_value
    
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
            response.response = "Knowledge base not initialized. Please restart the node."
            return response
        
        query = request.prompt.strip() if request.prompt else ""
        if not query:
            self.log_verbose("Empty query received")
            response.response = "I didn't catch that. Could you repeat?"
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
            self.conversation_history.append({
                'query': query,
                'response': result['response']
            })
            
            # Keep history manageable
            config = get_config()
            if len(self.conversation_history) > config.max_history_turns:
                self.log_verbose(f"Truncating conversation history from {len(self.conversation_history)} to {config.max_history_turns}")
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
    
    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []
        self.log_verbose("Conversation history cleared")

def main(args=None):
    node = None
    try:
        rclpy.init(args=args)
        node = conversationManager()
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\nShutdown requested (Ctrl+C)...")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if node:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()