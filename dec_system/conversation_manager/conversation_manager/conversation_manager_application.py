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

Actions:
    prompt (dec_interfaces/action/ConversationManager): Answer questions using RAG
        Feedback: status – "searching" | "generating"
        Result:   success, response

Author: Yohannes Tadesse Haile, Carnegie Mellon University Africa
Email: yohatad123@gmail.com
Date: February 28, 2026
Version: v1.0
"""

import rclpy
from pathlib import Path
from typing import List, Dict
from rclpy.node import Node
from rclpy.action import ActionServer
from ament_index_python.packages import get_package_share_directory
from dec_interfaces.action import ConversationManager
from .conversation_manager_implementation import (
    get_config,
    get_collection,
    setup_collection,
    handle_query,
    search,
    generate_response,
    generate_response_stream,
    extract_answer_from_raw,
    extract_intent_from_raw,
    apply_speech_tags,
    apply_config_file,
)

PACKAGE_PATH = Path(get_package_share_directory('conversation_manager'))

class ConversationManagerNode(Node):
    
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
        
        # Action server
        self._action_server = ActionServer(self, ConversationManager, 'conversation_manager', self.execute_callback)

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
    
    def execute_callback(self, goal_handle):
        """Handle RAG query as an action.

        Flow:
          1. Vector search (fast, ~50-200ms)
          2. Streaming LLM call — sentences are accumulated into the full response.
          3. Action result carries the full answer text; the BT TTS node is the
             sole consumer responsible for playback.
        """
        self.log_verbose(f"Action goal received: prompt=\"{goal_handle.request.prompt}\"")

        result = ConversationManager.Result()

        if self.collection is None:
            self.log_verbose("Collection is None - knowledge base not initialized")
            result.success = False
            result.response = "Knowledge base not initialized. Please restart the node."
            goal_handle.abort()
            return result

        query = goal_handle.request.prompt.strip() if goal_handle.request.prompt else ""
        if not query:
            self.log_verbose("Empty query received")
            result.success = False
            result.response = "I didn't catch that. Could you repeat?"
            goal_handle.abort()
            return result

        self.log_verbose(f"Processing query: '{query}'")

        feedback_msg = ConversationManager.Feedback()

        try:
            config = get_config()

            # Stage 1: vector search
            feedback_msg.status = 'searching'
            goal_handle.publish_feedback(feedback_msg)
            self.log_verbose("Searching knowledge base...")
            search_results = search(self.collection, query)

            # Stage 2: streaming LLM generation
            # Sentences are accumulated; the BT TTS node speaks the full response.
            feedback_msg.status = 'generating'
            goal_handle.publish_feedback(feedback_msg)
            self.log_verbose("Streaming response...")

            raw_out: List[str] = []
            yielded_sentences: List[str] = []

            for sentence in generate_response_stream(
                query, search_results, self.conversation_history, raw_out
            ):
                yielded_sentences.append(sentence)
                self.log_verbose(f"Accumulated: '{sentence}'")

            # Derive the canonical answer text, intent, and confidence
            # from the full raw LLM buffer captured during streaming.
            raw = raw_out[0] if raw_out else ""

            response_text = extract_answer_from_raw(raw) if raw else ""
            if not response_text.strip():
                response_text = " ".join(yielded_sentences)

            intent, confidence = extract_intent_from_raw(raw) if raw else ("UNKNOWN", 0.0)
            self.log_verbose(f"Intent: {intent} (confidence={confidence:.2f})")

            # Apply sentence-level speed tag based on intent (e.g. \rspd=85\ for
            # technical answers). Word-level tags (*pau=N*) are already converted
            # inside extract_answer_from_raw().
            response_text = apply_speech_tags(response_text, intent)

            # Update conversation history
            self.conversation_history.append({'query': query, 'response': response_text})
            if len(self.conversation_history) > config.max_history_turns:
                self.log_verbose(f"Truncating history to {config.max_history_turns} turns")
                self.conversation_history = self.conversation_history[-config.max_history_turns:]

            self.log_verbose(f"Response: {response_text[:200]}")

            result.success    = True
            result.response   = response_text
            result.intent     = intent
            result.confidence = confidence
            goal_handle.succeed()

        except Exception as e:
            self.get_logger().error(f"Query error: {e}")
            self.log_verbose(f"Detailed error in execute_callback: {str(e)}")
            result.success = False
            result.response = "Something went wrong. Please try again."
            goal_handle.abort()

        self.log_verbose("Action callback completed")
        return result
    
    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []
        self.log_verbose("Conversation history cleared")

def main(args=None):
    node = None
    try:
        rclpy.init(args=args)
        node = ConversationManagerNode()
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