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

Lifecycle:
    configure  → load YAML config, init ChromaDB collection, create lifecycle publisher
                 + action server
    activate   → activate publishers (managed by super().on_activate)
    deactivate → deactivate publishers (managed by super().on_deactivate)
    cleanup    → destroy publisher, clear collection + history

Author: Yohannes Tadesse Haile, Carnegie Mellon University Africa
Email: yohatad123@gmail.com
Date: February 28, 2026
Version: v1.0
"""

import rclpy
from pathlib import Path
from typing import List, Dict
from rclpy.action import ActionServer
from rclpy.lifecycle import LifecycleNode, TransitionCallbackReturn
from ament_index_python.packages import get_package_share_directory
from std_msgs.msg import String
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
    apply_config_file,
)

PACKAGE_PATH = Path(get_package_share_directory('conversation_manager'))


class ConversationManagerNode(LifecycleNode):

    def __init__(self):
        super().__init__('conversation_manager')

        # Declare parameters — heavy initialisation deferred to on_configure
        self.declare_parameter('collection_name', 'upanzi_knowledge')
        self.declare_parameter('verbose', False)

    # ── Lifecycle callbacks ─────────────────────────────────────────────────────

    def on_configure(self, _state) -> TransitionCallbackReturn:
        """Load config, init ChromaDB collection, create publisher and action server."""

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
                return TransitionCallbackReturn.FAILURE
        else:
            self.get_logger().error(f"Configuration file not found: {config_file}")
            return TransitionCallbackReturn.FAILURE

        # Log verbose status
        config = get_config()
        if config.verbose:
            self.get_logger().info("Verbose mode is ENABLED - detailed logging will be displayed")
        else:
            self.get_logger().info("Verbose mode is DISABLED - only essential logs will be shown")

        # Log configuration (hide API key)
        self.get_logger().info(f"LLM URL: {config.llm_base_url}")
        self.get_logger().info(f"LLM Model: {config.llm_model}")
        self.get_logger().info(
            f"LLM API Key: {'***' + config.llm_api_key[-4:] if len(config.llm_api_key) > 4 else '(default)'}"
        )
        self.get_logger().info(f"Embedding Model: {config.embedding_model}")
        self.get_logger().info(f"Data Default Path: {config.data_default_path}")

        # State
        self.collection = None
        self.conversation_history: List[Dict] = []

        # Init ChromaDB collection
        collection_name = self.get_parameter('collection_name').get_parameter_value().string_value
        try:
            self.initialize_collection(collection_name)
        except Exception as e:
            self.get_logger().error(f"Failed to initialize collection: {e}")
            return TransitionCallbackReturn.FAILURE

        # Managed publisher — silenced while INACTIVE, active once activated
        self.stream_pub = self.create_lifecycle_publisher(String, '/tts/input', 10)

        # Action server
        self._action_server = ActionServer(
            self, ConversationManager, 'conversation_manager', self.execute_callback
        )

        status = (
            f"Collection: {collection_name} ({self.collection.count()} docs)"
            if self.collection
            else "Collection: Failed to load"
        )
        self.get_logger().info(f"conversation_manager: configured. {status}")
        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, _state) -> TransitionCallbackReturn:
        """Activate managed publishers."""
        super().on_activate(_state)
        self.get_logger().info("conversation_manager: activated — ready to answer queries")
        return TransitionCallbackReturn.SUCCESS

    def on_deactivate(self, _state) -> TransitionCallbackReturn:
        """Deactivate managed publishers."""
        super().on_deactivate(_state)
        self.get_logger().info("conversation_manager: deactivated")
        return TransitionCallbackReturn.SUCCESS

    def on_cleanup(self, _state) -> TransitionCallbackReturn:
        """Destroy publisher and clear in-memory state."""
        self.destroy_lifecycle_publisher(self.stream_pub)
        self.collection = None
        self.conversation_history = []
        self.get_logger().info("conversation_manager: cleaned up")
        return TransitionCallbackReturn.SUCCESS

    def on_shutdown(self, _state) -> TransitionCallbackReturn:
        self.get_logger().info("conversation_manager: shutting down")
        return TransitionCallbackReturn.SUCCESS

    # ── Collection initialisation ───────────────────────────────────────────────

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

    # ── Helpers ──────────────────────────────────────────────────────────────────

    @property
    def verbose(self) -> bool:
        return get_config().verbose or self.get_parameter('verbose').get_parameter_value().bool_value

    def log_verbose(self, message: str) -> None:
        """Log message only if verbose mode is enabled"""
        if self.verbose:
            self.get_logger().info(f"[VERBOSE] {message}")

    # ── Action callback ───────────────────────────────────────────────────────────

    def execute_callback(self, goal_handle):
        """Handle RAG query as an action, streaming sentences to TTS as they arrive.

        Flow:
          1. Vector search (fast, ~50-200ms)
          2. Streaming LLM call — each complete answer sentence is published to
             /tts/input so text_to_speech can begin
             speaking before the full response is ready.
          3. Action result carries the full answer text for any other consumer.
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
            # Sentences are published to /tts/input
            # as they arrive, allowing TTS to start speaking immediately.
            feedback_msg.status = 'generating'
            goal_handle.publish_feedback(feedback_msg)
            self.log_verbose("Streaming response...")

            raw_out: List[str] = []
            yielded_sentences: List[str] = []

            for sentence in generate_response_stream(
                query, search_results, self.conversation_history, raw_out
            ):
                yielded_sentences.append(sentence)
                msg = String()
                msg.data = sentence
                self.stream_pub.publish(msg)
                self.log_verbose(f"Streamed: '{sentence}'")

            # Derive the canonical answer text, intent, and confidence
            # from the full raw LLM buffer captured during streaming.
            raw = raw_out[0] if raw_out else ""

            response_text = extract_answer_from_raw(raw) if raw else ""
            if not response_text.strip():
                response_text = " ".join(yielded_sentences)

            intent, confidence = extract_intent_from_raw(raw) if raw else ("UNKNOWN", 0.0)
            self.log_verbose(f"Intent: {intent} (confidence={confidence:.2f})")

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


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

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
