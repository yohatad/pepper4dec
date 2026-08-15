#!/usr/bin/env python3
""" conversation_manager_application.py

Entry point for the ConversationManagerNode lifecycle node.
Running this node provides a Retrieval-Augmented Generation (RAG) action server
that answers natural-language questions about the Upanzi Network knowledge base.

On configure, the node builds its RAG configuration from ROS parameters,
initializes (or builds) a ChromaDB collection from the knowledge-base JSON
file, and starts an action server for handling conversational queries. Each
goal triggers a vector search over the knowledge base followed by a streaming
LLM call; sentences are accumulated into the full response as they arrive.
The action result carries the full answer text, classified intent, and
confidence score. The BT SpeechWithFeedback node is the sole consumer
responsible for playback, using NAOqi ALAnimatedSpeech to interpret embedded
prosody tags and drive gestures.

Actions:
    /conversation_manager (dec_interfaces/action/ConversationManager)
        Answer a natural-language prompt using RAG; feedback reports
        "searching" | "generating", result carries success, response,
        intent, and confidence

Parameters (declared here, populated from config/converation_manager_configuration.yaml
via the launch file's parameters=[...]; run `ros2 param get/set /conversation_manager
<name>` to inspect or override at runtime):
    collection_name (str), verbose (bool), llm_base_url (str), llm_model (str),
    embedding_model (str), retrieval_mode (str), similarity_threshold (float),
    top_k (int), max_history_turns (int), context_turns (int),
    max_response_sentences (int), data_default_path (str)

    LLM_API_KEY is never a ROS parameter -- it must be exported as an
    environment variable, since ROS parameters are visible via `ros2 param
    dump`/introspection tools.

Author: Yohannes Tadesse Haile
Affiliation: Carnegie Mellon University Africa
Email: yohatad123@gmail.com
Date: February 28, 2026
Version: v1.0
"""

import rclpy
import rclpy.logging
from pathlib import Path
from typing import List, Dict
from rclpy.action import ActionServer
from rclpy.lifecycle import LifecycleNode, TransitionCallbackReturn
from ament_index_python.packages import get_package_share_directory
from dec_interfaces.action import ConversationManager
from .conversation_manager_implementation import (
    ConfigError,
    ConversationManagerConfig,
    DEFAULT_CONTEXT_TURNS,
    DEFAULT_DATA_PATH,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_LLM_BASE_URL,
    DEFAULT_LLM_MODEL,
    DEFAULT_MAX_HISTORY_TURNS,
    DEFAULT_MAX_RESPONSE_SENTENCES,
    DEFAULT_RETRIEVAL_MODE,
    DEFAULT_SIMILARITY_THRESHOLD,
    DEFAULT_TOP_K,
    DEFAULT_VERBOSE,
    get_config,
    set_config,
    get_collection,
    setup_collection,
    retrieve_documents,
    generate_response_stream,
    extract_answer_from_raw,
    extract_intent_from_raw,
    apply_speech_tags,
)

PACKAGE_PATH = Path(get_package_share_directory('conversation_manager'))


class ConversationManagerNode(LifecycleNode):
    """Lifecycle node that answers conversational queries via RAG and streams responses to TTS."""

    def __init__(self):
        super().__init__('conversation_manager')

        # Declare parameters — heavy initialisation deferred to on_configure.
        # Defaults here are generic fallbacks; config/converation_manager_configuration.yaml
        # supplies this deployment's actual values via the launch file's parameters=[...].
        self.declare_parameter('collection_name', 'upanzi_knowledge')
        self.declare_parameter('verbose', DEFAULT_VERBOSE)
        self.declare_parameter('llm_base_url', DEFAULT_LLM_BASE_URL)
        self.declare_parameter('llm_model', DEFAULT_LLM_MODEL)
        self.declare_parameter('embedding_model', DEFAULT_EMBEDDING_MODEL)
        self.declare_parameter('retrieval_mode', DEFAULT_RETRIEVAL_MODE)
        self.declare_parameter('similarity_threshold', DEFAULT_SIMILARITY_THRESHOLD)
        self.declare_parameter('top_k', DEFAULT_TOP_K)
        self.declare_parameter('max_history_turns', DEFAULT_MAX_HISTORY_TURNS)
        self.declare_parameter('context_turns', DEFAULT_CONTEXT_TURNS)
        self.declare_parameter('max_response_sentences', DEFAULT_MAX_RESPONSE_SENTENCES)
        self.declare_parameter('data_default_path', DEFAULT_DATA_PATH)

    # ── Lifecycle callbacks ─────────────────────────────────────────────────────

    def on_configure(self, _state) -> TransitionCallbackReturn:
        """Build RAG config from ROS params, init the ChromaDB collection, and start the action
        server.
        """

        # LLM_API_KEY is deliberately excluded: ConversationManagerConfig's
        # default_factory pulls it from the environment, never from a ROS parameter.
        config = ConversationManagerConfig(
            llm_base_url=self.get_parameter('llm_base_url').value,
            llm_model=self.get_parameter('llm_model').value,
            embedding_model=self.get_parameter('embedding_model').value,
            retrieval_mode=self.get_parameter('retrieval_mode').value,
            similarity_threshold=self.get_parameter('similarity_threshold').value,
            default_top_k=self.get_parameter('top_k').value,
            max_history_turns=self.get_parameter('max_history_turns').value,
            context_turns=self.get_parameter('context_turns').value,
            max_response_sentences=self.get_parameter('max_response_sentences').value,
            data_default_path=self.get_parameter('data_default_path').value,
            verbose=self.get_parameter('verbose').value,
        )
        try:
            set_config(config)
        except ConfigError as e:
            self.get_logger().error(f"Invalid configuration: {e}")
            return TransitionCallbackReturn.FAILURE

        # Log verbose status
        if config.verbose:
            self.get_logger().info("Verbose mode is ENABLED - detailed logging will be displayed")
        else:
            self.get_logger().info("Verbose mode is DISABLED - only essential logs will be shown")

        # Log configuration (hide API key)
        self.get_logger().info(f"LLM URL: {config.llm_base_url}")
        self.get_logger().info(f"LLM Model: {config.llm_model}")
        masked_key = (
            '***' + config.llm_api_key[-4:] if len(config.llm_api_key) > 4 else '(default)'
        )
        self.get_logger().info(f"LLM API Key: {masked_key}")
        self.get_logger().info(f"Embedding Model: {config.embedding_model}")
        self.get_logger().info(f"Data Default Path: {config.data_default_path}")
        self.get_logger().info(f"Retrieval Mode: {config.retrieval_mode}")

        # State
        self.collection = None
        self.conversation_history: List[Dict] = []

        collection_name = self.get_parameter('collection_name').get_parameter_value().string_value

        if config.retrieval_mode == 'rag':
            # Init ChromaDB collection (only needed for vector search)
            try:
                self.initialize_collection(collection_name)
            except Exception as e:
                self.get_logger().error(f"Failed to initialize collection: {e}")
                return TransitionCallbackReturn.FAILURE
        else:
            self.get_logger().info(
                "Retrieval mode is 'full_context' — skipping ChromaDB/embedding initialization"
            )

        # Action server
        self._action_server = ActionServer(
            self, ConversationManager, '/conversation_manager', self.execute_callback
        )

        if config.retrieval_mode == 'rag':
            status = (
                f"Collection: {collection_name} ({self.collection.count()} docs)"
                if self.collection
                else "Collection: Failed to load"
            )
        else:
            status = "Full-context mode: knowledge base sent directly to LLM"
        self.get_logger().info(f"conversation_manager: configured. {status}")
        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, _state) -> TransitionCallbackReturn:
        """Activate the managed text-to-speech publisher so the node is ready to answer queries."""
        super().on_activate(_state)
        self.get_logger().info("conversation_manager: activated — ready to answer queries")
        return TransitionCallbackReturn.SUCCESS

    def on_deactivate(self, _state) -> TransitionCallbackReturn:
        """Deactivate the managed text-to-speech publisher."""
        super().on_deactivate(_state)
        self.get_logger().info("conversation_manager: deactivated")
        return TransitionCallbackReturn.SUCCESS

    def on_cleanup(self, _state) -> TransitionCallbackReturn:
        """Destroy the action server, and clear the ChromaDB collection and history."""
        self._action_server.destroy()
        self.collection = None
        self.conversation_history = []
        self.get_logger().info("conversation_manager: cleaned up")
        return TransitionCallbackReturn.SUCCESS

    def on_shutdown(self, _state) -> TransitionCallbackReturn:
        """Log that the node is shutting down."""
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
                self.get_logger().info(
                    f"Loaded existing collection '{name}' with {count} documents")
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
        return get_config().verbose

    def log_verbose(self, message: str) -> None:
        """Log message only if verbose mode is enabled"""
        if self.verbose:
            self.get_logger().info(f"[VERBOSE] {message}")

    # ── Action callback ───────────────────────────────────────────────────────────

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

        config = get_config()
        if config.retrieval_mode == 'rag' and self.collection is None:
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
            # Stage 1: retrieval (vector search in "rag" mode, whole KB in "full_context" mode)
            feedback_msg.status = 'searching'
            goal_handle.publish_feedback(feedback_msg)
            self.log_verbose("Retrieving knowledge base context...")
            search_results = retrieve_documents(self.collection, query)

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

            result.success = True
            result.response = response_text
            result.intent = intent
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
        # If node exists, use its logger; else fall back to a standalone ROS logger
        if node is not None:
            node.get_logger().info("Shutdown requested (Ctrl+C)...")
        else:
            rclpy.logging.get_logger("conversation_manager").info("Shutdown requested (Ctrl+C)...")
    except Exception as e:
        if node is not None:
            node.get_logger().error(f"Error: {e}")
        else:
            rclpy.logging.get_logger("conversation_manager").error(f"Error: {e}")
    finally:
        if node:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
