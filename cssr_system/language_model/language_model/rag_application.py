"""
Simplified RAG ROS 2 Node for Pepper Robot Lab Assistant

Clean implementation without intent classification.
The application controls when to call the RAG service.
"""

import sys
from pathlib import Path
from typing import List, Dict, Optional

import rclpy
from rclpy.node import Node

# Package path setup
try:
    from ament_index_python.packages import get_package_share_directory
    PACKAGE_PATH = Path(get_package_share_directory('language_model'))
except ImportError:
    PACKAGE_PATH = Path(__file__).parent.parent.parent.parent.parent.parent.parent / "src" / "language_model"

sys.path.insert(0, str(PACKAGE_PATH))

from llm_interfaces.srv import Prompt, CreateCollection
from simple_rag import (
    RAGConfig,
    RAGError,
    set_config,
    get_config,
    get_collection,
    setup_collection,
    handle_query,
    read_config_file
)


class SimpleRAGNode(Node):
    """
    Simple RAG Node for Pepper robot.
    
    Services:
        - rag_prompt: Answer questions using RAG
        - create_collection: Initialize knowledge base
    """
    
    def __init__(self):
        super().__init__('rag_node')
        
        # Parameters
        self.declare_parameter('config_path', str(PACKAGE_PATH / 'config' / 'ragSystem.ini'))
        self.declare_parameter('collection_name', 'upanzi_knowledge')
        self.declare_parameter('verbose', False)
        
        # Load config
        config_path = self.get_parameter('config_path').get_parameter_value().string_value
        self._load_config(config_path)
        
        # State
        self._collection = None
        self._conversation_history: List[Dict] = []
        
        # Try to load existing collection
        collection_name = self.get_parameter('collection_name').get_parameter_value().string_value
        self._load_collection(collection_name)
        
        # Services
        self.create_service(Prompt, 'rag_prompt', self._prompt_callback)
        self.create_service(CreateCollection, 'create_collection', self._create_collection_callback)
        
        self.get_logger().info(f"RAG Node ready. Collection: {collection_name if self._collection else 'Not loaded'}")
    
    def _load_config(self, config_path: str):
        """Load configuration"""
        try:
            file_config = read_config_file(config_path)
            
            config = RAGConfig(
                llm_base_url=file_config.get('llmBaseUrl', RAGConfig.llm_base_url),
                llm_model=file_config.get('llmModel', RAGConfig.llm_model),
                chroma_host=file_config.get('chromaHost', RAGConfig.chroma_host),
                chroma_port=int(file_config.get('chromaPort', str(RAGConfig.chroma_port))),
                embedding_model=file_config.get('embeddingModel', RAGConfig.embedding_model),
                similarity_threshold=float(file_config.get('similarityThreshold', str(RAGConfig.similarity_threshold))),
                default_top_k=int(file_config.get('topK', str(RAGConfig.default_top_k))),
            )
            set_config(config)
            self._verbose = file_config.get('verboseMode', 'false').lower() == 'true'
            
        except Exception as e:
            self.get_logger().warn(f"Config load failed, using defaults: {e}")
            self._verbose = False
    
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
        return self._verbose or self.get_parameter('verbose').get_parameter_value().bool_value
    
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