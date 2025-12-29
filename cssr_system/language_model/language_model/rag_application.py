#!/home/roboticslab/miniconda3/envs/llama-3.1/bin python3

import sys
from pathlib import Path

parent_dir = Path(__file__).parent.parent.parent.parent.parent.parent.parent
parent_dir = parent_dir / "src/language_model"
sys.path.append(str(parent_dir / "language_model"))  # Ensure parent directory is in sys.path

from llm_interfaces.srv import Prompt, CreateCollection
from ragImplementation import *
from rclpy.node import Node
import rclpy

class RAGNode(Node):
    def __init__(self):
        super().__init__('rag_node')
        self.srv = self.create_service(Prompt, 'rag_prompt', self.rag_prompt_callback)
        self.create_collection_srv = self.create_service(CreateCollection, 'create_collection', self.create_collection_callback)

        self.get_logger().info("RAG Node is up and running.")

        # print(f"Parent directory before reading config: {Path(__file__).parent.parent.parent.parent.parent.parent.parent}")

        self.config = read_config(Path(__file__).parent.parent.parent.parent.parent.parent.parent / 'src' / 'language_model' / 'config' / 'ragSystem.ini')

        # Get collection for RAG system
        self.collection = get_similarity_search_collection("interactive_upanzi_search")

        if self.collection is None:
            self.get_logger().error("Call the create_collection service to create a collection before using the RAG system.")

        self.conversation_history = []

    def rag_prompt_callback(self, request, response):
        # Implement your callback logic here
        verbose_mode = self.config.get('verboseMode', 'false').lower() == 'true'

        if verbose_mode:
            self.get_logger().info(f"Received prompt: {request.prompt}")

        ai_response = handle_rag_query(self.collection, request.prompt, self.conversation_history, verbose_mode, int(self.config.get('topK', 3)))

        self.conversation_history.append({"role": "user", "content": request.prompt, "response": ai_response})

        # Keep conversation history manageable
        if len(self.conversation_history) > 5:
            self.conversation_history = self.conversation_history[-3:]

        if verbose_mode:
            self.get_logger().info(f"AI response: {ai_response}")

        response.response = ai_response
        return response

    def create_collection_callback(self, request, response):
        verbose_mode = self.config.get('verboseMode', 'false').lower() == 'true'

        if verbose_mode:
            self.get_logger().info(f"Creating collection: {request.name}")

        collection = create_collection_and_load_data(request.name, request.description, request.datafile_path, verbose_mode)

        if collection:
            response.success = 1
            response.message = f"Collection '{request.name}' created successfully."

            self.collection = collection  # Update the node's collection

            if verbose_mode:
                self.get_logger().info(response.message)
        else:
            response.success = 0
            response.message = f"Failed to create collection '{request.name}'. Debug logs for details."
            if verbose_mode:
                self.get_logger().info(response.message)

        return response


def main(args=None):
    rclpy.init(args=args)
    rag_node = RAGNode()
    rclpy.spin(rag_node)
    rag_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()