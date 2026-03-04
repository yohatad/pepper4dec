<div align="center">
<h1> Conversation Manager</h1>
</div>

<div align="center">
  <img src="../upanzi-logo.svg" alt="Upanzi Logo" style="width:50%; height:auto;">
</div>

The **Conversation Manager Package** implements a **Retrieval-Augmented Generation (RAG)** system for the Pepper robot to serve as a lab assistant at the Upanzi Network, Carnegie Mellon University Africa. The system allows Pepper to answer questions about Upanzi Network's research, projects, facilities, and impact areas using a knowledge base built from structured JSON data.

## Key Features
- **ROS2 Native**: Built for ROS2 Humble
- **Retrieval-Augmented Generation**: Combines vector search with large language models for accurate, context-aware responses
- **ChromaDB Integration**: Local vector database for privacy-preserving knowledge storage
- **Configurable LLM Support**: Compatible with any OpenAI-compatible API (DeepSeek, Groq, etc.)
- **Real-time Query Processing**: Processes natural language questions and returns knowledgeable responses
- **Conversation Memory**: Maintains context from previous interactions (last 5 turns)
- **Multi-format Data Support**: Handles structured JSON knowledge bases and flat document lists
- **ROS2 Service Interface**: Clean service-based architecture for integration with other ROS2 nodes

# 📄 Documentation
The main documentation for this deliverable is found in the relevant CSSR4Africa deliverables that provide more details.

# 🛠️ Installation 

## Prerequisites
- **ROS2 Humble** or newer
- **Python 3.8** or compatible version
- **ROS 2 installation** with `rclpy` support
- **Internet connection** for LLM API access (unless using local LLM)

## Package Installation

1. **Clone and Build the Workspace**
```bash
# Clone the repository (if not already done)
cd ~/ros2_ws/src
git clone <repository-url>

# Build the workspace
cd ~/ros2_ws
colcon build --packages-select conversation_manager
source install/setup.bash
```

2. **Install Python Dependencies**
```bash
# Install required Python packages
pip install numpy==1.24.4
pip install sentence-transformers chromadb openai pyyaml
```

# 🔧 Configuration Parameters
The configuration is managed via `config/rag_system_configuration.yaml`. The configuration file must be present for the node to start.

| Parameter                   | Description                                                      | Range/Values            | Default Value |
|-----------------------------|------------------------------------------------------------------|-------------------------|---------------|
| `llm.base_url`              | LLM API endpoint URL                                             | String (URL)            | `https://api.deepseek.com/v1` |
| `llm.api_key`               | API key for LLM service                                          | String                  | (from LLM_API_KEY env var)|
| `llm.model`                 | LLM model name                                                   | String                  | `deepseek-chat`|
| `embedding.model`           | Sentence transformer model for embeddings                        | String                  | `all-MiniLM-L6-v2`|
| `search.similarity_threshold` | Similarity threshold for document retrieval                    | `[0.0 - 1.0]`           | `0.15`        |
| `search.top_k`              | Number of documents to retrieve for context                      | Positive integer        | `5`           |
| `data.default_path`         | Path to JSON data file for knowledge base (relative to package) | String (path)           | `./data/upanzi_data.json` |
| `debug.verbose`             | Enable verbose logging                                           | Boolean                 | `true`        |

> **Note:**  
> - `llm.api_key` must be provided via the `LLM_API_KEY` environment variable
> - ChromaDB storage is automatically configured in the package data folder
> - The configuration file is required for node startup

## Example Configuration File (`config/rag_system_configuration.yaml`)
```yaml
# LLM Settings
llm:
  base_url: https://api.deepseek.com/v1
  model: deepseek-chat

# Embedding Model
embedding:
  model: all-MiniLM-L6-v2

# Search Settings
search:
  similarity_threshold: 0.15
  top_k: 5

# Data Settings
data:
  default_path: ./data/upanzi_data.json
  
# Debug Settings
debug:
  verbose: true
```

# 🚀 Running the Node

## Launch All Components
The RAG node can be started directly:

```bash
# Source the workspace
source ~/ros2_ws/install/setup.bash

# Run the RAG node
ros2 run conversation_manager conversation_manager
```

## Manual Node Execution
You can run the node with custom parameters:

```bash
# Run with custom collection name
ros2 run conversation_manager conversation_manager --ros-args -p collection_name:="custom_knowledge" -p verbose:=true
```

# 🖥️ Output
The node provides ROS2 services for knowledge base management and querying.

## Service Structure

The conversation manager provides a single service for querying the knowledge base:

### `/prompt` Service (`cssr_interfaces/srv/ConversationManagerPrompt`)
Query the knowledge base with a question.

**Request Fields:**
- `prompt` (string): The question to ask

**Response Fields:**
- `response` (string): The generated answer

## Knowledge Base Initialization
The knowledge base is automatically initialized at node startup using the configuration file (`config/rag_system_configuration.yaml`). The `data.default_path` parameter in the configuration specifies the JSON data file to load. The collection name is configured via the `collection_name` ROS parameter (default: `'upanzi_knowledge'`).

If the collection doesn't exist, it will be created and populated automatically from the specified data file. If it already exists, the existing collection will be reused.

## Example Usage

1. **Query the Knowledge Base**
```bash
ros2 service call /prompt cssr_interfaces/srv/ConversationManagerPrompt \
  "{prompt: 'What is the Upanzi Network?'}"
```

2. **More Example Queries**
```bash
# Ask about specific projects
ros2 service call /prompt cssr_interfaces/srv/ConversationManagerPrompt \
  "{prompt: 'What projects are focused on cybersecurity?'}"

# Ask about facilities
ros2 service call /prompt cssr_interfaces/srv/ConversationManagerPrompt \
  "{prompt: 'Tell me about the Digital Experience Center.'}"

# Ask about research areas
ros2 service call /prompt cssr_interfaces/srv/ConversationManagerPrompt \
  "{prompt: 'What are the main thrust areas of research?'}"
```

## Verification
To verify the node is working correctly:

```bash
# Check node status
ros2 node list

# Check available services
ros2 service list

# Test the service with a simple query
ros2 service call /prompt cssr_interfaces/srv/ConversationManagerPrompt "{prompt: 'Hello, are you working?'}"
```

# 🏗️ Architecture
The RAG system consists of three main components:

1. **Knowledge Base**: Structured JSON data about Upanzi Network stored in `data/upanzi_data.json`
2. **Vector Database**: ChromaDB with persistent local storage for document embeddings
3. **RAG Node**:
   - Receives queries via ROS2 services
   - Performs semantic search on the vector database
   - Retrieves relevant context documents
   - Generates responses using LLM with retrieved context
   - Maintains conversation history for context-aware responses

## Data Flow
1. User query → `/prompt` service
2. Query embedding → ChromaDB similarity search
3. Retrieved documents + query → LLM prompt
4. LLM response → Service response

## Knowledge Base Format
The system uses a structured JSON format with the following sections:
- `lab_info`: General information about Upanzi Network
- `goals`: Objectives and mission  
- `impact`: Outcomes and achievements
- `facilities`: Physical spaces and labs
- `thrust_areas`: Research focus areas (Cybersecurity, DPG/DPI, Data, etc.)
- `projects`: Detailed project descriptions with metadata

# 💡 Support

For issues or questions:
- Create an issue on GitHub
- Contact: <a href="mailto:yohatad123@gmail.com">yohatad123@gmail.com</a>, <a href="mailto:mahadanso79@gmail.com">mahadanso79@gmail.com</a><br>
- Visit: <a href="http://www.cssr4africa.org">www.cssr4africa.org</a>

# 📜 License
Copyright (C) 2023 Upanzi Network

2026-03-04
