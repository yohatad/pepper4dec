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
- **ROS2 Action Interface**: Action-based architecture for integration with other ROS2 nodes, with feedback during processing

# 📄 Documentation
The main documentation for this deliverable is found in the relevant DEC4Africa deliverables that provide more details.

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
# Install package requirements
pip install -r ~/ros2_ws/src/dec_system/conversation_manager/requirements.txt
```

# 🔧 Configuration Parameters
The configuration is managed via `config/converation_manager_configuration.yaml`. The configuration file must be present for the node to start.

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

## Example Configuration File (`config/converation_manager_configuration.yaml`)
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
The node provides a ROS2 action for querying the knowledge base.

## Action Interface

### `/prompt` Action (`dec_interfaces/action/ConversationManager`)
Query the knowledge base with a question. Publishes feedback during processing and returns the final answer.

**Goal Fields:**
- `prompt` (string): The question to ask

**Feedback Fields:**
- `status` (string): Current processing stage — `searching` or `generating`

**Result Fields:**
- `response` (string): The generated answer (plain text, extracted from LLM JSON output)
- `success` (bool): Whether the query was processed successfully

## Knowledge Base Initialization
The knowledge base is automatically initialized at node startup using the configuration file (`config/converation_manager_configuration.yaml`). The `data.default_path` parameter in the configuration specifies the JSON data file to load. The collection name is configured via the `collection_name` ROS parameter (default: `'upanzi_knowledge'`).

If the collection doesn't exist, it will be created and populated automatically from the specified data file. If it already exists, the existing collection will be reused.

## Example Usage

1. **Query the Knowledge Base**
```bash
ros2 action send_goal /prompt dec_interfaces/action/ConversationManager \
  "{prompt: 'What is the Upanzi Network?'}"
```

2. **More Example Queries**
```bash
# Ask about specific projects
ros2 action send_goal /prompt dec_interfaces/action/ConversationManager \
  "{prompt: 'What projects are focused on cybersecurity?'}"

# Ask about facilities
ros2 action send_goal /prompt dec_interfaces/action/ConversationManager \
  "{prompt: 'Tell me about the Digital Experience Center.'}"

# Ask about research areas
ros2 action send_goal /prompt dec_interfaces/action/ConversationManager \
  "{prompt: 'What are the main thrust areas of research?'}"
```

## Verification
To verify the node is working correctly:

```bash
# Check node status
ros2 node list

# Check available actions
ros2 action list

# Test with a simple query
ros2 action send_goal /prompt dec_interfaces/action/ConversationManager "{prompt: 'Hello, are you working?'}"
```

# 🏗️ Architecture
The RAG system consists of three main components:

1. **Knowledge Base**: Structured JSON data about Upanzi Network stored in `data/upanzi_data.json`
2. **Vector Database**: ChromaDB with persistent local storage for document embeddings
3. **RAG Node**:
   - Receives queries via ROS2 actions
   - Performs semantic search on the vector database
   - Retrieves relevant context documents
   - Generates responses using LLM with retrieved context
   - Maintains conversation history for context-aware responses

## Data Flow
1. User query → `/prompt` action goal
2. Feedback: `searching` → ChromaDB similarity search
3. Feedback: `generating` → Retrieved documents + query → LLM prompt
4. LLM JSON response parsed → `answer` field returned as action result

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
<!-- - Visit: <a href="http://www.dec4africa.org">www.dec4africa.org</a> -->

# 📜 License
Copyright (C) 2026 Upanzi Network
Licensed under the BSD-3-Clause License. See individual package licenses for details.
