<div align="center">
<h1> Language Model Package for Pepper Robot Lab Assistant (ROS2) </h1>
</div>

<div align="center">
  <img src="../upanzi-logo.svg" alt="Upanzi Logo" style="width:50%; height:auto;">
</div>

The **Language Model Package** implements a **Retrieval-Augmented Generation (RAG)** system for the Pepper robot to serve as a lab assistant at the Upanzi Network, Carnegie Mellon University Africa. The system allows Pepper to answer questions about Upanzi Network's research, projects, facilities, and impact areas using a knowledge base built from structured JSON data.

## Key Features
- **ROS2 Native**: Built for ROS2 Humble/Humble+
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
The configuration is managed via `config/rag_system_configuration.yaml` or environment variables:

| Parameter                   | Description                                                      | Range/Values            | Default Value |
|-----------------------------|------------------------------------------------------------------|-------------------------|---------------|
| `llm.base_url`              | LLM API endpoint URL                                             | String (URL)            | `https://api.deepseek.com/v1` |
| `llm.api_key`               | API key for LLM service                                          | String                  | (from env var)|
| `llm.model`                 | LLM model name                                                   | String                  | `deepseek-chat`|
| `chroma.path`               | Path for ChromaDB local storage                                  | String (path)           | `./chroma_data`|
| `embedding.model`           | Sentence transformer model for embeddings                        | String                  | `all-MiniLM-L6-v2`|
| `search.similarity_threshold` | Similarity threshold for document retrieval                    | `[0.0 - 1.0]`           | `0.15`        |
| `search.top_k`              | Number of documents to retrieve for context                      | Positive integer        | `5`           |

## Environment Variables (Alternative Configuration)
```bash
# LLM Settings
export LLM_BASE_URL="https://api.deepseek.com/v1"
export LLM_API_KEY="your-api-key-here"
export LLM_MODEL="deepseek-chat"

# ChromaDB Settings
export CHROMA_PATH="./chroma_data"

# Embedding Settings
export EMBEDDING_MODEL="all-MiniLM-L6-v2"

# Debug Settings
export RAG_VERBOSE="true"  # Optional: for debug logging
```

> **Note:**  
> The YAML configuration file takes precedence over environment variables.

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

### 1. `/create_collection` Service (`cssr_interface/srv/LanguageModelCreateCollection`)
Creates and populates a new knowledge base collection from a JSON file.

**Request Fields:**
- `name` (string): Collection name (e.g., "upanzi_knowledge")
- `datafile_path` (string): Path to JSON data file
- `description` (string): Optional description

**Response Fields:**
- `success` (int32): 1 for success, 0 for failure
- `message` (string): Status message

### 2. `/rag_prompt` Service (`cssr_interface/srv/LanguageModelPrompt`)
Query the knowledge base with a question.

**Request Fields:**
- `prompt` (string): The question to ask

**Response Fields:**
- `response` (string): The generated answer

## Example Usage

1. **Create a Knowledge Base Collection**
```bash
ros2 service call /create_collection cssr_interface/srv/LanguageModelCreateCollection \
  "{name: 'upanzi_knowledge', datafile_path: '$(pwd)/ros2_ws/src/cssr4africa/cssr_system/language_model/data/upanzi_data.json', description: 'Upanzi Network knowledge base'}"
```

2. **Query the Knowledge Base**
```bash
ros2 service call /rag_prompt cssr_interface/srv/LanguageModelPrompt \
  "{prompt: 'What is the Upanzi Network?'}"
```

3. **More Example Queries**
```bash
# Ask about specific projects
ros2 service call /rag_prompt cssr_interface/srv/LanguageModelPrompt \
  "{prompt: 'What projects are focused on cybersecurity?'}"

# Ask about facilities
ros2 service call /rag_prompt cssr_interface/srv/LanguageModelPrompt \
  "{prompt: 'Tell me about the Digital Experience Center.'}"

# Ask about research areas
ros2 service call /rag_prompt cssr_interface/srv/LanguageModelPrompt \
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
ros2 service call /rag_prompt cssr_interface/srv/LanguageModelPrompt "{prompt: 'Hello, are you working?'}"
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
1. User query → `/rag_prompt` service
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
- Contact: <a href="mailto:yohanneh@andrew.cmu.edu">yohanneh@andrew.cmu.edu</a>, <a href="mailto:mahadanso79@gmail.com">mahadanso79@gmail.com</a><br>
- Visit: <a href="http://www.cssr4africa.org">www.cssr4africa.org</a>

# 📜 License
Copyright (C) 2023 CSSR4Africa Consortium  
Funded by African Engineering and Technology Network (Afretec)  
Inclusive Digital Transformation Research Grant Programme

2026-01-11 (Updated for ROS2)
