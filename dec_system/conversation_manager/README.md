<div align="center">
<h1> Conversation Manager</h1>
</div>

<div align="center">
  <img src="../upanzi-logo.svg" alt="Upanzi Logo" style="width:70%; height:auto;">
</div>

The **Conversation Manager Package** implements a **Retrieval-Augmented Generation (RAG)** system for the Pepper robot to serve as a lab assistant at the Upanzi Network, Carnegie Mellon University Africa. The system allows Pepper to answer questions about Upanzi Network's research, projects, facilities, and impact areas using a knowledge base built from structured JSON data.

## Key Features
- **ROS2 Native**: Built for ROS2 Humble
- **Retrieval-Augmented Generation**: Combines vector search with large language models for accurate, context-aware responses
- **ChromaDB Integration**: Local vector database for privacy-preserving knowledge storage
- **Configurable LLM Support**: Compatible with any OpenAI-compatible API (DeepSeek, Groq, etc.)
- **Streaming TTS Output**: Publishes answer sentences to `/tts/input` as they arrive from the LLM, enabling Pepper to start speaking before the full response is ready
- **Conversation Memory**: Maintains context from previous interactions (configurable number of turns)
- **Multi-format Data Support**: Handles structured JSON knowledge bases and flat document lists
- **ROS2 Action Interface**: Action-based architecture for integration with other ROS2 nodes and the BehaviorTree controller, with feedback during processing

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
git clone https://github.com/yohatad/pepper4dec.git

# Build the workspace
cd ~/ros2_ws
colcon build --packages-select conversation_manager
source install/setup.bash
```

2. **Install Python Dependencies**
```bash
pip install -r ~/ros2_ws/src/pepper4dec/dec_system/conversation_manager/requirements.txt
```

# 🔧 Configuration Parameters
The configuration is managed via `config/converation_manager_configuration.yaml`. The file must be present for the node to start.

| Parameter                      | Description                                                      | Range/Values     | Default Value                 |
|--------------------------------|------------------------------------------------------------------|------------------|-------------------------------|
| `llm.base_url`                 | LLM API endpoint URL                                             | String (URL)     | `https://api.deepseek.com/v1` |
| `llm.api_key`                  | API key for LLM service                                          | String           | (from `LLM_API_KEY` env var)  |
| `llm.model`                    | LLM model name                                                   | String           | `deepseek-chat`               |
| `embedding.model`              | Sentence transformer model for embeddings                        | String           | `all-MiniLM-L6-v2`           |
| `search.similarity_threshold`  | Similarity threshold for document retrieval                      | `[0.0 – 1.0]`   | `0.15`                        |
| `search.top_k`                 | Number of documents to retrieve for context                      | Positive integer | `15`                           |
| `conversation.max_history_turns` | Number of past turns kept in conversation memory               | Positive integer | `10`                           |
| `data.default_path`            | Path to JSON knowledge base (relative to package share dir)      | String (path)    | `./data/upanzi_data.json`     |
| `debug.verbose`                | Enable verbose logging                                           | Boolean          | `true`                        |

> **Note:**
> - `llm.api_key` must be provided via the `LLM_API_KEY` environment variable.
> - ChromaDB storage is automatically configured in the package data folder.
> - The configuration file is required for node startup.

## Example Configuration File (`config/converation_manager_configuration.yaml`)
```yaml
llm:
  base_url: https://api.deepseek.com/v1
  model: deepseek-chat

embedding:
  model: all-MiniLM-L6-v2

search:
  similarity_threshold: 0.15
  top_k: 5

conversation:
  max_history_turns: 5

data:
  default_path: ./data/upanzi_data.json

debug:
  verbose: true
```

# 🚀 Running the Node

```bash
# Source the workspace
source ~/ros2_ws/install/setup.bash

# Run the node
ros2 run conversation_manager conversation_manager
```

You can override the ChromaDB collection name or enable extra logging at runtime:
```bash
ros2 run conversation_manager conversation_manager \
  --ros-args -p collection_name:="custom_knowledge" -p verbose:=true
```

# 🖥️ ROS Interface

## Action Server

### `/conversation_manager` (`dec_interfaces/action/ConversationManager`)
Receives a natural-language prompt, performs a RAG query, and returns the generated answer.
Sentences are streamed to `/tts/input` while the LLM is generating so that the TTS node can start speaking before the full response is ready.

**Goal Fields:**
| Field    | Type   | Description                       |
|----------|--------|-----------------------------------|
| `prompt` | string | Natural-language question to ask  |

**Feedback Fields:**
| Field    | Type   | Values                            |
|----------|--------|-----------------------------------|
| `status` | string | `"searching"` \| `"generating"`   |

**Result Fields:**
| Field      | Type   | Description                                         |
|------------|--------|-----------------------------------------------------|
| `response` | string | Full generated answer (plain text)                  |
| `success`  | bool   | `true` if the query was processed without error     |

## Publisher

### `/tts/input` (`std_msgs/String`)
Individual answer sentences published one at a time as they arrive from the LLM stream.
The `text_to_speech` node subscribes here to begin playback before the action completes.

## BehaviorTree Integration

The node is called from the `behaviorController` via the `ConversationManager` BT node, which wraps this action server. The generated `response` is written to the blackboard and passed directly to the `TTS` BT node (wrapping `dec_interfaces/action/TTS`) for playback through the configured Kokoro or ElevenLabs backend.

Typical BT sequence:
```
SpeechRecognition → ConversationManager → TTS
```

## Knowledge Base Initialization
The knowledge base is automatically initialized at node startup using `config/converation_manager_configuration.yaml`. The `data.default_path` parameter specifies the JSON data file to load. The collection name is set via the `collection_name` ROS parameter (default: `'upanzi_knowledge'`).

If the collection does not exist it will be created and populated from the data file automatically. If it already exists the existing collection is reused.

# 🏗️ Architecture

The RAG system has three main components:

1. **Knowledge Base**: Structured JSON data stored in `data/upanzi_data.json`
2. **Vector Database**: ChromaDB with persistent local storage for document embeddings
3. **Conversation Manager Node**:
   - Receives prompts via the `/conversation_manager` action server
   - Performs semantic search on the vector database (`searching` feedback)
   - Streams LLM-generated sentences to `/tts/input` as they arrive (`generating` feedback)
   - Returns the full answer text as the action result for any other consumer
   - Maintains per-session conversation history for context-aware responses

## Data Flow

```
User utterance
      │
      ▼
/conversation_manager action goal  (prompt)
      │
      ├─► Stage 1: ChromaDB similarity search  →  feedback: "searching"
      │
      ├─► Stage 2: Streaming LLM generation    →  feedback: "generating"
      │         │
      │         └─► sentences ──► /tts/input  (TTS starts speaking immediately)
      │
      └─► Action result: full response text    →  TTSNode waits for playback to finish
```

## Knowledge Base JSON Format
```
upanzi_data.json
├── lab_info      – General information about Upanzi Network
├── goals         – Objectives and mission
├── impact        – Outcomes and achievements
├── facilities    – Physical spaces and labs
├── thrust_areas  – Research focus areas (Cybersecurity, DPG/DPI, Data, …)
└── projects      – Detailed project descriptions with metadata
```

# 🧪 Testing

```bash
# Check the node is running
ros2 node list

# Verify the action server is available
ros2 action list

# Send a test query
ros2 action send_goal /conversation_manager dec_interfaces/action/ConversationManager \
  "{prompt: 'What is the Upanzi Network?'}"

# More example queries
ros2 action send_goal /conversation_manager dec_interfaces/action/ConversationManager \
  "{prompt: 'What projects are focused on cybersecurity?'}"

ros2 action send_goal /conversation_manager dec_interfaces/action/ConversationManager \
  "{prompt: 'Tell me about the Digital Experience Center.'}"

# Monitor TTS streaming output
ros2 topic echo /tts/input
```

# 💡 Support

For issues or questions:
- Create an issue on GitHub
- Contact: <a href="mailto:yohatad123@gmail.com">yohatad123@gmail.com</a>, <a href="mailto:mahadanso79@gmail.com">mahadanso79@gmail.com</a>

# 📜 License
Copyright (C) 2026 Upanzi Network  
Licensed under the BSD-3-Clause License. See individual package licenses for details.
