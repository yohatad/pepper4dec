"""
Simplified RAG Implementation for Pepper Robot Lab Assistant

Environment Variables:
    LLM_API_KEY: API key for LLM service (MUST be exported as environment variable)
    
Configuration File (config/rag_system_configuration.yaml):
    All other settings (LLM base URL, model, ChromaDB path, embedding model, etc.)
"""

import os
import json
import threading
import openai
import chromadb
import yaml
from chromadb.utils import embedding_functions
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from ament_index_python.packages import get_package_share_directory
from pathlib import Path


# =============================================================================
# Configuration
# =============================================================================

# Package path for ChromaDB storage (always in package data folder)
PACKAGE_PATH = Path(get_package_share_directory('language_model'))

# Default values as module constants
DEFAULT_LLM_BASE_URL = "http://localhost:8080/v1"
DEFAULT_LLM_API_KEY = "sk-no-key-required"
DEFAULT_LLM_MODEL = "HuggingFaceTB/SmolLM3-3B"
DEFAULT_CHROMA_PATH = str(PACKAGE_PATH / 'data')
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
DEFAULT_SIMILARITY_THRESHOLD = 0.15
DEFAULT_TOP_K = 5
DEFAULT_MAX_HISTORY_TURNS = 5
DEFAULT_CONTEXT_TURNS = 3
DEFAULT_MAX_RESPONSE_SENTENCES = 3
DEFAULT_DATA_PATH = "./data/upanzi_data.json"
DEFAULT_VERBOSE = False


def log_verbose(message: str) -> None:
    """Log message only if verbose mode is enabled"""
    config = get_config()
    if config.verbose:
        print(f"[VERBOSE] {message}")


@dataclass
class RAGConfig:
    """
    Configuration for RAG system.
    
    LLM_API_KEY must be exported as environment variable.
    All other values are loaded from configuration file with sensible defaults.
    """
    
    # LLM settings (only LLM_API_KEY comes from environment variable)
    llm_base_url: str = DEFAULT_LLM_BASE_URL
    llm_api_key: str = field(
        default_factory=lambda: os.getenv("LLM_API_KEY", DEFAULT_LLM_API_KEY)
    )
    llm_model: str = DEFAULT_LLM_MODEL
    
    # ChromaDB settings (embedded mode - no server needed)
    chroma_path: str = DEFAULT_CHROMA_PATH
    
    # Embedding settings
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    
    # Search settings
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD
    default_top_k: int = DEFAULT_TOP_K
    
    # Conversation settings
    max_history_turns: int = DEFAULT_MAX_HISTORY_TURNS
    context_turns: int = DEFAULT_CONTEXT_TURNS
    
    # Response settings
    max_response_sentences: int = DEFAULT_MAX_RESPONSE_SENTENCES
    
    # Data settings
    data_default_path: str = DEFAULT_DATA_PATH
    
    # Debug settings
    verbose: bool = DEFAULT_VERBOSE
    
    def validate(self) -> Tuple[bool, List[str]]:
        """
        Validate configuration values.
        
        Returns:
            Tuple of (is_valid, list of error messages)
        """
        errors = []
        
        if not self.llm_base_url:
            errors.append("llm_base_url cannot be empty")
        
        if not self.llm_api_key or self.llm_api_key == DEFAULT_LLM_API_KEY:
            errors.append("LLM_API_KEY must be exported as environment variable")
        
        if not self.llm_model:
            errors.append("llm_model cannot be empty")
        
        if not 0.0 <= self.similarity_threshold <= 1.0:
            errors.append(f"similarity_threshold must be between 0 and 1, got {self.similarity_threshold}")
        
        if self.default_top_k < 1:
            errors.append(f"default_top_k must be at least 1, got {self.default_top_k}")
        
        if self.max_history_turns < 0:
            errors.append(f"max_history_turns must be non-negative, got {self.max_history_turns}")
        
        if self.context_turns < 0:
            errors.append(f"context_turns must be non-negative, got {self.context_turns}")
        
        if self.context_turns > self.max_history_turns:
            errors.append(f"context_turns ({self.context_turns}) should not exceed max_history_turns ({self.max_history_turns})")
        
        return len(errors) == 0, errors

class RAGError(Exception):
    """Custom exception for RAG-related errors"""
    pass

class ConfigError(RAGError):
    """Exception for configuration-related errors"""
    pass

# =============================================================================
# Global Clients (Thread-Safe)
# =============================================================================

global_config: Optional[RAGConfig] = None
openai_client_instance: Optional[openai.OpenAI] = None
chroma_client_instance: Optional[chromadb.PersistentClient] = None
embedding_function_instance = None
config_lock = threading.Lock()
client_lock = threading.Lock()


def get_config() -> RAGConfig:
    """Get or create default configuration"""
    global global_config
    with config_lock:
        if global_config is None:
            global_config = RAGConfig()
        return global_config


def set_config(config: RAGConfig) -> None:
    """
    Set custom configuration.
    
    Args:
        config: RAGConfig instance to use
        
    Raises:
        ConfigError: If configuration validation fails
    """
    global global_config, openai_client_instance, chroma_client_instance, embedding_function_instance
    
    is_valid, errors = config.validate()
    if not is_valid:
        raise ConfigError(f"Invalid configuration: {'; '.join(errors)}")
    
    with config_lock:
        global_config = config
    
    with client_lock:
        openai_client_instance = None
        chroma_client_instance = None
        embedding_function_instance = None
    
    log_verbose(f"Configuration updated: llm_base_url={config.llm_base_url}, llm_model={config.llm_model}")

def get_openai_client() -> openai.OpenAI:
    """Get or create OpenAI client (thread-safe)"""
    global openai_client_instance
    
    if openai_client_instance is None:
        with client_lock:
            if openai_client_instance is None:
                config = get_config()
                log_verbose(f"Creating OpenAI client with base_url: {config.llm_base_url}")
                openai_client_instance = openai.OpenAI(
                    base_url=config.llm_base_url,
                    api_key=config.llm_api_key,
                    timeout=30.0
                )
    return openai_client_instance

def get_chroma_client() -> chromadb.PersistentClient:
    """Get or create ChromaDB client (embedded mode - no server needed, thread-safe)"""
    global chroma_client_instance
    
    if chroma_client_instance is None:
        with client_lock:
            if chroma_client_instance is None:
                config = get_config()
                try:
                    log_verbose(f"Creating ChromaDB client with path: {config.chroma_path}")
                    chroma_client_instance = chromadb.PersistentClient(path=config.chroma_path)
                except Exception as e:
                    raise RAGError(f"Failed to initialize ChromaDB: {e}")
    return chroma_client_instance

def get_embedding_function():
    """Get or create embedding function (thread-safe)"""
    global embedding_function_instance
    
    if embedding_function_instance is None:
        with client_lock:
            if embedding_function_instance is None:
                config = get_config()
                log_verbose(f"Creating embedding function with model: {config.embedding_model}")
                embedding_function_instance = embedding_functions.SentenceTransformerEmbeddingFunction(
                    model_name=config.embedding_model
                )
    return embedding_function_instance

# =============================================================================
# Data Loading
# =============================================================================

def load_json_data(file_path: str) -> List[Dict]:
    """
    Load and parse JSON knowledge base file.
    
    Args:
        file_path: Path to JSON file
        
    Returns:
        List of document dictionaries
        
    Raises:
        RAGError: If file not found or invalid JSON
    """
    try:
        log_verbose(f"Loading JSON data from: {file_path}")
        with open(file_path, 'r', encoding='utf-8') as file:
            json_data = json.load(file)

        if isinstance(json_data, dict):
            result = parse_upanzi_format(json_data)
        elif isinstance(json_data, list):
            result = json_data
        else:
            raise RAGError(f"Unexpected JSON type: {type(json_data).__name__}")

        log_verbose(f"Loaded {len(result)} documents from {file_path}")
        return result

    except json.JSONDecodeError as e:
        raise RAGError(f"Invalid JSON in {file_path}: {e}")
    except FileNotFoundError:
        raise RAGError(f"File not found: {file_path}")

def parse_upanzi_format(json_data: Dict) -> List[Dict]:
    """Parse Upanzi lab JSON format with 'text' fields"""
    documents = []
    
    # Lab info
    if 'lab_info' in json_data:
        lab = json_data['lab_info']
        documents.append({
            'doc_id': lab.get('id', 'lab_info'),
            'title': lab.get('name', 'Upanzi Network'),
            'text': lab.get('text', ''),
            'keywords': lab.get('keywords', []),
            'category': 'lab_info'
        })
    
    # Goals
    if 'goals' in json_data:
        goals = json_data['goals']
        documents.append({
            'doc_id': goals.get('id', 'goals'),
            'title': 'Upanzi Network Goals',
            'text': goals.get('text', ''),
            'keywords': goals.get('keywords', []),
            'category': 'goals'
        })
    
    # Impact
    if 'impact' in json_data:
        impact = json_data['impact']
        documents.append({
            'doc_id': impact.get('id', 'impact'),
            'title': 'Upanzi Network Impact',
            'text': impact.get('text', ''),
            'keywords': impact.get('keywords', []),
            'category': 'impact'
        })
    
    # Facilities
    if 'facilities' in json_data:
        for facility in json_data['facilities']:
            documents.append({
                'doc_id': facility.get('id', 'facility'),
                'title': facility.get('name', 'Facility'),
                'text': facility.get('text', ''),
                'keywords': facility.get('keywords', []) + facility.get('focus_areas', []),
                'category': 'facility'
            })
    
    # Thrust areas
    if 'thrust_areas' in json_data:
        for thrust in json_data['thrust_areas']:
            documents.append({
                'doc_id': f"thrust_{thrust.get('id', '')}",
                'title': thrust.get('name', ''),
                'text': thrust.get('text', ''),
                'keywords': thrust.get('keywords', []),
                'category': 'thrust_area'
            })
    
    # Projects
    if 'projects' in json_data:
        for project in json_data['projects']:
            keywords = project.get('keywords', []).copy()
            if project.get('thrust_area'):
                keywords.append(project['thrust_area'])
            if project.get('status'):
                keywords.append(project['status'])
            
            documents.append({
                'doc_id': f"project_{project.get('id', '')}",
                'title': project.get('name', ''),
                'text': project.get('text', ''),
                'keywords': keywords,
                'category': 'project',
                'thrust_area': project.get('thrust_area', ''),
                'status': project.get('status', '')
            })
    
    log_verbose(f"Parsed {len(documents)} documents from Upanzi format")
    return documents

# =============================================================================
# Collection Management
# =============================================================================
def create_collection(name: str, description: str = "") -> chromadb.Collection:
    """
    Create or get ChromaDB collection.
    
    Args:
        name: Collection name
        description: Optional description
        
    Returns:
        ChromaDB Collection object
    """
    existing = get_collection(name)
    if existing:
        log_verbose(f"Using existing collection: {name}")
        return existing
    
    client = get_chroma_client()
    ef = get_embedding_function()
    
    log_verbose(f"Creating new collection: {name}")
    collection = client.create_collection(
        name=name,
        metadata={'description': description} if description else None,
        embedding_function=ef
    )
    log_verbose(f"Created collection: {name}")
    return collection


def get_collection(name: str) -> Optional[chromadb.Collection]:
    """Get existing collection"""
    try:
        client = get_chroma_client()
        ef = get_embedding_function()
        collection = client.get_collection(name=name, embedding_function=ef)
        log_verbose(f"Retrieved collection: {name}")
        return collection
    except Exception:
        log_verbose(f"Collection not found: {name}")
        return None

def populate_collection(collection: chromadb.Collection, documents: List[Dict]) -> int:
    """
    Add documents to collection.
    
    Args:
        collection: ChromaDB collection
        documents: List of document dictionaries
        
    Returns:
        Number of documents added
    """
    docs = []
    ids = []
    metadatas = []
    
    for doc in documents:
        text = doc.get('text', '')
        if not text:
            continue
        
        # Build searchable content
        title = doc.get('title', '')
        keywords = doc.get('keywords', [])
        
        content = f"Title: {title}\n" if title else ""
        if keywords:
            content += f"Keywords: {', '.join(keywords)}\n"
        content += text
        
        docs.append(content)
        ids.append(doc.get('doc_id', str(len(ids))))
        metadatas.append({
            'title': title,
            'category': doc.get('category', ''),
            'keywords': ' '.join(keywords)
        })
    
    if docs:
        log_verbose(f"Adding {len(docs)} documents to collection")
        collection.add(documents=docs, ids=ids, metadatas=metadatas)
    else:
        log_verbose("No documents to add to collection")
    
    log_verbose(f"Added {len(docs)} documents to collection")
    return len(docs)

def setup_collection(name: str, json_path: str, description: str = "") -> chromadb.Collection:
    """
    Convenience: Create collection and load data.
    
    Args:
        name: Collection name
        json_path: Path to JSON data file
        description: Optional description
        
    Returns:
        Populated ChromaDB collection
        
    Raises:
        RAGError: If data loading fails
    """
    log_verbose(f"Setting up collection '{name}' from {json_path}")
    documents = load_json_data(json_path)
    collection = create_collection(name, description)
    
    if collection.count() == 0:
        populate_collection(collection, documents)
    else:
        log_verbose(f"Collection '{name}' already has {collection.count()} documents")
    
    log_verbose(f"Collection '{name}' ready with {collection.count()} documents")
    return collection

# =============================================================================
# Search
# =============================================================================

def search(
    collection: chromadb.Collection,
    query: str,
    top_k: int = None,
    category_filter: str = None
) -> List[Dict]:
    """
    Search for relevant documents.
    
    Args:
        collection: ChromaDB collection
        query: Search query
        top_k: Number of results (default from config)
        category_filter: Optional category to filter by
        
    Returns:
        List of results with doc_id, title, content, score
    """
    if not query or not query.strip():
        log_verbose("Empty query, returning empty results")
        return []
    
    config = get_config()
    top_k = top_k or config.default_top_k
    
    log_verbose(f"Searching for query: '{query}' with top_k={top_k}, category_filter={category_filter}")
    
    where_filter = None
    if category_filter:
        where_filter = {"category": category_filter}
    
    try:
        results = collection.query(
            query_texts=[query],
            n_results=top_k,
            where=where_filter
        )
        
        if not results or not results['ids'] or not results['ids'][0]:
            log_verbose("No search results found")
            return []
        
        formatted = []
        for i in range(len(results['ids'][0])):
            score = 1 - results['distances'][0][i]
            
            if score < config.similarity_threshold:
                log_verbose(f"Result {results['ids'][0][i]} below threshold ({score} < {config.similarity_threshold})")
                continue
            
            formatted.append({
                'doc_id': results['ids'][0][i],
                'title': results['metadatas'][0][i].get('title', ''),
                'content': results['documents'][0][i],
                'score': score
            })
        
        log_verbose(f"Search returned {len(formatted)} results above threshold")
        return formatted
        
    except Exception as e:
        log_verbose(f"Search failed with error: {e}")
        raise RAGError(f"Search failed: {e}")

# =============================================================================
# Response Generation
# =============================================================================

SYSTEM_PROMPT = """You are Pepper, a friendly humanoid robot lab assistant at the Upanzi Network, 
located at Carnegie Mellon University Africa in Rwanda.

Your role is to help visitors learn about the lab's research on digital public infrastructure, 
cybersecurity, and technology for Africa.

Guidelines:
- Be friendly, helpful, and concise (2-3 sentences unless more detail is needed)
- Use the provided context to answer accurately
- If you don't know something, say so honestly
- Never make up information
- Watch for misspellings of project names like 'Upanzi', 'picoCTF', 'MOSIP', etc."""


def generate_response(
    query: str,
    search_results: List[Dict],
    conversation_history: List[Dict] = None
) -> str:
    """
    Generate response using LLM with retrieved context.
    
    Args:
        query: User's question
        search_results: Retrieved documents from search
        conversation_history: Optional previous conversation turns
        
    Returns:
        Generated response string
        
    Raises:
        RAGError: If LLM call fails
    """
    config = get_config()
    log_verbose(f"Generating response for query: '{query}'")
    
    # Build context from search results
    context = ""
    if search_results:
        context_parts = [f"[{r['title']}]\n{r['content']}" for r in search_results[:3]]
        context = "\n\n".join(context_parts)
        log_verbose(f"Using {len(search_results)} search results as context")
    
    # Build messages
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    
    # Add conversation history if provided (use configured context_turns)
    if conversation_history:
        history_to_use = conversation_history[-config.context_turns:]
        log_verbose(f"Using {len(history_to_use)} previous conversation turns")
        for turn in history_to_use:
            messages.append({"role": "user", "content": turn.get("query", "")})
            messages.append({"role": "assistant", "content": turn.get("response", "")})
    
    # Add current query with context
    user_content = f'Question: "{query}"'
    if context:
        user_content += f"\n\nRelevant information:\n{context}"
    
    messages.append({"role": "user", "content": user_content})
    
    try:
        client = get_openai_client()
        log_verbose(f"Sending request to LLM (model: {config.llm_model})")
        response = client.chat.completions.create(
            model=config.llm_model,
            messages=messages
        )
        
        if response.choices:
            answer = response.choices[0].message.content.strip()
            
            # Clean thinking tags if present
            if "</think>" in answer:
                answer = answer.split("</think>")[-1].strip()
            
            log_verbose(f"LLM response received: {answer[:100]}...")
            return answer
        
        log_verbose("LLM returned no choices")
        return "I'm sorry, I couldn't generate a response. Please try again."
        
    except Exception as e:
        log_verbose(f"LLM request failed: {e}")
        raise RAGError(f"LLM request failed: {e}")

# =============================================================================
# Main Query Handler
# =============================================================================

def handle_query(
    collection: chromadb.Collection,
    query: str,
    conversation_history: List[Dict] = None,
    category_filter: str = None,
    top_k: int = None
) -> Dict:
    """
    Handle a user query end-to-end.
    
    Args:
        collection: ChromaDB collection
        query: User's question
        conversation_history: Previous conversation turns (will be truncated to max_history_turns)
        category_filter: Optional category to filter search
        top_k: Number of documents to retrieve
        
    Returns:
        Dict with 'response', 'sources', and 'query'
        
    Raises:
        RAGError: If search or generation fails
    """
    if not query or not query.strip():
        log_verbose("Empty query received")
        return {
            'response': "I didn't catch that. Could you please repeat?",
            'sources': [],
            'query': query
        }
    
    config = get_config()
    log_verbose(f"Handling query: '{query}'")
    
    # Truncate history to configured maximum
    if conversation_history and len(conversation_history) > config.max_history_turns:
        log_verbose(f"Truncating conversation history from {len(conversation_history)} to {config.max_history_turns} turns")
        conversation_history = conversation_history[-config.max_history_turns:]
    
    # Search for relevant documents
    log_verbose("Searching for relevant documents...")
    results = search(collection, query, top_k=top_k, category_filter=category_filter)
    
    # Generate response
    log_verbose("Generating response...")
    response = generate_response(query, results, conversation_history)
    
    log_verbose(f"Query handled successfully, response length: {len(response)}")
    return {
        'response': response,
        'sources': [r['doc_id'] for r in results],
        'query': query
    }

# =============================================================================
# Utility Functions
# =============================================================================
def read_yaml_config(file_path: str) -> Tuple[Dict[str, Any], Optional[str]]:
    """
    Read configuration from YAML file.
    
    Args:
        file_path: Path to YAML config file
        
    Returns:
        Tuple of (config dict, error message or None)
    """
    try:
        log_verbose(f"Reading YAML config from: {file_path}")
        with open(file_path, 'r') as f:
            config = yaml.safe_load(f)
            if config is None:
                return {}, "YAML file is empty"
            if not isinstance(config, dict):
                return {}, f"YAML root must be a dictionary, got {type(config).__name__}"
            return config, None
    except FileNotFoundError:
        return {}, f"Config file not found: {file_path}"
    except yaml.YAMLError as e:
        return {}, f"Invalid YAML syntax: {e}"
    except Exception as e:
        return {}, f"Failed to read config file: {e}"

def safe_float(value: Any, default: float, name: str) -> Tuple[float, Optional[str]]:
    """Safely convert value to float"""
    if value is None:
        return default, None
    try:
        return float(value), None
    except (ValueError, TypeError):
        return default, f"Invalid value for {name}: '{value}' (using default: {default})"

def safe_int(value: Any, default: int, name: str) -> Tuple[int, Optional[str]]:
    """Safely convert value to int"""
    if value is None:
        return default, None
    try:
        return int(value), None
    except (ValueError, TypeError):
        return default, f"Invalid value for {name}: '{value}' (using default: {default})"

def safe_str(value: Any, default: str, name: str) -> Tuple[str, Optional[str]]:
    """Safely convert value to string"""
    if value is None:
        return default, None
    try:
        return str(value), None
    except Exception:
        return default, f"Invalid value for {name}: '{value}' (using default: {default})"

def safe_bool(value: Any, default: bool, name: str) -> Tuple[bool, Optional[str]]:
    """Safely convert value to boolean"""
    if value is None:
        return default, None
    if isinstance(value, bool):
        return value, None
    if isinstance(value, str):
        if value.lower() in ('true', 'yes', '1', 'on'):
            return True, None
        if value.lower() in ('false', 'no', '0', 'off'):
            return False, None
    try:
        return bool(value), None
    except Exception:
        return default, f"Invalid value for {name}: '{value}' (using default: {default})"

def apply_config_file(file_path: str) -> Tuple[bool, List[str]]:
    """
    Load and apply configuration from YAML file.
    
    Args:
        file_path: Path to YAML config file
        
    Returns:
        Tuple of (success, list of warnings/errors)
        
    Example YAML format:
        llm:
          base_url: "https://api.groq.com/openai/v1"
          # api_key should be set as environment variable LLM_API_KEY
          model: "llama-3.1-8b-instant"
        embedding:
          model: "all-MiniLM-L6-v2"
        search:
          similarity_threshold: 0.15
          top_k: 5
        data:
          default_path: "./data/upanzi_data.json"
        debug:
          verbose: true
    """
    messages = []
    
    yaml_config, error = read_yaml_config(file_path)
    if error:
        return False, [error]
    
    log_verbose(f"Successfully read YAML config from {file_path}")
    
    # Extract nested values with defaults
    llm = yaml_config.get('llm', {})
    embedding = yaml_config.get('embedding', {})
    search = yaml_config.get('search', {})
    conversation = yaml_config.get('conversation', {})
    data = yaml_config.get('data', {})
    debug = yaml_config.get('debug', {})
    
    # Parse numeric values safely
    similarity_threshold, warn = safe_float(
        search.get('similarity_threshold'), 
        DEFAULT_SIMILARITY_THRESHOLD, 
        'similarity_threshold'
    )
    if warn:
        messages.append(warn)
    
    top_k, warn = safe_int(search.get('top_k'), DEFAULT_TOP_K, 'top_k')
    if warn:
        messages.append(warn)
    
    max_history_turns, warn = safe_int(
        conversation.get('max_history_turns'), 
        DEFAULT_MAX_HISTORY_TURNS, 
        'max_history_turns'
    )
    if warn:
        messages.append(warn)
    
    context_turns, warn = safe_int(
        conversation.get('context_turns'), 
        DEFAULT_CONTEXT_TURNS, 
        'context_turns'
    )
    if warn:
        messages.append(warn)
    
    # Get data default path
    data_default_path, warn = safe_str(
        data.get('default_path'),
        DEFAULT_DATA_PATH,
        'data.default_path'
    )
    if warn:
        messages.append(warn)
    
    # Parse verbose
    verbose, warn = safe_bool(
        debug.get('verbose'),
        DEFAULT_VERBOSE,
        'debug.verbose'
    )
    if warn:
        messages.append(warn)
    
    try:
        # ChromaDB is always stored in the package data folder
        chroma_path = str(PACKAGE_PATH / 'data')
        log_verbose(f"Using package data folder for ChromaDB: {chroma_path}")
        
        config = RAGConfig(
            llm_base_url=llm.get('base_url', DEFAULT_LLM_BASE_URL),
            # LLM_API_KEY must be exported as environment variable, not from config file
            llm_model=llm.get('model', DEFAULT_LLM_MODEL),
            chroma_path=chroma_path,  # Always use package data folder
            embedding_model=embedding.get('model', DEFAULT_EMBEDDING_MODEL),
            similarity_threshold=similarity_threshold,
            default_top_k=top_k,
            max_history_turns=max_history_turns,
            context_turns=context_turns,
            data_default_path=data_default_path,
            verbose=verbose,
        )
        
        # This will validate and raise ConfigError if invalid
        set_config(config)
        log_verbose(f"Configuration applied successfully: verbose={verbose}")
        return True, messages
        
    except ConfigError as e:
        log_verbose(f"ConfigError: {e}")
        return False, [str(e)] + messages
    except Exception as e:
        log_verbose(f"Exception applying configuration: {e}")
        return False, [f"Failed to apply configuration: {e}"] + messages
