""" conversation_manager_implementation.py

Core RAG (Retrieval-Augmented Generation) implementation for the conversation manager.
Provides configuration loading, ChromaDB collection management, knowledge-base
ingestion, vector search, and (streaming) LLM response generation used by
ConversationManagerNode.

Configuration is held in a module-level ConversationManagerConfig dataclass, loaded
from config/converation_manager_configuration.yaml via apply_config_file().
The LLM_API_KEY must be exported as an environment variable; all other settings
(llm, embedding, search, conversation, data, debug) come from the YAML file with
sensible defaults. Query handling (handle_query / generate_response_stream)
performs a similarity search against the ChromaDB knowledge-base collection,
builds a prompt with conversation history and retrieved context, and calls the
LLM to produce an answer plus an intent/confidence classification.

Author: Yohannes Tadesse Haile
Affiliation: Carnegie Mellon University Africa
Email: yohatad123@gmail.com
Date: February 28, 2026
Version: v1.0
"""

import os
import re
import json
import openai
import chromadb
import rclpy.logging
from chromadb.utils import embedding_functions
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from ament_index_python.packages import get_package_share_directory
from pathlib import Path
from .conversation_manager_utilities import (
    Colors,
    print_search_results,
    print_conversation_history,
    print_llm_request,
    read_yaml_config,
    safe_float,
    safe_int,
    safe_str,
    safe_bool,
)

logger = rclpy.logging.get_logger('conversation_manager')


# =============================================================================
# Configuration
# =============================================================================

# Package path for ChromaDB storage (always in package data folder)
try:
    PACKAGE_PATH = Path(get_package_share_directory('conversation_manager'))
except Exception:
    # Fallback for environments where ROS2 is not fully sourced (e.g., testing)
    PACKAGE_PATH = Path(__file__).parent.parent

# Default values as module constants
DEFAULT_LLM_BASE_URL = "http://localhost:8080/v1"
DEFAULT_LLM_API_KEY = "sk-no-key-required"
DEFAULT_LLM_MODEL = "HuggingFaceTB/SmolLM3-3B"
DEFAULT_CHROMA_PATH = str(PACKAGE_PATH / 'data')
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
DEFAULT_RETRIEVAL_MODE = "rag"
VALID_RETRIEVAL_MODES = ("rag", "full_context")
DEFAULT_SIMILARITY_THRESHOLD = 0.15
DEFAULT_TOP_K = 10
DEFAULT_MAX_HISTORY_TURNS = 15
DEFAULT_CONTEXT_TURNS = 10
DEFAULT_MAX_RESPONSE_SENTENCES = 3
DEFAULT_DATA_PATH = str(PACKAGE_PATH / 'data' / 'upanzi_data.json')
DEFAULT_SYSTEM_PROMPT_PATH = str(PACKAGE_PATH / 'data' / 'system_prompt.txt')
DEFAULT_VERBOSE = False

def is_verbose() -> bool:
    """Check verbose state without auto-creating a default config."""
    return global_config is not None and global_config.verbose


@dataclass
class ConversationManagerConfig:
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

    # Retrieval settings: "rag" (vector search) or "full_context" (send entire KB every turn)
    retrieval_mode: str = DEFAULT_RETRIEVAL_MODE

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
        
        if not self.llm_api_key:
            errors.append("llm_api_key cannot be empty")
        elif self.llm_api_key == DEFAULT_LLM_API_KEY and not os.getenv("LLM_API_KEY"):
            logger.warning(
                "LLM_API_KEY environment variable is not set; "
                "using placeholder key (acceptable for local servers that require no auth)"
            )
        
        if not self.llm_model:
            errors.append("llm_model cannot be empty")

        if self.retrieval_mode not in VALID_RETRIEVAL_MODES:
            errors.append(
                f"retrieval_mode must be one of {VALID_RETRIEVAL_MODES}, got '{self.retrieval_mode}'"
            )

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
# Global Clients
# =============================================================================

global_config: Optional[ConversationManagerConfig] = None
openai_client_instance: Optional[openai.OpenAI] = None
chroma_client_instance: Optional[chromadb.PersistentClient] = None
embedding_function_instance = None
full_context_cache: Optional[List[Dict]] = None


def get_config() -> ConversationManagerConfig:
    """Get or create default configuration"""
    global global_config
    if global_config is None:
        global_config = ConversationManagerConfig()
    return global_config

def set_config(config: ConversationManagerConfig) -> None:
    """
    Set custom configuration.

    Args:
        config: ConversationManagerConfig instance to use

    Raises:
        ConfigError: If configuration validation fails
    """
    global global_config, openai_client_instance, chroma_client_instance, embedding_function_instance, full_context_cache

    is_valid, errors = config.validate()
    if not is_valid:
        raise ConfigError(f"Invalid configuration: {'; '.join(errors)}")

    global_config = config
    openai_client_instance = None
    chroma_client_instance = None
    embedding_function_instance = None
    full_context_cache = None

    logger.debug(f"Configuration updated: llm_base_url={config.llm_base_url}, llm_model={config.llm_model}")

def get_openai_client() -> openai.OpenAI:
    """Get or create OpenAI client"""
    global openai_client_instance

    if openai_client_instance is None:
        config = get_config()
        logger.debug(f"Creating OpenAI client with base_url: {config.llm_base_url}")
        openai_client_instance = openai.OpenAI(
            base_url=config.llm_base_url,
            api_key=config.llm_api_key,
            timeout=30.0
        )
    return openai_client_instance

def get_chroma_client() -> chromadb.PersistentClient:
    """Get or create ChromaDB client (embedded mode - no server needed)"""
    global chroma_client_instance

    if chroma_client_instance is None:
        config = get_config()
        try:
            logger.debug(f"Creating ChromaDB client with path: {config.chroma_path}")
            chroma_client_instance = chromadb.PersistentClient(path=config.chroma_path)
        except Exception as e:
            raise RAGError(f"Failed to initialize ChromaDB: {e}")
    return chroma_client_instance

def get_embedding_function():
    """Get or create embedding function"""
    global embedding_function_instance

    if embedding_function_instance is None:
        config = get_config()
        logger.debug(f"Creating embedding function with model: {config.embedding_model}")
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
        logger.debug(f"Loading JSON data from: {file_path}")
        with open(file_path, 'r', encoding='utf-8') as file:
            json_data = json.load(file)

        if isinstance(json_data, dict):
            result = parse_upanzi_format(json_data)
        elif isinstance(json_data, list):
            result = json_data
        else:
            raise RAGError(f"Unexpected JSON type: {type(json_data).__name__}")

        logger.debug(f"Loaded {len(result)} documents from {file_path}")
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
    
    logger.debug(f"Parsed {len(documents)} documents from Upanzi format")
    return documents

def _build_document_content(doc: Dict) -> str:
    """Build searchable/displayable content from a document's title, keywords, and text."""
    title = doc.get('title', '')
    keywords = doc.get('keywords', [])

    content = f"Title: {title}\n" if title else ""
    if keywords:
        content += f"Keywords: {', '.join(keywords)}\n"
    content += doc.get('text', '')
    return content

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
        logger.debug(f"Using existing collection: {name}")
        return existing
    
    client = get_chroma_client()
    ef = get_embedding_function()
    
    logger.debug(f"Creating new collection: {name}")
    metadata = {'hnsw:space': 'cosine'}
    if description:
        metadata['description'] = description
    collection = client.create_collection(
        name=name,
        metadata=metadata,
        embedding_function=ef
    )
    logger.debug(f"Created collection: {name}")
    return collection


def get_collection(name: str) -> Optional[chromadb.Collection]:
    """Get existing collection, or None if it does not exist."""
    try:
        client = get_chroma_client()
        ef = get_embedding_function()
        collection = client.get_collection(name=name, embedding_function=ef)
        logger.debug(f"Retrieved collection: {name}")
        return collection
    except Exception as e:
        if "does not exist" in str(e).lower() or "not found" in str(e).lower():
            logger.debug(f"Collection not found: {name}")
            return None
        raise RAGError(f"Failed to get collection '{name}': {e}")

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
    seen_ids = set()

    for doc in documents:
        text = doc.get('text', '')
        if not text:
            continue

        doc_id = doc.get('doc_id') or str(len(ids))
        if doc_id in seen_ids:
            logger.warning(f"Skipping document with duplicate doc_id: {doc_id}")
            continue
        seen_ids.add(doc_id)

        docs.append(_build_document_content(doc))
        ids.append(doc_id)
        metadatas.append({
            'title': doc.get('title', ''),
            'category': doc.get('category', ''),
            'keywords': ' '.join(doc.get('keywords', []))
        })

    if docs:
        logger.debug(f"Adding {len(docs)} documents to collection")
        collection.add(documents=docs, ids=ids, metadatas=metadatas)
        logger.debug(f"Added {len(docs)} documents to collection")
    else:
        logger.debug("No documents to add to collection")

    return len(docs)

def setup_collection(name: str, json_path: str, description: str = "", force_reload: bool = False) -> chromadb.Collection:
    """
    Convenience: Create collection and load data.

    Args:
        name: Collection name
        json_path: Path to JSON data file
        description: Optional description
        force_reload: If True, delete and recreate the collection so data is
                      refreshed from the JSON file (useful when the source data
                      has changed).

    Returns:
        Populated ChromaDB collection

    Raises:
        RAGError: If data loading fails
    """
    logger.debug(f"Setting up collection '{name}' from {json_path}")
    documents = load_json_data(json_path)

    if force_reload:
        existing = get_collection(name)
        if existing is not None:
            logger.debug(f"force_reload=True: deleting existing collection '{name}'")
            get_chroma_client().delete_collection(name)

    collection = create_collection(name, description)

    if collection.count() == 0:
        populate_collection(collection, documents)
    else:
        logger.debug(f"Collection '{name}' already has {collection.count()} documents")

    logger.debug(f"Collection '{name}' ready with {collection.count()} documents")
    return collection

# =============================================================================
# Full-context retrieval (retrieval_mode == "full_context")
# =============================================================================

def get_full_context_documents(category_filter: str = None) -> List[Dict]:
    """
    Load the entire knowledge base as a list of "search results" (score fixed
    at 1.0), formatted the same way as retrieve_documents() output so callers don't need
    to know which retrieval mode is active. Loaded once and cached; the cache
    is cleared on set_config() (e.g. if data_default_path changes).

    Args:
        category_filter: Optional category to filter by

    Returns:
        List of results with doc_id, title, content, score
    """
    global full_context_cache

    if full_context_cache is None:
        config = get_config()
        logger.debug(f"Loading full knowledge base from: {config.data_default_path}")
        documents = load_json_data(config.data_default_path)
        full_context_cache = [
            {
                'doc_id': doc.get('doc_id') or str(i),
                'title': doc.get('title', ''),
                'content': _build_document_content(doc),
                'score': 1.0,
                'category': doc.get('category', ''),
            }
            for i, doc in enumerate(documents)
            if doc.get('text', '')
        ]
        logger.debug(f"Loaded {len(full_context_cache)} documents for full-context mode")

    if category_filter:
        return [r for r in full_context_cache if r.get('category') == category_filter]
    return list(full_context_cache)

# =============================================================================
# Retrieval
# =============================================================================

def retrieve_documents(
    collection: Optional[chromadb.Collection],
    query: str,
    top_k: int = None,
    category_filter: str = None
) -> List[Dict]:
    """
    Retrieve relevant documents for a query.

    Dispatches on config.retrieval_mode:
      - "rag": vector similarity search against the ChromaDB collection.
      - "full_context": return the entire knowledge base, ignoring `collection`
        and `top_k` (no vector search is performed).

    Args:
        collection: ChromaDB collection (unused in "full_context" mode)
        query: Search query
        top_k: Number of results (default from config, "rag" mode only)
        category_filter: Optional category to filter by

    Returns:
        List of results with doc_id, title, content, score
    """
    if not query or not query.strip():
        logger.debug("Empty query, returning empty results")
        return []

    config = get_config()

    if config.retrieval_mode == "full_context":
        logger.debug("Full-context mode: returning entire knowledge base")
        return get_full_context_documents(category_filter=category_filter)

    top_k = top_k if top_k is not None else config.default_top_k

    logger.debug(f"Searching for query: '{query}' with top_k={top_k}, category_filter={category_filter}")
    
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
            logger.debug("No search results found")
            return []
        
        formatted = []
        for i in range(len(results['ids'][0])):
            score = 1 - results['distances'][0][i]
            
            if score < config.similarity_threshold:
                logger.debug(f"Result {results['ids'][0][i]} below threshold ({score} < {config.similarity_threshold})")
                continue
            
            formatted.append({
                'doc_id': results['ids'][0][i],
                'title': results['metadatas'][0][i].get('title', ''),
                'content': results['documents'][0][i],
                'score': score
            })
        
        logger.debug(f"Search returned {len(formatted)} results above threshold")
        return formatted
        
    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise RAGError(f"Search failed: {e}")

# =============================================================================
# Response Generation
# =============================================================================

system_prompt_cache: Optional[str] = None

def load_system_prompt() -> str:
    """Load system prompt from data/system_prompt.txt, caching after first read."""
    global system_prompt_cache
    if system_prompt_cache is not None:
        return system_prompt_cache
    try:
        with open(DEFAULT_SYSTEM_PROMPT_PATH, 'r', encoding='utf-8') as f:
            system_prompt_cache = f.read().strip()
        logger.debug(f"Loaded system prompt from: {DEFAULT_SYSTEM_PROMPT_PATH}")
    except FileNotFoundError:
        logger.error(f"System prompt file not found: {DEFAULT_SYSTEM_PROMPT_PATH}")
        system_prompt_cache = ""
    return system_prompt_cache


def generate_response(
    query: str,
    search_results: List[Dict],
    conversation_history: List[Dict] = None) -> str:
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
    logger.debug(f"Generating response for query: '{query}'")
    
    # Print search results in verbose mode
    if search_results:
        print_search_results(is_verbose(), search_results)
    
    # Build context from search results
    context = ""
    if search_results:
        context_parts = [f"[{r['title']}]\n{r['content']}" for r in search_results]
        context = "\n\n".join(context_parts)
        logger.debug(f"Using {len(search_results)} search results as context")
    
    # Build messages
    messages = [{"role": "system", "content": load_system_prompt()}]
    
    # Add conversation history if provided (use configured context_turns)
    if conversation_history:
        history_to_use = conversation_history[-config.context_turns:]
        
        # Print conversation history in verbose mode
        print_conversation_history(is_verbose(), conversation_history, config.context_turns)
        
        logger.debug(f"Using {len(history_to_use)} previous conversation turns")
        for turn in history_to_use:
            messages.append({"role": "user", "content": turn.get("query", "")})
            messages.append({"role": "assistant", "content": turn.get("response", "")})
    
    # Add current query with context
    user_content = f'Question: "{query}"'
    if context:
        user_content += f"\n\nRelevant information:\n{context}"
    
    messages.append({"role": "user", "content": user_content})
    
    # Print complete LLM request in verbose mode
    print_llm_request(is_verbose(), messages, config.llm_model)
    
    try:
        client = get_openai_client()
        logger.debug(f"Sending request to LLM (model: {config.llm_model})")
        # ~50 tokens per sentence for the visible answer, plus 512 tokens of
        # headroom so that thinking-capable models (which emit <think>…</think>
        # blocks before the actual reply) are not cut off mid-reasoning.
        max_tokens = config.max_response_sentences * 50 + 512
        response = client.chat.completions.create(
            model=config.llm_model,
            messages=messages,
            max_tokens=max_tokens
        )
        
        if response.choices:
            raw = response.choices[0].message.content.strip()

            # Clean thinking tags if present
            if "</think>" in raw:
                raw = raw.split("</think>")[-1].strip()

            if is_verbose():
                logger.info(f"{Colors.BG_CYAN}{Colors.BLACK}{Colors.BOLD} LLM RESPONSE {Colors.RESET}\n{raw}")

            # Extract the spoken answer from the structured JSON response
            try:
                parsed = json.loads(raw)
                answer = parsed.get("answer", raw)
            except (json.JSONDecodeError, AttributeError):
                answer = raw

            logger.debug(f"LLM response received: {answer[:100]}...")
            return answer
        
        logger.debug("LLM returned no choices")
        return "I'm sorry, I couldn't generate a response. Please try again."
        
    except Exception as e:
        logger.error(f"LLM request failed: {e}")
        raise RAGError(f"LLM request failed: {e}")

# =============================================================================
# Main Query Handler
# =============================================================================

def handle_query(
    collection: chromadb.Collection,
    query: str,
    conversation_history: List[Dict] = None,
    category_filter: str = None,
    top_k: int = None) -> Dict:
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
        logger.debug("Empty query received")
        return {
            'response': "I didn't catch that. Could you please repeat?",
            'sources': [],
            'query': query
        }
    
    config = get_config()
    logger.debug(f"Handling query: '{query}'")
    
    # Truncate history to configured maximum
    if conversation_history and len(conversation_history) > config.max_history_turns:
        logger.debug(f"Truncating conversation history from {len(conversation_history)} to {config.max_history_turns} turns")
        conversation_history = conversation_history[-config.max_history_turns:]
    
    # Search for relevant documents
    logger.debug("Searching for relevant documents...")
    results = retrieve_documents(collection, query, top_k=top_k, category_filter=category_filter)
    
    # Generate response
    logger.debug("Generating response...")
    response = generate_response(query, results, conversation_history)
    
    logger.debug(f"Query handled successfully, response length: {len(response)}")
    return {
        'response': response,
        'sources': [r['doc_id'] for r in results],
        'query': query
    }


# =============================================================================
# Streaming Response Generation
# =============================================================================

def parse_json_string_value(s: str) -> Tuple[str, bool]:
    """
    Parse characters of a JSON string value after the opening quote.

    Returns (decoded_text, is_complete).
    is_complete is True when the unescaped closing quote is found.
    Stops at buffer boundary (incomplete escape) without error.
    """
    result: List[str] = []
    i = 0
    escape_map = {'n': '\n', 't': '\t', 'r': '\r', '"': '"', '\\': '\\', '/': '/'}
    while i < len(s):
        c = s[i]
        if c == '\\':
            if i + 1 >= len(s):
                break  # Incomplete escape at buffer boundary
            next_c = s[i + 1]
            if next_c == 'u':
                if i + 5 < len(s):
                    result.append(chr(int(s[i + 2:i + 6], 16)))
                    i += 6
                else:
                    break  # Incomplete \uXXXX at buffer boundary
            else:
                result.append(escape_map.get(next_c, next_c))
                i += 2
        elif c == '"':
            return ''.join(result), True
        else:
            result.append(c)
            i += 1
    return ''.join(result), False


def _parse_llm_json(raw: str) -> dict:
    """
    Parse the JSON object from a raw LLM response, stripping any
    <think>…</think> chain-of-thought prefix first.

    NAOqi prosody tags (e.g. \\vct=108\\, \\rspd=82\\, \\pau=200\\) contain
    backslashes that are not valid JSON escape sequences.  If the first parse
    attempt fails we escape those lone backslashes and retry.

    Returns an empty dict if both attempts fail.
    """
    cleaned = raw.strip()
    if "</think>" in cleaned:
        cleaned = cleaned.split("</think>", 1)[1].strip()

    # First attempt — try as-is (handles correctly-escaped JSON)
    try:
        return json.loads(cleaned)
    except (json.JSONDecodeError, AttributeError, ValueError):
        pass

    # Second attempt — escape lone backslashes from NAOqi tags so JSON can parse.
    # Only preserve \" \\ \/ and \uXXXX as valid JSON escapes.
    # \r \n \b \f \t are intentionally NOT preserved because the LLM uses \r
    # as the start of \rspd= tags, not as a carriage-return escape.
    try:
        escaped = re.sub(r'\\(?!["\\/u])', r'\\\\', cleaned)
        return json.loads(escaped)
    except (json.JSONDecodeError, AttributeError, ValueError):
        return {}


def extract_answer_from_raw(raw: str) -> str:
    """
    Extract the spoken answer from a raw LLM response.

    Handles:
      - Thinking-capable models:  <think>...</think>{"answer": "..."}
      - JSON structured response: {"intent": "...", "answer": "..."}
      - Plain-text fallback:      raw text returned as-is

    Post-processing:
      - Strips surrounding curly/smart quotes the LLM sometimes adds.
      - Converts *tag=N* placeholders → \\tag=N\\ NAOqi control sequences.

    Note: sentence-level speed tags are applied separately in apply_speech_tags().
    """
    parsed = _parse_llm_json(raw)
    if parsed:
        answer = parsed.get("answer", raw)
    else:
        # Plain-text fallback
        answer = raw.strip()
        if "</think>" in answer:
            answer = answer.split("</think>", 1)[1].strip()

    # Strip surrounding smart/curly quotes e.g. "..." or "..."
    answer = answer.strip('""''"\'')

    # Convert *tag=N* placeholders → \tag=N\ NAOqi control sequences.
    # The LLM uses * to avoid JSON backslash escape conflicts.
    answer = re.sub(r'\*([a-z]+=\d+)\*', r'\\\1\\', answer)

    return answer.strip()


# Intents that benefit from slightly slower speech for clarity
_SLOW_INTENTS = {"ASK_EXHIBIT_QUESTION", "ASK_TOUR_META"}

def apply_speech_tags(answer: str, intent: str) -> str:
    """
    Prepend a sentence-level NAOqi speed tag based on intent.

    Word-level tags (*pau=N*, etc.) are already embedded in the answer by the
    LLM and converted by extract_answer_from_raw(). This function only adds
    the sentence-level \\rspd=N\\ prefix so the LLM never has to generate it.

    Args:
        answer: cleaned answer text (may already contain \\pau=N\\ tags)
        intent: LLM-classified intent string

    Returns:
        answer with \\rspd=N\\ prepended for slow intents, unchanged otherwise.
    """
    if not answer:
        return answer

    # No speed tag for short fixed responses
    if answer.strip().lower() in ("yes", "no"):
        return answer

    if intent in _SLOW_INTENTS:
        return "\\rspd=85\\" + answer

    return answer


def extract_intent_from_raw(raw: str) -> Tuple[str, float]:
    """
    Extract the intent label and confidence score from a raw LLM response.

    Returns ``(intent, confidence)`` where intent is one of:
      ASK_EXHIBIT_QUESTION | ASK_TOUR_META | NAVIGATION_REQUEST |
      SOCIAL_SMALL_TALK | OFF_TOPIC | STOP | AFFIRMATIVE | NEGATIVE
    and confidence is in [0.0, 1.0].

    Falls back to ("UNKNOWN", 0.0) if the JSON cannot be parsed or the
    fields are missing.
    """
    parsed = _parse_llm_json(raw)
    intent     = parsed.get("intent",     "UNKNOWN") if parsed else "UNKNOWN"
    confidence = parsed.get("confidence", 0.0)       if parsed else 0.0
    try:
        confidence = float(confidence)
    except (TypeError, ValueError):
        confidence = 0.0
    return str(intent), confidence


def generate_response_stream(
    query: str,
    search_results: List[Dict],
    conversation_history: Optional[List[Dict]],
    raw_response_out: List[str],
) -> Any:  # Iterator[str]
    """
    Stream the LLM response, yielding each answer sentence as soon as it arrives.

    The LLM is expected to return a JSON object with an "answer" field
    (matching the system prompt format).  Characters of the answer value are
    extracted token-by-token and yielded as complete sentences.

    Args:
        query:               User question.
        search_results:      Documents returned by vector search.
        conversation_history: Previous conversation turns (may be None).
        raw_response_out:    Single-element mutable list.  The full raw LLM
                             response is appended here when streaming finishes,
                             allowing the caller to parse intent/confidence.

    Yields:
        Individual answer sentences (str).
    """
    config = get_config()
    client = get_openai_client()

    # ---- Build messages (same logic as generate_response) ----
    messages: List[Dict] = [{"role": "system", "content": load_system_prompt()}]
    if conversation_history:
        for turn in conversation_history[-config.context_turns:]:
            messages.append({"role": "user", "content": turn.get("query", "")})
            messages.append({"role": "assistant", "content": turn.get("response", "")})

    user_content = f'Question: "{query}"'
    if search_results:
        context_parts = [f"[{r['title']}]\n{r['content']}" for r in search_results]
        user_content += "\n\nRelevant information:\n" + "\n\n".join(context_parts)
    messages.append({"role": "user", "content": user_content})

    max_tokens = config.max_response_sentences * 50 + 512

    raw_buffer = ""
    answer_chars = ""    # Accumulated decoded answer-value text seen so far
    answer_open_idx = -1  # Index in post-thinking buffer where answer value begins
    answer_done = False
    pending = ""          # Characters awaiting a sentence boundary

    try:
        stream = client.chat.completions.create(
            model=config.llm_model,
            messages=messages,
            max_tokens=max_tokens,
            stream=True,
        )

        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if not delta:
                continue
            raw_buffer += delta

            if answer_done:
                continue

            # ---- Strip chain-of-thought thinking section ----
            # Once </think> is present, post_think is everything after it.
            # While still inside <think>...</think>, skip processing.
            if "</think>" in raw_buffer:
                post_think = raw_buffer.split("</think>", 1)[1]
            elif "<think>" in raw_buffer:
                continue  # Still inside thinking block
            else:
                post_think = raw_buffer

            # ---- Locate the "answer" value start ----
            if answer_open_idx < 0:
                m = re.search(r'"answer"\s*:\s*"', post_think)
                if m:
                    answer_open_idx = m.end()

            if answer_open_idx < 0:
                continue  # Haven't reached the answer field yet

            # ---- Parse new characters from the answer value ----
            new_text, complete = parse_json_string_value(post_think[answer_open_idx:])
            new_chars = new_text[len(answer_chars):]
            answer_chars = new_text
            pending += new_chars

            if complete:
                answer_done = True
                if pending.strip():
                    yield pending.strip()
                pending = ""
            else:
                # Yield all complete sentences (ended by .!? followed by space)
                while True:
                    m2 = re.search(r'[.!?](?:\s|$)', pending)
                    if not m2:
                        break
                    cut = m2.end()
                    sentence = pending[:cut].strip()
                    if sentence:
                        yield sentence
                    pending = pending[cut:]

        # Yield any remaining partial sentence at end of stream
        if pending.strip():
            yield pending.strip()

        # Fallback: if the answer field was never found (non-JSON response),
        # yield the cleaned full response.
        if not answer_chars:
            fallback = extract_answer_from_raw(raw_buffer)
            if fallback.strip():
                yield fallback.strip()

    except Exception as e:
        logger.error(f"LLM streaming failed: {e}")
        yield "I'm sorry, I couldn't generate a response. Please try again."

    finally:
        raw_response_out.append(raw_buffer)

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
        retrieval:
          mode: "rag"  # "rag" or "full_context"
        search:
          similarity_threshold: 0.15
          top_k: 5
        conversation:
          max_history_turns: 15
          context_turns: 10
          max_response_sentences: 3
        data:
          default_path: "./data/upanzi_data.json"
        debug:
          verbose: true
    """
    messages = []
    
    yaml_config, error = read_yaml_config(file_path)
    if error:
        return False, [error]
    
    logger.debug(f"Successfully read YAML config from {file_path}")
    
    # Extract nested values with defaults
    llm = yaml_config.get('llm', {})
    embedding = yaml_config.get('embedding', {})
    retrieval = yaml_config.get('retrieval', {})
    search_cfg = yaml_config.get('search', {})
    conversation = yaml_config.get('conversation', {})
    data = yaml_config.get('data', {})
    debug = yaml_config.get('debug', {})

    # Parse retrieval mode
    retrieval_mode, warn = safe_str(
        retrieval.get('mode'),
        DEFAULT_RETRIEVAL_MODE,
        'retrieval.mode'
    )
    if warn:
        messages.append(warn)

    # Parse numeric values safely
    similarity_threshold, warn = safe_float(
        search_cfg.get('similarity_threshold'),
        DEFAULT_SIMILARITY_THRESHOLD,
        'similarity_threshold'
    )
    if warn:
        messages.append(warn)

    top_k, warn = safe_int(search_cfg.get('top_k'), DEFAULT_TOP_K, 'top_k')
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

    max_response_sentences, warn = safe_int(
        conversation.get('max_response_sentences'),
        DEFAULT_MAX_RESPONSE_SENTENCES,
        'max_response_sentences'
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
        logger.debug(f"Using package data folder for ChromaDB: {DEFAULT_CHROMA_PATH}")
        
        config = ConversationManagerConfig(
            llm_base_url=llm.get('base_url', DEFAULT_LLM_BASE_URL),
            # LLM_API_KEY must be exported as environment variable, not from config file
            llm_model=llm.get('model', DEFAULT_LLM_MODEL),
            chroma_path=DEFAULT_CHROMA_PATH,
            embedding_model=embedding.get('model', DEFAULT_EMBEDDING_MODEL),
            retrieval_mode=retrieval_mode,
            similarity_threshold=similarity_threshold,
            default_top_k=top_k,
            max_history_turns=max_history_turns,
            context_turns=context_turns,
            max_response_sentences=max_response_sentences,
            data_default_path=data_default_path,
            verbose=verbose,
        )
        
        # This will validate and raise ConfigError if invalid
        set_config(config)
        logger.debug(f"Configuration applied successfully: verbose={verbose}")
        return True, messages
        
    except ConfigError as e:
        logger.error(f"Configuration error: {e}")
        return False, [str(e)] + messages
    except Exception as e:
        logger.error(f"Failed to apply configuration: {e}")
        return False, [f"Failed to apply configuration: {e}"] + messages
