"""
Simplified RAG Implementation for Pepper Robot Lab Assistant

Clean RAG system without intent classification.
Designed for controlled interaction where the application manages conversation flow.

Environment Variables:
    LLM_BASE_URL: LLM API endpoint (default: http://localhost:8080/v1)
    LLM_API_KEY: API key for LLM service (default: sk-no-key-required)
    LLM_MODEL: Model name (default: HuggingFaceTB/SmolLM3-3B)
    CHROMA_PATH: Path for embedded ChromaDB storage (default: ./chroma_data)
    EMBEDDING_MODEL: Sentence transformer model (default: all-MiniLM-L6-v2)
"""

import os
import json
import openai
import chromadb
from chromadb.utils import embedding_functions
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class RAGConfig:
    """
    Configuration for RAG system.
    
    Values are loaded from environment variables with sensible defaults.
    Set environment variables to override:
        export LLM_API_KEY="your-api-key"
        export LLM_BASE_URL="https://api.groq.com/openai/v1"
    """
    
    # LLM settings (from environment)
    llm_base_url: str = field(
        default_factory=lambda: os.getenv("LLM_BASE_URL", "http://localhost:8080/v1")
    )
    llm_api_key: str = field(
        default_factory=lambda: os.getenv("LLM_API_KEY", "sk-no-key-required")
    )
    llm_model: str = field(
        default_factory=lambda: os.getenv("LLM_MODEL", "HuggingFaceTB/SmolLM3-3B")
    )
    
    # ChromaDB settings (embedded mode - no server needed)
    chroma_path: str = field(
        default_factory=lambda: os.getenv("CHROMA_PATH", "./chroma_data")
    )
    
    # Embedding settings
    embedding_model: str = field(
        default_factory=lambda: os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    )
    
    # Search settings
    similarity_threshold: float = 0.15
    default_top_k: int = 5
    
    # Response settings
    max_response_sentences: int = 3


class RAGError(Exception):
    """Custom exception for RAG-related errors"""
    pass


# =============================================================================
# Global Clients
# =============================================================================

_config: Optional[RAGConfig] = None
_openai_client: Optional[openai.OpenAI] = None
_chroma_client: Optional[chromadb.HttpClient] = None
_embedding_function = None


def get_config() -> RAGConfig:
    """Get or create default configuration"""
    global _config
    if _config is None:
        _config = RAGConfig()
    return _config


def set_config(config: RAGConfig) -> None:
    """Set custom configuration"""
    global _config, _openai_client, _chroma_client, _embedding_function
    _config = config
    _openai_client = None
    _chroma_client = None
    _embedding_function = None


def get_openai_client() -> openai.OpenAI:
    """Get or create OpenAI client"""
    global _openai_client
    if _openai_client is None:
        config = get_config()
        _openai_client = openai.OpenAI(
            base_url=config.llm_base_url,
            api_key=config.llm_api_key,
            timeout=30.0
        )
    return _openai_client


def get_chroma_client() -> chromadb.ClientAPI:
    """Get or create ChromaDB client (embedded mode - no server needed)"""
    global _chroma_client
    if _chroma_client is None:
        config = get_config()
        try:
            # Embedded mode - data stored locally, no server required
            _chroma_client = chromadb.PersistentClient(path=config.chroma_path)
        except Exception as e:
            raise RAGError(f"Failed to initialize ChromaDB: {e}")
    return _chroma_client


def get_embedding_function():
    """Get or create embedding function"""
    global _embedding_function
    if _embedding_function is None:
        config = get_config()
        _embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=config.embedding_model
        )
    return _embedding_function


# =============================================================================
# Data Loading
# =============================================================================

def load_json_data(file_path: str) -> List[Dict]:
    """Load and parse JSON knowledge base file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            json_data = json.load(file)

        if isinstance(json_data, dict):
            return _parse_upanzi_format(json_data)
        elif isinstance(json_data, list):
            return json_data
        else:
            raise RAGError(f"Unexpected JSON type: {type(json_data).__name__}")

    except json.JSONDecodeError as e:
        raise RAGError(f"Invalid JSON: {e}")
    except FileNotFoundError:
        raise RAGError(f"File not found: {file_path}")


def _parse_upanzi_format(json_data: Dict) -> List[Dict]:
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
    
    print(f"Loaded {len(documents)} documents")
    return documents


# =============================================================================
# Collection Management
# =============================================================================

def create_collection(name: str, description: str = "") -> chromadb.Collection:
    """Create or get ChromaDB collection"""
    existing = get_collection(name)
    if existing:
        print(f"Using existing collection: {name}")
        return existing
    
    client = get_chroma_client()
    ef = get_embedding_function()
    
    collection = client.create_collection(
        name=name,
        metadata={'description': description} if description else None,
        embedding_function=ef
    )
    print(f"Created collection: {name}")
    return collection


def get_collection(name: str) -> Optional[chromadb.Collection]:
    """Get existing collection"""
    try:
        client = get_chroma_client()
        ef = get_embedding_function()
        return client.get_collection(name=name, embedding_function=ef)
    except Exception:
        return None


def delete_collection(name: str) -> bool:
    """Delete a collection"""
    try:
        client = get_chroma_client()
        client.delete_collection(name)
        return True
    except Exception:
        return False


def populate_collection(collection: chromadb.Collection, documents: List[Dict]) -> int:
    """Add documents to collection"""
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
        collection.add(documents=docs, ids=ids, metadatas=metadatas)
    
    print(f"Added {len(docs)} documents to collection")
    return len(docs)


def setup_collection(name: str, json_path: str, description: str = "") -> chromadb.Collection:
    """Convenience: Create collection and load data"""
    documents = load_json_data(json_path)
    collection = create_collection(name, description)
    
    if collection.count() == 0:
        populate_collection(collection, documents)
    
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
        return []
    
    config = get_config()
    top_k = top_k or config.default_top_k
    
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
            return []
        
        formatted = []
        for i in range(len(results['ids'][0])):
            score = 1 - results['distances'][0][i]
            
            if score < config.similarity_threshold:
                continue
            
            formatted.append({
                'doc_id': results['ids'][0][i],
                'title': results['metadatas'][0][i].get('title', ''),
                'content': results['documents'][0][i],
                'score': score
            })
        
        return formatted
        
    except Exception as e:
        print(f"Search error: {e}")
        return []


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
    """
    config = get_config()
    
    # Build context from search results
    context = ""
    if search_results:
        context_parts = [f"[{r['title']}]\n{r['content']}" for r in search_results[:3]]
        context = "\n\n".join(context_parts)
    
    # Build messages
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    
    # Add conversation history if provided
    if conversation_history:
        for turn in conversation_history[-3:]:  # Last 3 turns
            messages.append({"role": "user", "content": turn.get("query", "")})
            messages.append({"role": "assistant", "content": turn.get("response", "")})
    
    # Add current query with context
    user_content = f'Question: "{query}"'
    if context:
        user_content += f"\n\nRelevant information:\n{context}"
    
    messages.append({"role": "user", "content": user_content})
    
    try:
        client = get_openai_client()
        response = client.chat.completions.create(
            model=config.llm_model,
            messages=messages
        )
        
        if response.choices:
            answer = response.choices[0].message.content.strip()
            
            # Clean thinking tags if present
            if "</think>" in answer:
                answer = answer.split("</think>")[-1].strip()
            
            return answer
        
        return "I'm sorry, I couldn't generate a response. Please try again."
        
    except Exception as e:
        print(f"LLM error: {e}")
        return "I encountered an error. Please try again."


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
        conversation_history: Previous conversation turns
        category_filter: Optional category to filter search
        top_k: Number of documents to retrieve
        
    Returns:
        Dict with 'response', 'sources', and 'query'
    """
    if not query or not query.strip():
        return {
            'response': "I didn't catch that. Could you please repeat?",
            'sources': [],
            'query': query
        }
    
    # Search for relevant documents
    results = search(collection, query, top_k=top_k, category_filter=category_filter)
    
    # Generate response
    response = generate_response(query, results, conversation_history)
    
    return {
        'response': response,
        'sources': [r['doc_id'] for r in results],
        'query': query
    }


# =============================================================================
# Utility Functions
# =============================================================================

def read_config_file(file_path: str) -> Dict[str, str]:
    """Read configuration from ini-style file"""
    config = {}
    try:
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and ' ' in line:
                    key, value = line.split(' ', 1)
                    config[key] = value.strip()
    except Exception as e:
        print(f"Config error: {e}")
    return config


def apply_config_file(file_path: str) -> None:
    """Load and apply configuration from file"""
    file_config = read_config_file(file_path)
    
    config = RAGConfig(
        llm_base_url=file_config.get('llmBaseUrl', RAGConfig.llm_base_url),
        llm_model=file_config.get('llmModel', RAGConfig.llm_model),
        chroma_host=file_config.get('chromaHost', RAGConfig.chroma_host),
        chroma_port=int(file_config.get('chromaPort', RAGConfig.chroma_port)),
        embedding_model=file_config.get('embeddingModel', RAGConfig.embedding_model),
        similarity_threshold=float(file_config.get('similarityThreshold', RAGConfig.similarity_threshold)),
        default_top_k=int(file_config.get('topK', RAGConfig.default_top_k)),
    )
    
    set_config(config)