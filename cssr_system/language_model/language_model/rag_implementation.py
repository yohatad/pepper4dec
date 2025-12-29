import chromadb
import json
import openai
from chromadb.utils import embedding_functions
from typing import List, Dict, Any

client = openai.OpenAI(
    base_url="http://localhost:8080/v1",
    api_key = "sk-no-key-required"
)

# Initialize ChromaDB client
chroma_client = chromadb.HttpClient(host='localhost', port=8000)

def load_json_data(file_path: str) -> List[Dict]:
    """Load data from JSON file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            json_data = json.load(file)

        subsections = []

        # Ensure each item has required fields and normalize the structure
        for i, item in enumerate(json_data):
            # Normalize food_id to string
            if 'doc_id' not in item:
                item['doc_id'] = str(i + 1)
            else:
                item['doc_id'] = str(item['doc_id'])
            
            # Ensure required fields exist
            if 'section' not in item:
                item['section'] = ''
            if 'subsections' not in item:
                item['subsections'] = []
            if 'content' not in item:
                item['content'] = ''
            
            # Extract nested subsections if available
            for idx, sub_section in enumerate(item.get('subsections', [])):
                sub_section['doc_id'] = str(sub_section.get('doc_id', idx))
                sub_section['doc_id'] = str(item['doc_id']+'_'+sub_section['doc_id'])
                sub_section['title'] = sub_section.get('title', '')
                sub_section['overview'] = sub_section.get('overview', '')
                sub_section['tags'] = sub_section.get('tags', [])
                sub_section['status'] = sub_section.get('status', '')
                subsections.append(sub_section)

            del item['subsections']

        result = json_data + subsections

        print(f"Successfully loaded {len(result)} items from {file_path}")
        return result

    except Exception as e:
        print(f"Error loading json data: {e}")
        return []

def create_similarity_search_collection(collection_name: str, collection_metadata: dict = None):
    collection = get_similarity_search_collection(collection_name)
    if collection:
        print(f"Collection '{collection_name}' already exists. Using existing collection.")
        return collection
    
    """Create ChromaDB collection with sentence transformer embeddings"""
    # try:
    #     # Try to delete existing collection to start fresh
    #     chroma_client.delete_collection(collection_name)
    # except:
    #     pass
    
    # Create embedding function
    sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    
    # Create new collection
    return chroma_client.create_collection(
        name=collection_name,
        metadata=collection_metadata,
        configuration={
            "hnsw": {"space": "cosine"},
            "embedding_function": sentence_transformer_ef
        }
    )

def get_similarity_search_collection(collection_name: str):
    """Retrieve ChromaDB collection"""
    try:
        return chroma_client.get_collection(name=collection_name)
    except Exception as e:
        print(f"Error retrieving collection {collection_name}: {e}")
        return None

def populate_similarity_collection(collection, data_items: List[Dict]):
    """Populate collection with data and generate embeddings"""
    documents = []
    metadatas = []
    ids = []
    
    # Create unique IDs to avoid duplicates
    used_ids = set()
    
    for i, data in enumerate(data_items):
        if data.get("content", '') == '' and data.get("overview", '') == '':
            continue
        
        # Create comprehensive text for embedding using rich JSON structure
        if 'section' in data:
            text = f"{data['section']}: "
            text += f"{data.get('content', '')}. "
            metadata = {
                "section": data["section"]
                }
        else:
            tags = ''
            for tag in data.get('tags', []):
                tags += tag + ', '
            text = f"Title: {data.get('title', '')} \n"
            text += f"Tags: {tags} \n" if tags else ''
            text += f"Status: {data.get('status', '')} \n" if 'status' in data else ''
            text += f"{data.get('overview', '')}. "
            metadata = {
                "title": data.get('title', ''),
                "tags": " ".join(data.get('tags', [])),
                "status": data.get('status', '')
                }
        
        # Generate unique ID to avoid duplicates
        base_id = str(data.get('doc_id', i))
        unique_id = base_id
        counter = 1
        while unique_id in used_ids:
            unique_id = f"{base_id}_{counter}"
            counter += 1
        used_ids.add(unique_id)
        
        documents.append(text)
        ids.append(unique_id)
        metadatas.append(metadata)
    
    # Add all data to collection
    collection.add(
        documents=documents,
        metadatas=metadatas,
        ids=ids
    )
    
    print(f"Added {len(documents)} items to collection")

def perform_similarity_search(collection, query: str, n_results: int = 5) -> List[Dict]:
    """Perform similarity search and return formatted results"""
    try:
        results = collection.query(
            query_texts=[query],
            n_results=n_results
        )
        
        if not results or not results['ids'] or len(results['ids'][0]) == 0:
            return []
        
        formatted_results = []
        # print(results)
        for i in range(len(results['ids'][0])):
            # Calculate similarity score (1 - distance)
            similarity_score = 1 - results['distances'][0][i]
            section = ''
            if "section" not in results['metadatas'][0][i]:
                section = results['metadatas'][0][i]['title']
            else:
                section = results['metadatas'][0][i]['section']
            result = {
                'doc_id': results['ids'][0][i],
                'section': section,
                'content': results['documents'][0][i],
                'similarity_score': similarity_score,
                'distance': results['distances'][0][i]
            }
            if result['similarity_score'] > 0.05:  # Filter out very low similarity scores
                formatted_results.append(result)
        
        return formatted_results
        
    except Exception as e:
        print(f"Error in similarity search: {e}")
        return []

def perform_filtered_similarity_search(collection, query: str, section_filter: str = None, 
                                     n_results: int = 5) -> List[Dict]:
    """Perform filtered similarity search with metadata constraints"""
    where_clause = None
    
    # Build filters list
    filters = []
    if section_filter:
        filters.append({"section": section_filter})
    
    # Construct where clause based on number of filters
    if len(filters) == 1:
        where_clause = filters[0]
    elif len(filters) > 1:
        where_clause = {"$and": filters}
    
    try:
        results = collection.query(
            query_texts=[query],
            n_results=n_results,
            where=where_clause
        )
        
        if not results or not results['ids'] or len(results['ids'][0]) == 0:
            return []
        
        formatted_results = []
        for i in range(len(results['ids'][0])):
            similarity_score = 1 - results['distances'][0][i]
            
            result = {
                'doc_id': results['ids'][0][i],
                'section': results['metadatas'][0][i]['section'],
                'content': results['documents'][0][i],
                'similarity_score': similarity_score,
                'distance': results['distances'][0][i]
            }
            if result['similarity_score'] > 0.15:  # Filter out very low similarity scores
                formatted_results.append(result)
        
        return formatted_results
        
    except Exception as e:
        print(f"Error in filtered search: {e}")
        return []

def clear_collection(collection):
    """Clear all items from the collection"""
    try:
        collection.delete()
        print("Collection cleared successfully")
    except Exception as e:
        print(f"Error clearing collection: {e}")

def delete_collection(collection_name: str):
    """Delete the entire collection"""
    try:
        chroma_client.delete_collection(collection_name)
        print(f"Collection '{collection_name}' deleted successfully")
    except Exception as e:
        print(f"Error deleting collection '{collection_name}': {e}")

def list_collections() -> List[str]:
    """List all existing collections"""
    try:
        collections = chroma_client.list_collections()
        return [col.name for col in collections]
    except Exception as e:
        print(f"Error listing collections: {e}")
        return []

def get_collection_stats(collection) -> Dict[str, Any]:
    """Get statistics about the collection"""
    try:
        stats = collection.count()
        return stats
    except Exception as e:
        print(f"Error getting collection stats: {e}")
        return {}

def read_config(file_path):
    config = {}
    with open(file_path, 'r') as file:
        for line in file:
            if line.strip() and not line.startswith('#'):
                key, value = line.split(' ', 1)
                config[key] = value.strip()
    return config

def create_collection_and_load_data(name: str, description: str, data_file_path: str, verbose_mode: bool = False):
    """Create collection and load data from config file"""
    try:
        
        if verbose_mode:
            print(f"Loading data from: {data_file_path}")
        
        data_items = load_json_data(data_file_path)
        
        if verbose_mode:
            print(f"Loaded {len(data_items)} items from data file.")
        
        collection = create_similarity_search_collection(
            name,
            {'description': description}
        )
        
        populate_similarity_collection(collection, data_items)
        
        if verbose_mode:
            print("Collection created and populated successfully.")
        
        return collection
    
    except Exception as e:
        print(f"Error in creating collection and loading data: {e}")
        return None


def prepare_context_for_llm(query: str, search_results: List[Dict]) -> str:
    """Prepare structured context from search results for LLM"""
    if not search_results:
        return None
    
    context_docs = []
    
    for i, result in enumerate(search_results, 1):
        context_docs.append(f"{result['content']}")
    
    return "\n\n".join(context_docs)

def generate_llm_rag_response(query: str, search_results: List[Dict], conversation_history: List[str], intent=False) -> str:
    """Generate response using llama.cpp with retrieved context"""
    try:
        # Prepare context from search results
        context = prepare_context_for_llm(query, search_results)

        # Build messages for chat completion
        messages = []
        
        sys_prompt_intent = "You are pepper, a humanoid robot that acts as a lab assistant here to help visitors with questions. A user is visiting the Robotics lab in Carnegie Mellon University Africa and they were asked a question. You need to determinde their intent. Please provide only 'positive' or 'negative' as answer to the query and no further text."
        system_prompt = "You are pepper, a humanoid robot that acts as a lab assistant here to help visitors with questions. A user is visiting the Robotics lab in Carnegie Mellon University Africa and asking questions. Please provide a helpful, short response of not more than three sentences to their query. Look out for terms like 'Upanzi', 'Roomba', 'Lynx motion', etc. They might be wrongly spelled. If you don't know the answer, just say you don't know. Never make up an answer. Be polite and friendly."

        if intent:
            messages = [
                {"role": "system", "content": sys_prompt_intent},
                {"role": "user", "content": f'"{query}"'}
            ]
        else:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f'"{query}" \n Information that might be relevant to the query: \n {context}'} if context else {"role": "user", "content": f'"{query}"'}
            ]
            
        # print(f'Prepared Messages for LLM: {messages}')

        # Generate response using IBM Granite
        generated_response = client.chat.completions.create(
            model="HuggingFaceTB/SmolLM3-3B",
            messages=messages,
        )
        
        # Extract the generated text
        if len(generated_response.choices) > 0:
            response_text = generated_response.choices[0].message.content
            response_text = response_text.split("</think>")[-1]

            # Clean up the response if needed
            response_text = response_text.strip()
            
            # If response is too short, provide a fallback
            if len(response_text) < 50 and not intent:
                return generate_fallback_response(query, search_results)
            
            return response_text
        else:
            return generate_fallback_response(query, search_results)
            
    except Exception as e:
        print(f"❌ LLM Error: {e}")
        return generate_fallback_response(query, search_results)

def generate_fallback_response(query: str, search_results: List[Dict]) -> str:
    """Generate fallback response when LLM fails"""

    return "I don't understand what you mean. Tsegazeab can you handle the question please!"

def handle_rag_query(collection, query: str, conversation_history: List[str], verbose_mode: bool = False, top_k: int = 3, intent=False) -> str:
    """Handle user query with enhanced RAG approach"""
    if verbose_mode:
        print(f"\n🔍 Searching vector database for: '{query}'...")
    
    search_results = []
    # Perform similarity search with more results for better context
    if not intent:
        search_results = perform_similarity_search(collection, query, top_k)

    if verbose_mode:
        print(f"✅ Found {len(search_results)} relevant matches")
        print("🧠 Generating AI-powered response...")
    
    # Generate enhanced RAG response using LLM
    ai_response = generate_llm_rag_response(query, search_results, conversation_history, intent=intent)

    if verbose_mode:
        print(f"\n🤖 Bot: {ai_response}")
    
    # Show detailed results for reference
    if verbose_mode:
        print(f"\n📊 Search Results Details:")
        print("-" * 45)
        for i, result in enumerate(search_results[:3], 1):
            print(f"{i}. 🍽️  {result['section']}")
            print(f"   📍 {result['content']} | 📈 {result['similarity_score']*100:.1f}% match")
            if i < 3:
                print()

    return ai_response