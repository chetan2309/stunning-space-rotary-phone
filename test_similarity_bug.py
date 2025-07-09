#!/usr/bin/env python3
"""
Test script to demonstrate the similarity filtering bug.
This shows how the current implementation returns irrelevant test suggestions.
"""

try:
    import pysqlite3
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass

import chromadb
from sentence_transformers import SentenceTransformer

# Configuration
DB_PATH = "testrail_db"
COLLECTION_NAME = "testrail_embeddings"
MODEL_NAME = 'all-MiniLM-L6-v2'
NUM_TESTS_TO_SUGGEST = 3

def test_current_behavior():
    """Test the current behavior that shows the bug"""
    print("=== TESTING CURRENT BEHAVIOR (WITH BUG) ===")
    
    model = SentenceTransformer(MODEL_NAME)
    client = chromadb.PersistentClient(path=DB_PATH)
    collection = client.get_collection(name=COLLECTION_NAME)
    
    # Test with completely unrelated code changes
    unrelated_code = """
    Pull Request Title: Fix CSS styling for navigation bar
    
    --- File: styles.css ---
    .navbar {
        background-color: #333;
        padding: 10px;
    }
    
    .navbar-brand {
        color: white;
        font-size: 18px;
    }
    """
    
    print(f"Query: {unrelated_code[:100]}...")
    query_embedding = model.encode(unrelated_code).tolist()
    results = collection.query(query_embeddings=[query_embedding], n_results=NUM_TESTS_TO_SUGGEST)
    
    print(f"\nResults returned: {len(results['metadatas'][0]) if results['metadatas'][0] else 0}")
    
    if results and results.get('metadatas') and results['metadatas'][0]:
        print("\nCurrent implementation suggests these tests:")
        for i, (meta, distance) in enumerate(zip(results['metadatas'][0], results['distances'][0])):
            print(f"  {i+1}. T{meta['case_id']}: {meta['title']} (distance: {distance:.4f})")
    
    print(f"\n❌ BUG: Even for CSS changes, the system suggests {NUM_TESTS_TO_SUGGEST} tests!")
    print("   These suggestions are likely irrelevant due to high distance scores.")

def test_with_similarity_threshold():
    """Test with a similarity threshold to filter irrelevant results"""
    print("\n=== TESTING WITH SIMILARITY THRESHOLD (FIXED) ===")
    
    SIMILARITY_THRESHOLD = 0.7  # Lower distance = more similar
    
    model = SentenceTransformer(MODEL_NAME)
    client = chromadb.PersistentClient(path=DB_PATH)
    collection = client.get_collection(name=COLLECTION_NAME)
    
    # Test with the same unrelated code
    unrelated_code = """
    Pull Request Title: Fix CSS styling for navigation bar
    
    --- File: styles.css ---
    .navbar {
        background-color: #333;
        padding: 10px;
    }
    """
    
    print(f"Query: {unrelated_code[:100]}...")
    query_embedding = model.encode(unrelated_code).tolist()
    results = collection.query(query_embeddings=[query_embedding], n_results=NUM_TESTS_TO_SUGGEST)
    
    # Filter by similarity threshold
    filtered_tests = []
    if results and results.get('metadatas') and results['metadatas'][0]:
        for meta, distance in zip(results['metadatas'][0], results['distances'][0]):
            if distance <= SIMILARITY_THRESHOLD:
                filtered_tests.append({"id": meta['case_id'], "title": meta['title'], "distance": distance})
    
    print(f"\nFiltered results (distance <= {SIMILARITY_THRESHOLD}): {len(filtered_tests)}")
    
    if filtered_tests:
        print("\nRelevant test suggestions:")
        for i, test in enumerate(filtered_tests):
            print(f"  {i+1}. T{test['id']}: {test['title']} (distance: {test['distance']:.4f})")
    else:
        print("✅ FIXED: No relevant tests found for CSS changes - this is correct!")
    
    # Test with more relevant code
    print("\n" + "="*60)
    relevant_code = """
    Pull Request Title: Add user authentication validation
    
    --- File: auth.js ---
    function validateLogin(username, password) {
        if (!username || !password) {
            return false;
        }
        return authenticateUser(username, password);
    }
    """
    
    print(f"Query: {relevant_code[:100]}...")
    query_embedding = model.encode(relevant_code).tolist()
    results = collection.query(query_embeddings=[query_embedding], n_results=NUM_TESTS_TO_SUGGEST)
    
    # Filter by similarity threshold
    filtered_tests = []
    if results and results.get('metadatas') and results['metadatas'][0]:
        for meta, distance in zip(results['metadatas'][0], results['distances'][0]):
            if distance <= SIMILARITY_THRESHOLD:
                filtered_tests.append({"id": meta['case_id'], "title": meta['title'], "distance": distance})
    
    print(f"\nFiltered results (distance <= {SIMILARITY_THRESHOLD}): {len(filtered_tests)}")
    
    if filtered_tests:
        print("\nRelevant test suggestions:")
        for i, test in enumerate(filtered_tests):
            print(f"  {i+1}. T{test['id']}: {test['title']} (distance: {test['distance']:.4f})")
    else:
        print("No relevant tests found for authentication changes")

if __name__ == "__main__":
    test_current_behavior()
    test_with_similarity_threshold()