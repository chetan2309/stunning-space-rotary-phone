#!/usr/bin/env python3
"""
Test script to verify the similarity filtering fix works correctly.
"""

try:
    import pysqlite3
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass

import chromadb
from sentence_transformers import SentenceTransformer

# Use the same configuration as the fixed code
DB_PATH = "testrail_db"
COLLECTION_NAME = "testrail_embeddings"
MODEL_NAME = 'all-MiniLM-L6-v2'
NUM_TESTS_TO_SUGGEST = 5
SIMILARITY_THRESHOLD = 0.8

def test_similarity_filtering():
    """Test the fixed similarity filtering logic"""
    print("=== TESTING FIXED SIMILARITY FILTERING ===")
    
    model = SentenceTransformer(MODEL_NAME)
    client = chromadb.PersistentClient(path=DB_PATH)
    collection = client.get_collection(name=COLLECTION_NAME)
    
    test_cases = [
        {
            "name": "CSS Styling Changes",
            "code": """
            Pull Request Title: Fix CSS styling for navigation bar
            
            --- File: styles.css ---
            .navbar {
                background-color: #333;
                padding: 10px;
            }
            """,
            "expected_relevant": False
        },
        {
            "name": "Login Authentication",
            "code": """
            Pull Request Title: Add user authentication validation
            
            --- File: auth.js ---
            function validateLogin(username, password) {
                if (!username || !password) {
                    return false;
                }
                return authenticateUser(username, password);
            }
            """,
            "expected_relevant": True
        },
        {
            "name": "Database Migration",
            "code": """
            Pull Request Title: Add database migration for user table
            
            --- File: migration.sql ---
            ALTER TABLE users ADD COLUMN last_login TIMESTAMP;
            CREATE INDEX idx_users_last_login ON users(last_login);
            """,
            "expected_relevant": False
        }
    ]
    
    for test_case in test_cases:
        print(f"\n{'='*60}")
        print(f"Testing: {test_case['name']}")
        print(f"Expected to find relevant tests: {test_case['expected_relevant']}")
        print(f"{'='*60}")
        
        query_embedding = model.encode(test_case['code']).tolist()
        results = collection.query(query_embeddings=[query_embedding], n_results=NUM_TESTS_TO_SUGGEST)
        
        # Apply the same filtering logic as the fixed code
        selected_tests = []
        if results and results.get('metadatas') and results['metadatas'][0] and results.get('distances'):
            print(f"\nRaw results from ChromaDB:")
            for i, (meta, distance) in enumerate(zip(results['metadatas'][0], results['distances'][0])):
                print(f"  {i+1}. T{meta['case_id']}: {meta['title']} (distance: {distance:.4f})")
            
            # Filter tests by similarity threshold
            for meta, distance in zip(results['metadatas'][0], results['distances'][0]):
                if distance <= SIMILARITY_THRESHOLD:
                    selected_tests.append({
                        "id": meta['case_id'], 
                        "title": meta['title'],
                        "similarity_score": round(1 - distance, 3)
                    })
            
            # Sort by similarity (highest first) and limit to top 3
            selected_tests.sort(key=lambda x: x['similarity_score'], reverse=True)
            selected_tests = selected_tests[:3]
        
        print(f"\nFiltered results (distance <= {SIMILARITY_THRESHOLD}):")
        if selected_tests:
            print(f"Found {len(selected_tests)} relevant test(s):")
            for test in selected_tests:
                similarity_percentage = int(test['similarity_score'] * 100)
                print(f"  - T{test['id']}: {test['title']} (Relevance: {similarity_percentage}%)")
        else:
            print("No relevant tests found - this is correct for irrelevant code changes!")
        
        # Verify expectation
        has_relevant = len(selected_tests) > 0
        if has_relevant == test_case['expected_relevant']:
            print(f"✅ PASS: Expectation met")
        else:
            print(f"❌ FAIL: Expected relevant={test_case['expected_relevant']}, got relevant={has_relevant}")

def test_github_comment_format():
    """Test the GitHub comment formatting"""
    print(f"\n{'='*60}")
    print("TESTING GITHUB COMMENT FORMAT")
    print(f"{'='*60}")
    
    # Mock test data
    selected_tests = [
        {"id": 1, "title": "Valid Login 1", "similarity_score": 0.234},
        {"id": 4, "title": "Invalid Login 2", "similarity_score": 0.198}
    ]
    
    testrail_url = "https://example.testrail.com/"
    
    # Format comment like the fixed code does
    comment = "🤖 **Intelligent Test Case Suggestion** 🤖\n\n"
    if selected_tests:
        comment += f"Based on the code changes, I found **{len(selected_tests)}** relevant test case(s) that the QA team should consider running:\n\n"
        for test in selected_tests:
            case_url = f"{testrail_url}index.php?/cases/view/{test['id']}"
            similarity_percentage = int(test['similarity_score'] * 100)
            comment += f"- **T{test['id']}**: [{test['title']}]({case_url}) *(Relevance: {similarity_percentage}%)*\n"
        
        comment += f"\n*Note: Only tests with >20% relevance are suggested (similarity threshold: {int((1-SIMILARITY_THRESHOLD)*100)}%)*"
    
    print("Generated GitHub comment:")
    print(comment)

if __name__ == "__main__":
    test_similarity_filtering()
    test_github_comment_format()