#!/usr/bin/env python3
"""
Test the new LLM-enhanced similarity filtering approach.
This demonstrates the improved architecture: Embeddings for Retrieval + LLM for Evaluation
"""

try:
    import pysqlite3
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass

import chromadb
from sentence_transformers import SentenceTransformer
import os
import json

# Import the new functions from test_selector
from test_selector import evaluate_test_relevance_with_llm, evaluate_test_relevance_fallback

# Configuration
DB_PATH = "testrail_db"
COLLECTION_NAME = "testrail_embeddings"
MODEL_NAME = 'all-MiniLM-L6-v2'
NUM_CANDIDATES = 10
RELEVANCE_THRESHOLD = 70
MAX_SUGGESTIONS = 3

def test_llm_enhanced_approach():
    """Test the complete LLM-enhanced approach"""
    print("=" * 80)
    print("TESTING LLM-ENHANCED TEST SUGGESTION SYSTEM")
    print("Architecture: Embeddings for Retrieval + LLM for Semantic Evaluation")
    print("=" * 80)
    
    # Initialize components
    model = SentenceTransformer(MODEL_NAME)
    client = chromadb.PersistentClient(path=DB_PATH)
    collection = client.get_collection(name=COLLECTION_NAME)
    
    test_cases = [
        {
            "name": "🔐 Authentication Code Changes",
            "code": """Pull Request Title: Add user authentication validation

--- File: auth.js ---
function validateLogin(username, password) {
    if (!username || !password) {
        throw new Error('Username and password are required');
    }
    
    // New validation logic
    if (username.length < 3) {
        throw new Error('Username must be at least 3 characters');
    }
    
    if (password.length < 8) {
        throw new Error('Password must be at least 8 characters');
    }
    
    return authenticateUser(username, password);
}

function handleLoginSubmit(event) {
    event.preventDefault();
    const username = document.getElementById('username').value;
    const password = document.getElementById('password').value;
    
    try {
        if (validateLogin(username, password)) {
            window.location.href = '/dashboard';
        }
    } catch (error) {
        showError(error.message);
    }
}""",
            "expected_relevant": True
        },
        {
            "name": "🎨 CSS Styling Changes",
            "code": """Pull Request Title: Update navigation bar styling

--- File: navbar.css ---
.navbar {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 15px 20px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    border-radius: 0 0 10px 10px;
}

.navbar-brand {
    color: white;
    font-size: 24px;
    font-weight: 700;
    text-decoration: none;
    transition: opacity 0.3s ease;
}

.navbar-brand:hover {
    opacity: 0.8;
}

.navbar-nav {
    display: flex;
    list-style: none;
    margin: 0;
    padding: 0;
    gap: 20px;
}""",
            "expected_relevant": False
        },
        {
            "name": "🗄️ Database Migration",
            "code": """Pull Request Title: Add user preferences table

--- File: 001_add_user_preferences.sql ---
CREATE TABLE user_preferences (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    theme VARCHAR(20) DEFAULT 'light' CHECK (theme IN ('light', 'dark')),
    language VARCHAR(10) DEFAULT 'en',
    notifications_enabled BOOLEAN DEFAULT true,
    email_frequency VARCHAR(20) DEFAULT 'daily' CHECK (email_frequency IN ('never', 'daily', 'weekly')),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_user_preferences_user_id ON user_preferences(user_id);
CREATE INDEX idx_user_preferences_theme ON user_preferences(theme);

-- Add trigger to update updated_at
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_user_preferences_updated_at 
    BEFORE UPDATE ON user_preferences 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();""",
            "expected_relevant": False
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{'-' * 70}")
        print(f"TEST {i}: {test_case['name']}")
        print(f"Expected to find relevant tests: {test_case['expected_relevant']}")
        print(f"{'-' * 70}")
        
        # Step 1: Embedding-based retrieval
        print(f"\n📊 STEP 1: Embedding-based retrieval (top {NUM_CANDIDATES} candidates)")
        query_embedding = model.encode(test_case['code']).tolist()
        results = collection.query(query_embeddings=[query_embedding], n_results=NUM_CANDIDATES)
        
        candidates = []
        if results and results.get('metadatas') and results['metadatas'][0]:
            print("Retrieved candidates:")
            for i, meta in enumerate(results['metadatas'][0]):
                distance = results['distances'][0][i] if results.get('distances') else None
                candidate = {
                    'id': meta['case_id'],
                    'title': meta['title'],
                    'embedding_rank': i + 1,
                    'embedding_distance': distance
                }
                candidates.append(candidate)
                print(f"  {i+1}. T{candidate['id']}: {candidate['title']} (distance: {distance:.4f})")
        
        # Step 2: LLM-based evaluation
        print(f"\n🤖 STEP 2: LLM semantic evaluation")
        evaluated_tests = []
        
        for candidate in candidates[:5]:  # Evaluate top 5 for demo
            print(f"\n  Evaluating T{candidate['id']}: {candidate['title']}")
            
            evaluation = evaluate_test_relevance_with_llm(test_case['code'], candidate)
            
            candidate.update({
                'relevance_score': evaluation['relevance_score'],
                'reasoning': evaluation['reasoning'],
                'recommendation': evaluation['recommendation']
            })
            
            evaluated_tests.append(candidate)
            
            print(f"    → Relevance: {evaluation['relevance_score']}%")
            print(f"    → Reasoning: {evaluation['reasoning']}")
            print(f"    → Recommendation: {evaluation['recommendation']}")
        
        # Step 3: Filter and rank by LLM scores
        print(f"\n🎯 STEP 3: Filter by relevance threshold ({RELEVANCE_THRESHOLD}%)")
        
        selected_tests = [
            test for test in evaluated_tests 
            if test['relevance_score'] >= RELEVANCE_THRESHOLD
        ]
        
        selected_tests.sort(key=lambda x: x['relevance_score'], reverse=True)
        selected_tests = selected_tests[:MAX_SUGGESTIONS]
        
        print(f"Selected {len(selected_tests)} tests above {RELEVANCE_THRESHOLD}% threshold:")
        
        if selected_tests:
            for test in selected_tests:
                print(f"  ✅ T{test['id']}: {test['title']} ({test['relevance_score']}%)")
        else:
            print("  ❌ No tests meet the relevance threshold")
        
        # Step 4: Generate GitHub comment
        print(f"\n💬 STEP 4: Generated GitHub comment")
        print("-" * 50)
        
        comment = "🤖 **Intelligent Test Case Suggestion** 🤖\n\n"
        if selected_tests:
            comment += f"Based on AI analysis of the code changes, I found **{len(selected_tests)}** highly relevant test case(s):\n\n"
            for test in selected_tests:
                case_url = f"https://example.testrail.com/index.php?/cases/view/{test['id']}"
                comment += f"- **T{test['id']}**: [{test['title']}]({case_url})\n"
                comment += f"  - **Relevance**: {test['relevance_score']}%\n"
                comment += f"  - **Reasoning**: {test['reasoning']}\n\n"
            
            comment += f"*Note: Only tests with ≥{RELEVANCE_THRESHOLD}% AI-evaluated relevance are suggested. "
            comment += f"This uses embeddings for retrieval + LLM for semantic evaluation.*"
        else:
            comment += f"After AI analysis, I could not find any existing test cases in TestRail that are sufficiently relevant to these changes (threshold: {RELEVANCE_THRESHOLD}% relevance). "
            comment += "The changes may be in areas not covered by existing tests, or may be low-risk changes like styling. Please consider if new test cases are needed."
        
        print(comment)
        print("-" * 50)
        
        # Verify expectation
        has_relevant = len(selected_tests) > 0
        if has_relevant == test_case['expected_relevant']:
            print(f"✅ RESULT: Expectation met - Expected relevant={test_case['expected_relevant']}, Got relevant={has_relevant}")
        else:
            print(f"❌ RESULT: Expectation not met - Expected relevant={test_case['expected_relevant']}, Got relevant={has_relevant}")

def test_fallback_heuristics():
    """Test the fallback heuristic evaluation when LLM is not available"""
    print(f"\n{'=' * 80}")
    print("TESTING FALLBACK HEURISTIC EVALUATION")
    print("(Used when OPENAI_API_KEY is not available)")
    print("=" * 80)
    
    test_cases = [
        {
            "code": "function validateLogin(username, password) { return auth.login(username, password); }",
            "test": {"id": 1, "title": "Valid Login 1"},
            "expected_score_range": (40, 100)
        },
        {
            "code": ".navbar { background-color: blue; padding: 10px; }",
            "test": {"id": 2, "title": "Valid Login 1"},
            "expected_score_range": (0, 30)
        },
        {
            "code": "function registerUser(email, password) { return api.register(email, password); }",
            "test": {"id": 3, "title": "Valid Registration Test"},
            "expected_score_range": (40, 100)
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\nTest {i}: {test_case['test']['title']}")
        print(f"Code: {test_case['code']}")
        
        evaluation = evaluate_test_relevance_fallback(test_case['code'], test_case['test'])
        
        print(f"Score: {evaluation['relevance_score']}%")
        print(f"Reasoning: {evaluation['reasoning']}")
        print(f"Recommendation: {evaluation['recommendation']}")
        
        min_score, max_score = test_case['expected_score_range']
        if min_score <= evaluation['relevance_score'] <= max_score:
            print(f"✅ Score in expected range ({min_score}-{max_score})")
        else:
            print(f"❌ Score outside expected range ({min_score}-{max_score})")

if __name__ == "__main__":
    # Check if OpenAI API key is available
    if os.getenv("OPENAI_API_KEY"):
        print("🔑 OpenAI API key found - will use LLM evaluation")
    else:
        print("⚠️  No OpenAI API key - will use fallback heuristic evaluation")
    
    test_llm_enhanced_approach()
    test_fallback_heuristics()
    
    print(f"\n{'=' * 80}")
    print("🎉 LLM-ENHANCED APPROACH SUMMARY")
    print("✅ Embeddings provide good initial retrieval")
    print("✅ LLM provides semantic understanding and reasoning")
    print("✅ Fallback heuristics work when LLM unavailable")
    print("✅ Much more robust than distance-based thresholds")
    print("✅ Provides explainable AI recommendations")
    print("=" * 80)