#!/usr/bin/env python3
"""
Final demonstration of the fixed LLM-enhanced test suggestion system.
This shows the complete working solution that addresses the original similarity filtering bug.
"""

try:
    import pysqlite3
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass

import chromadb
from sentence_transformers import SentenceTransformer
from test_selector import evaluate_test_relevance_with_llm

# Configuration matching the fixed system
DB_PATH = "testrail_db"
COLLECTION_NAME = "testrail_embeddings"
MODEL_NAME = 'all-MiniLM-L6-v2'
NUM_CANDIDATES = 10
RELEVANCE_THRESHOLD = 35
MAX_SUGGESTIONS = 3

def simulate_complete_system(code_changes, testrail_url="https://example.testrail.com/"):
    """
    Simulate the complete fixed system: Embeddings for Retrieval + LLM for Evaluation
    """
    print("🔄 Starting intelligent test case analysis...")
    
    # Step 1: Embedding-based retrieval
    print(f"📊 Retrieving {NUM_CANDIDATES} candidate tests using embeddings...")
    model = SentenceTransformer(MODEL_NAME)
    client = chromadb.PersistentClient(path=DB_PATH)
    collection = client.get_collection(name=COLLECTION_NAME)
    
    query_embedding = model.encode(code_changes).tolist()
    results = collection.query(query_embeddings=[query_embedding], n_results=NUM_CANDIDATES)
    
    # Step 2: LLM evaluation of candidates
    print("🤖 Evaluating test relevance with AI...")
    evaluated_tests = []
    
    if results and results.get('metadatas') and results['metadatas'][0]:
        for i, meta in enumerate(results['metadatas'][0]):
            test_case = {
                'id': meta['case_id'],
                'title': meta['title'],
                'description': meta.get('description', ''),
                'embedding_rank': i + 1,
                'embedding_distance': results['distances'][0][i] if results.get('distances') else None
            }
            
            evaluation = evaluate_test_relevance_with_llm(code_changes, test_case)
            
            test_case.update({
                'relevance_score': evaluation['relevance_score'],
                'reasoning': evaluation['reasoning'],
                'recommendation': evaluation['recommendation']
            })
            
            evaluated_tests.append(test_case)
    
    # Step 3: Filter by LLM relevance threshold
    selected_tests = [
        test for test in evaluated_tests 
        if test['relevance_score'] >= RELEVANCE_THRESHOLD
    ]
    
    # Sort by relevance score and limit results
    selected_tests.sort(key=lambda x: x['relevance_score'], reverse=True)
    selected_tests = selected_tests[:MAX_SUGGESTIONS]
    
    print(f"✅ Selected {len(selected_tests)} tests above {RELEVANCE_THRESHOLD}% relevance threshold")
    
    # Step 4: Generate GitHub comment
    comment = "🤖 **Intelligent Test Case Suggestion** 🤖\n\n"
    if selected_tests:
        comment += f"Based on AI analysis of the code changes, I found **{len(selected_tests)}** highly relevant test case(s):\n\n"
        for test in selected_tests:
            case_url = f"{testrail_url}index.php?/cases/view/{test['id']}"
            comment += f"- **T{test['id']}**: [{test['title']}]({case_url})\n"
            comment += f"  - **Relevance**: {test['relevance_score']}%\n"
            comment += f"  - **Reasoning**: {test['reasoning']}\n\n"
        
        comment += f"*Note: Only tests with ≥{RELEVANCE_THRESHOLD}% AI-evaluated relevance are suggested. "
        comment += f"This uses embeddings for retrieval + LLM for semantic evaluation.*"
    else:
        comment += f"After AI analysis, I could not find any existing test cases in TestRail that are sufficiently relevant to these changes (threshold: {RELEVANCE_THRESHOLD}% relevance). "
        comment += "The changes may be in areas not covered by existing tests, or may be low-risk changes like styling. Please consider if new test cases are needed."
    
    return selected_tests, comment

def main():
    print("=" * 90)
    print("🎯 DEMONSTRATION: FIXED LLM-ENHANCED TEST SUGGESTION SYSTEM")
    print("=" * 90)
    print("Problem: Original system used brittle distance thresholds")
    print("Solution: Embeddings for Retrieval + LLM for Semantic Evaluation")
    print("=" * 90)
    
    test_scenarios = [
        {
            "name": "🔐 Authentication Code Changes (Should find relevant tests)",
            "code": """Pull Request Title: Improve login validation and error handling

--- File: auth.js ---
function validateLogin(username, password) {
    // Enhanced validation with better error messages
    if (!username || username.trim().length === 0) {
        throw new ValidationError('Username is required and cannot be empty');
    }
    
    if (!password || password.length < 8) {
        throw new ValidationError('Password must be at least 8 characters long');
    }
    
    // Check for common weak passwords
    const weakPasswords = ['password', '12345678', 'qwerty123'];
    if (weakPasswords.includes(password.toLowerCase())) {
        throw new ValidationError('Password is too weak. Please choose a stronger password.');
    }
    
    return authenticateUser(username, password);
}

function handleLoginError(error) {
    const errorDisplay = document.getElementById('login-error');
    errorDisplay.textContent = error.message;
    errorDisplay.style.display = 'block';
    
    // Log failed login attempts for security
    logSecurityEvent('failed_login', { username: error.username, timestamp: new Date() });
}"""
        },
        {
            "name": "🎨 CSS Styling Changes (Should find NO relevant tests)",
            "code": """Pull Request Title: Update button styles and color scheme

--- File: buttons.css ---
.btn-primary {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border: none;
    border-radius: 8px;
    padding: 12px 24px;
    font-weight: 600;
    color: white;
    transition: all 0.3s ease;
    box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
}

.btn-primary:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
}

.btn-secondary {
    background: transparent;
    border: 2px solid #667eea;
    color: #667eea;
    border-radius: 8px;
    padding: 10px 22px;
    font-weight: 600;
    transition: all 0.3s ease;
}

.btn-secondary:hover {
    background: #667eea;
    color: white;
}"""
        },
        {
            "name": "🗄️ Database Schema Changes (Should find NO relevant tests)",
            "code": """Pull Request Title: Add audit logging table and indexes

--- File: 002_audit_logging.sql ---
-- Create audit log table for tracking user actions
CREATE TABLE audit_logs (
    id BIGSERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id) ON DELETE SET NULL,
    action VARCHAR(100) NOT NULL,
    resource_type VARCHAR(50) NOT NULL,
    resource_id VARCHAR(100),
    old_values JSONB,
    new_values JSONB,
    ip_address INET,
    user_agent TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for efficient querying
CREATE INDEX idx_audit_logs_user_id ON audit_logs(user_id);
CREATE INDEX idx_audit_logs_action ON audit_logs(action);
CREATE INDEX idx_audit_logs_resource ON audit_logs(resource_type, resource_id);
CREATE INDEX idx_audit_logs_created_at ON audit_logs(created_at);

-- Partition table by month for better performance
CREATE TABLE audit_logs_y2024m01 PARTITION OF audit_logs
    FOR VALUES FROM ('2024-01-01') TO ('2024-02-01');

-- Function to automatically create monthly partitions
CREATE OR REPLACE FUNCTION create_monthly_partition(table_name text, start_date date)
RETURNS void AS $$
DECLARE
    partition_name text;
    end_date date;
BEGIN
    partition_name := table_name || '_y' || to_char(start_date, 'YYYY') || 'm' || to_char(start_date, 'MM');
    end_date := start_date + interval '1 month';
    
    EXECUTE format('CREATE TABLE %I PARTITION OF %I FOR VALUES FROM (%L) TO (%L)',
                   partition_name, table_name, start_date, end_date);
END;
$$ LANGUAGE plpgsql;"""
        }
    ]
    
    for i, scenario in enumerate(test_scenarios, 1):
        print(f"\n{'🔹' * 30}")
        print(f"SCENARIO {i}: {scenario['name']}")
        print(f"{'🔹' * 30}")
        
        selected_tests, comment = simulate_complete_system(scenario['code'])
        
        print(f"\n📋 RESULTS:")
        if selected_tests:
            print(f"✅ Found {len(selected_tests)} relevant test(s):")
            for test in selected_tests:
                print(f"   • T{test['id']}: {test['title']} ({test['relevance_score']}% relevance)")
                print(f"     Reasoning: {test['reasoning']}")
        else:
            print("❌ No relevant tests found (this may be correct for non-functional changes)")
        
        print(f"\n💬 GITHUB COMMENT PREVIEW:")
        print("-" * 60)
        print(comment)
        print("-" * 60)
    
    print(f"\n{'=' * 90}")
    print("🎉 SUMMARY: BUG FIXED WITH LLM-ENHANCED APPROACH")
    print("=" * 90)
    print("✅ BEFORE: Brittle distance thresholds always returned irrelevant tests")
    print("✅ AFTER:  AI semantic evaluation only suggests truly relevant tests")
    print("✅ ROBUST: Works with or without OpenAI API (fallback heuristics)")
    print("✅ EXPLAINABLE: Provides reasoning for each suggestion")
    print("✅ MAINTAINABLE: No more guessing distance threshold values")
    print("✅ SCALABLE: Can easily add more sophisticated LLM prompts")
    print("=" * 90)
    
    print(f"\n🔧 TECHNICAL ARCHITECTURE:")
    print("1. 📊 Embeddings retrieve top N candidates from vector database")
    print("2. 🤖 LLM evaluates semantic relevance of each candidate")
    print("3. 🎯 Filter by LLM confidence scores (not distance thresholds)")
    print("4. 💬 Generate explainable recommendations for GitHub")
    print("5. 🛡️  Fallback heuristics when LLM unavailable")

if __name__ == "__main__":
    main()