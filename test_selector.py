# ---
# File: test_selector.py
# This file contains the core logic for analyzing code and finding tests.
# --- FIX for sqlite3 version issue with ChromaDB ---
try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass

import chromadb
from github import Github
from sentence_transformers import SentenceTransformer
import openai
import os
import json

# --- Configuration ---
DB_PATH = "testrail_db" # The local directory where the DB is stored
COLLECTION_NAME = "testrail_embeddings"
MODEL_NAME = 'all-MiniLM-L6-v2'
NUM_CANDIDATES = 10  # Retrieve more candidates for LLM evaluation
RELEVANCE_THRESHOLD = 35  # LLM relevance score threshold (0-100) - conservative but practical
MAX_SUGGESTIONS = 3  # Maximum number of test suggestions to return

AZURE_OPENAI_KEY="5366f9c0121f4852afeb69388c2aff3a"
AZURE_OPENAI_ENDPOINT="https://agtech-llm-openai.openai.azure.com/"
AZURE_OPENAI_VERSION="2024-12-01-preview"
AZURE_OPENAI_DEPLOYMENT="o3"

def evaluate_test_relevance_with_llm(code_changes, test_case):
    """
    Use LLM to evaluate if a test case is relevant to the code changes.
    Returns a relevance score (0-100) and reasoning.
    """
    try:
        # Set up Azure OpenAI client (fallback to a simple heuristic if no API key)
        azure_openai_key = AZURE_OPENAI_KEY
        azure_openai_endpoint = AZURE_OPENAI_ENDPOINT
        azure_openai_version = AZURE_OPENAI_VERSION
        
        if not azure_openai_key or not azure_openai_endpoint:
            print("Warning: No AZURE_OPENAI_KEY or AZURE_OPENAI_ENDPOINT found, using fallback heuristic evaluation")
            return evaluate_test_relevance_fallback(code_changes, test_case)
        
        client = openai.AzureOpenAI(
            api_key=azure_openai_key,
            api_version=azure_openai_version,
            azure_endpoint=azure_openai_endpoint
        )
        
        prompt = f"""You are an expert QA engineer tasked with evaluating the relevance of test cases in relation to code changes in a pull request. Your goal is to determine whether a given test case should be included or excluded based on its relevance to the code changes.
        First, review the following information about the pull request:

First, review the following information about the pull request and the code changes:
CODE CHANGES:
{code_changes}

TEST CASE:
ID: {test_case['id']}
Test case description: {test_case['title']}

Steps:
{test_case['steps']}

Expected Result:
{test_case['expected_result']}

TASK: Evaluate if this test case is relevant to the code changes above.

Consider:
- Does the test case cover functionality that could be affected by these changes?
- Are there semantic relationships between the code and test areas?
- Would running this test help validate the changes work correctly?

Respond with a JSON object containing:
- "relevance_score": integer from 0-100 (0=completely irrelevant, 100=highly relevant)
- "reasoning": string explaining your evaluation
- "recommendation": "include" or "exclude"

Example response:
{{"relevance_score": 85, "reasoning": "This login test is highly relevant because the code changes modify authentication validation logic", "recommendation": "include"}}
"""

        # Use your Azure OpenAI deployment name here
        deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT", AZURE_OPENAI_DEPLOYMENT)
        #print("Prompt for the test case is ", prompt)
        response = client.chat.completions.create(
            model=deployment_name,  # This should match your Azure deployment name
            messages=[{"role": "user", "content": prompt}],
            max_completion_tokens=300
        )
        print(response.choices[0].message.content)
        result = json.loads(response.choices[0].message.content)
        print('-----------------------------------------')
        print(result)
        return result
        
    except Exception as e:
        print("Eval failed", e)
        #print(f"LLM evaluation failed: {e}, using fallback")
        #return evaluate_test_relevance_fallback(code_changes, test_case)

def evaluate_test_relevance_fallback(code_changes, test_case):
    """
    Fallback heuristic evaluation when LLM is not available.
    """
    code_lower = code_changes.lower()
    title_lower = test_case['title'].lower()
    
    # Simple keyword matching heuristics
    relevance_score = 0
    reasoning_parts = []
    
    # Authentication/Login related
    auth_keywords = ['login', 'auth', 'password', 'username', 'signin', 'credential', 'validate']
    if any(keyword in code_lower for keyword in auth_keywords):
        if any(keyword in title_lower for keyword in auth_keywords):
            relevance_score += 50
            reasoning_parts.append("authentication-related code and test")
    
    # Registration related
    reg_keywords = ['register', 'signup', 'registration', 'create account']
    if any(keyword in code_lower for keyword in reg_keywords):
        if any(keyword in title_lower for keyword in reg_keywords):
            relevance_score += 40
            reasoning_parts.append("registration-related code and test")
    
    # UI/Frontend related
    ui_keywords = ['button', 'form', 'input', 'click', 'submit', 'validate']
    if any(keyword in code_lower for keyword in ui_keywords):
        if any(keyword in title_lower for keyword in ui_keywords):
            relevance_score += 30
            reasoning_parts.append("UI interaction code and test")
    
    # Penalize irrelevant combinations
    css_keywords = ['css', 'style', 'color', 'font', 'padding', 'margin']
    if any(keyword in code_lower for keyword in css_keywords):
        if not any(keyword in title_lower for keyword in ['style', 'ui', 'display']):
            relevance_score = max(0, relevance_score - 30)
            reasoning_parts.append("CSS changes unlikely to affect functional tests")
    
    reasoning = f"Heuristic evaluation based on: {', '.join(reasoning_parts) if reasoning_parts else 'no clear keyword matches'}"
    recommendation = "include" if relevance_score >= RELEVANCE_THRESHOLD else "exclude"
    
    return {
        "relevance_score": min(100, relevance_score),
        "reasoning": reasoning,
        "recommendation": recommendation
    }

def analyze_pr_and_get_suggestions(repo_name, pr_number, github_token, testrail_url):
    """
    Analyzes a PR, finds relevant tests, and posts a comment.
    """
    g = Github(github_token)
    repo = g.get_repo(repo_name)
    pr = repo.get_pull(pr_number)
    
    # 1. Get code changes from the PR.
    files = pr.get_files()
    code_text = f"Pull Request Title: {pr.title}\n\n Pull Request description: {pr.body}"
    relevant_files_changed = False
    for file in files:
        if file.filename.endswith(('.js', '.jsx', '.vue', '.ts', '.tsx')):
            relevant_files_changed = True
            code_text += f"--- File: {file.filename} ---\n{file.patch}\n\n"

    if not relevant_files_changed:
        return "No relevant front-end code changes found in this PR."

    # 2. Find candidate tests using embeddings (retrieval phase)
    print("Loading model and connecting to DB...")
    model = SentenceTransformer(MODEL_NAME)
    client = chromadb.PersistentClient(path=DB_PATH)
    collection = client.get_collection(name=COLLECTION_NAME)
    
    print(f"Retrieving {NUM_CANDIDATES} candidate tests using embeddings...")
    query_embedding = model.encode(code_text).tolist()
    results = collection.query(query_embeddings=[query_embedding], n_results=NUM_CANDIDATES)
    #print("Results are these.......................")
    #print(results)
    
    # 3. Use LLM to evaluate relevance of each candidate (evaluation phase)
    print("Evaluating test relevance with LLM...")
    evaluated_tests = []
    
    if results and results.get('metadatas') and results['metadatas'][0]:
        for i, meta in enumerate(results['metadatas'][0]):
            test_case = {
                'id': meta['case_id'],
                'title': meta['title'],
                'steps': meta.get('custom_steps', ''),
                "expected_result": meta.get('custom_expected', ''),
                'embedding_rank': i + 1,
                'embedding_distance': results['distances'][0][i] if results.get('distances') else None
            }
            
            print(f"  Evaluating C{test_case['id']}: {test_case['title']}")
            evaluation = evaluate_test_relevance_with_llm(code_text, test_case)
            test_case.update({
                'relevance_score': evaluation['relevance_score'],
                'reasoning': evaluation['reasoning'],
                'recommendation': evaluation['recommendation']
            })
            
            evaluated_tests.append(test_case)
    
    # 4. Filter by LLM relevance threshold and sort by relevance score
    selected_tests = [
        test for test in evaluated_tests 
        if test['relevance_score'] >= RELEVANCE_THRESHOLD
    ]
    
    # Sort by relevance score (highest first) and limit results
    selected_tests.sort(key=lambda x: x['relevance_score'], reverse=True)
    selected_tests = selected_tests[:MAX_SUGGESTIONS]
    
    print(f"Selected {len(selected_tests)} tests above {RELEVANCE_THRESHOLD}% relevance threshold")

    # 5. Format the comment to be posted on GitHub.
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
    
    # 4. Post the comment back to the PR.
    print("Posting comment to PR...")
    pr.create_issue_comment(comment)
    
    return "Suggestion comment posted successfully."