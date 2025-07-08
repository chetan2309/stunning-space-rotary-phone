# ---
# File: test_selector.py
# This file contains the core logic for analyzing code and finding tests.

import chromadb
from github import Github
from sentence_transformers import SentenceTransformer

# --- Configuration ---
DB_PATH = "testrail_db" # The local directory where the DB is stored
COLLECTION_NAME = "testrail_embeddings"
MODEL_NAME = 'all-MiniLM-L6-v2'
NUM_TESTS_TO_SUGGEST = 3

def analyze_pr_and_get_suggestions(repo_name, pr_number, github_token, testrail_url):
    """
    Analyzes a PR, finds relevant tests, and posts a comment.
    """
    g = Github(github_token)
    repo = g.get_repo(repo_name)
    pr = repo.get_pull(pr_number)
    
    # 1. Get code changes from the PR.
    files = pr.get_files()
    code_text = f"Pull Request Title: {pr.title}\n\n"
    relevant_files_changed = False
    for file in files:
        if file.filename.endswith(('.js', '.jsx', '.vue', '.ts', '.tsx')):
            relevant_files_changed = True
            code_text += f"--- File: {file.filename} ---\n{file.patch}\n\n"

    if not relevant_files_changed:
        return "No relevant front-end code changes found in this PR."

    # 2. Find relevant tests in our vector database.
    print("Loading model and connecting to DB...")
    model = SentenceTransformer(MODEL_NAME)
    client = chromadb.PersistentClient(path=DB_PATH)
    collection = client.get_collection(name=COLLECTION_NAME)
    
    print("Analyzing code and querying for tests...")
    query_embedding = model.encode(code_text).tolist()
    results = collection.query(query_embeddings=[query_embedding], n_results=NUM_TESTS_TO_SUGGEST)
    
    selected_tests = []
    if results and results.get('metadatas') and results['metadatas'][0]:
        selected_tests = [{"id": meta['case_id'], "title": meta['title']} for meta in results['metadatas'][0]]

    # 3. Format the comment to be posted on GitHub.
    comment = "🤖 **Intelligent Test Case Suggestion** 🤖\n\n"
    if selected_tests:
        comment += "Based on the code changes, the QA team should consider manually running:\n\n"
        for test in selected_tests:
            case_url = f"{testrail_url}index.php?/cases/view/{test['id']}"
            comment += f"- **T{test['id']}**: [{test['title']}]({case_url})\n"
    else:
        comment += "I could not find any existing test cases in TestRail that seem relevant to these changes. Please consider if a new test case is needed."
    
    # 4. Post the comment back to the PR.
    print("Posting comment to PR...")
    pr.create_issue_comment(comment)
    
    return "Suggestion comment posted successfully."