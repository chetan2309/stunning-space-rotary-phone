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

# --- Configuration ---
DB_PATH = "testrail_db" # The local directory where the DB is stored
COLLECTION_NAME = "testrail_embeddings"
MODEL_NAME = 'all-MiniLM-L6-v2'
NUM_TESTS_TO_SUGGEST = 10

def analyze_pr_and_get_suggestions(repo_name, pr_number, github_token, testrail_url):
    """
    Analyzes a PR, finds relevant tests, and posts a comment.
    """
    g = Github(github_token)
    repo = g.get_repo(repo_name)
    pr = repo.get_pull(pr_number)
    print("Entire pr body is ", pr)
    print("Entire pr body via repo is ", repo)
    
    # 1. Get code changes from the PR.
    # Include the PR body for more context.
    pr_body = pr.body if pr.body else ""
    code_text = f"Pull Request Title: {pr.title}\n\nPull Request Description: {pr_body}\n\n"
    print("Code text is", code_text)

    files = pr.get_files()
    #code_text = f"Pull Request Title: {pr.title}\n\n"
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
    print(results)
    
    selected_tests = []
    SIMILARITY_THRESHOLD = 0.7
    #if results and results.get('metadatas') and results['metadatas'][0]:
    #    selected_tests = [{"id": meta['case_id'], "title": meta['title']} for meta in results['metadatas'][0]]
    # Filter results by distance threshold
    #if results and results.get('ids')[0]:
    #    for i, distance in enumerate(results['distances'][0]):
    #        if distance < DISTANCE_THRESHOLD:
    #            print("Distance is lless than threshold", distance)
    #            meta = results['metadatas'][0][i]
    #            selected_tests.append({"id": meta['case_id'], "title": meta['title']})
    
    # --- UPDATED FILTERING LOGIC ---
    # The 'distances' field will now contain cosine similarity scores.
    # We want scores *above* the threshold.
    if results and results.get('ids')[0]:
        # ChromaDB still calls it 'distances', but it's now a similarity score.
        scores = results['distances'][0]
        for i, score in enumerate(scores):
            if score > SIMILARITY_THRESHOLD: # Higher is better!
                meta = results['metadatas'][0][i]
                selected_tests.append({"id": meta['case_id'], "title": meta['title']})
    # --- END UPDATED LOGIC ---

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