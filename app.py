# File: app.py
# This is the main web server that listens for requests from GitHub.

# --- FIX for sqlite3 version issue with ChromaDB ---
try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass
import os
from flask import Flask, request, jsonify
from test_selector import analyze_pr_and_get_suggestions

app = Flask(__name__)

# --- Configuration ---
# These will be loaded from the hosting environment (Render).
#SECRET_TOKEN = os.getenv("SECRET_TOKEN")
SECRET_TOKEN = "my-super-secret-password-12345"
TESTRAIL_URL = os.getenv("TESTRAIL_URL")

@app.route('/')
def home():
    """A simple welcome message to confirm the service is running."""
    return "Intelligent Test Runner Service is alive!", 200

@app.route('/analyze', methods=['POST'])
def analyze_webhook():
    """Endpoint to receive webhooks from a GitHub Action."""
    
    # 1. Security Check: Ensure the request is from our GitHub Action.
    request_token = request.headers.get('X-Secret-Token')
    print("request_token is = ", request_token)
    print("SECRET_TOKEN is = ", SECRET_TOKEN)
    if not request_token or request_token != SECRET_TOKEN:
        print("Unauthorized attempt blocked.")
        return jsonify({"error": "Unauthorized"}), 403

    # 2. Get data from the webhook payload.
    data = request.json
    repo_name = data.get('repo')
    pr_number = data.get('pr_number')
    github_token = data.get('github_token')

    if not all([repo_name, pr_number, github_token]):
        return jsonify({"error": "Missing required data: repo, pr_number, or github_token"}), 400

    print(f"Received request for repo: {repo_name}, PR: #{pr_number}")
    
    # 3. Run the core analysis logic.
    try:
        result_message = analyze_pr_and_get_suggestions(repo_name, pr_number, github_token, TESTRAIL_URL)
        print(f"Analysis complete: {result_message}")
        return jsonify({"status": "success", "message": result_message}), 200
    except Exception as e:
        print(f"An error occurred during analysis: {e}")
        return jsonify({"error": f"An internal error occurred: {e}"}), 500

if __name__ == '__main__':
    # The preprocess_testrail.py script must be run once before starting.
    # The hosting service (Render) will use a Gunicorn server, not this.
    app.run(host='0.0.0.0', port=int(os.getenv('PORT', 8080)))