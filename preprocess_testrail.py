# ---
# File: preprocess_testrail.py
# This is a one-time setup script to "teach" the agent about your tests.

import os
import chromadb
from sentence_transformers import SentenceTransformer
from testrail_api import TestRailAPI

# --- Configuration ---
# Load from environment variables for security
TESTRAIL_URL = os.getenv("TESTRAIL_URL")
TESTRAIL_USER = os.getenv("TESTRAIL_USER")
TESTRAIL_PASSWORD = os.getenv("TESTRAIL_PASSWORD")
PROJECT_ID = 1 # IMPORTANT: Change to your TestRail project's ID

DB_PATH = "testrail_db"
COLLECTION_NAME = "testrail_embeddings"
MODEL_NAME = 'all-MiniLM-L6-v2'

def create_testrail_embeddings():
    """Fetches test cases from TestRail and stores their embeddings."""
    if not all([TESTRAIL_URL, TESTRAIL_USER, TESTRAIL_PASSWORD]):
        print("Error: Please set TESTRAIL_URL, TESTRAIL_USER, and TESTRAIL_PASSWORD environment variables.")
        return

    print("Connecting to TestRail...")
    api = TestRailAPI(TESTRAIL_URL, TESTRAIL_USER, TESTRAIL_PASSWORD)
    cases = api.cases.get_cases(project_id=PROJECT_ID)
    print(f"Found {len(cases)} test cases in Project ID {PROJECT_ID}.")

    model = SentenceTransformer(MODEL_NAME)
    client = chromadb.PersistentClient(path=DB_PATH)
    collection = client.get_or_create_collection(name=COLLECTION_NAME)

    documents, metadatas, ids = [], [], []
    for case in cases:
        steps = case.get('custom_steps_separated', [])
        steps_text = " ".join([step['content'] for step in steps if 'content' in step])
        full_text = f"Title: {case['title']}. Steps: {steps_text}"
        
        documents.append(full_text)
        metadatas.append({"case_id": case['id'], "title": case['title']})
        ids.append(str(case['id']))

    print(f"Generating embeddings for {len(documents)} test cases...")
    embeddings = model.encode(documents).tolist()

    print(f"Adding {len(ids)} documents to collection '{COLLECTION_NAME}'...")
    collection.add(embeddings=embeddings, documents=documents, metadatas=metadatas, ids=ids)
    print("✅ TestRail database pre-processing complete!")

if __name__ == "__main__":
    create_testrail_embeddings()