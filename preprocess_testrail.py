# ---
# File: preprocess_testrail.py
# This is a one-time setup script to "teach" the agent about your tests.
# --- FIX for sqlite3 version issue with ChromaDB ---
try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass

import os
import chromadb
from sentence_transformers import SentenceTransformer
from testrail_api import TestRailAPI

# --- Configuration ---
# Load from environment variables for security
#TESTRAIL_URL = os.getenv("TESTRAIL_URL")
#TESTRAIL_USER = os.getenv("TESTRAIL_USER")
#TESTRAIL_PASSWORD = os.getenv("TESTRAIL_PASSWORD")
TESTRAIL_URL="https://myrodeotwo.testrail.io"
TESTRAIL_USER="santavant2@gmail.com"
TESTRAIL_PASSWORD="pTRh0krr5AtiA7rRj2.H-h8lkrI5nNCfI5SdchROr"
TESTRAIL_PROJECT_ID="1"
PROJECT_ID = 1 # IMPORTANT: Change to your TestRail project's ID

DB_PATH = "testrail_db"
COLLECTION_NAME = "testrail_embeddings"
MODEL_NAME = 'all-MiniLM-L6-v2'

def create_testrail_embeddings():
    """Fetches test cases from TestRail and stores their embeddings."""
    print("TESTRAIL_URL =", os.getenv("TESTRAIL_URL"))
    print("TESTRAIL_USER =", os.getenv("TESTRAIL_USER"))
    if not all([TESTRAIL_URL, TESTRAIL_USER, TESTRAIL_PASSWORD]):
        print("Error: Please set TESTRAIL_URL, TESTRAIL_USER, and TESTRAIL_PASSWORD environment variables.")
        return

    print("Connecting to TestRail...")
    api = TestRailAPI(TESTRAIL_URL, TESTRAIL_USER, TESTRAIL_PASSWORD)
    #cases = api.cases.get_cases(project_id=PROJECT_ID)
    #print(f"Found {len(cases)} test cases in Project ID {PROJECT_ID}.")

    response = api.cases.get_cases(project_id=PROJECT_ID)
    cases_list = response.get('cases', []) # Safely get the list of cases
    
    if not cases_list:
        print("No test cases found in the response from TestRail.")
        return

    print(f"Found {len(cases_list)} test cases in Project ID {PROJECT_ID}.")

    model = SentenceTransformer(MODEL_NAME)
    client = chromadb.PersistentClient(path=DB_PATH)
    collection = client.get_or_create_collection(name=COLLECTION_NAME, 
                                                 metadata={"hnsw:space": "cosine"})

    documents, metadatas, ids = [], [], []
    for case in cases_list:
        # *** FIX: Handle cases where 'custom_steps_separated' might be None ***
        steps = case.get('custom_steps_separated') # Can be None if no steps exist
        steps_text = "" # Default to an empty string
        
        # Only process steps if the 'steps' variable is a list (not None)
        if steps:
            # Safely get content from each step, providing an empty string if 'content' is missing
            steps_text = " ".join([step.get('content', '') for step in steps if step])
        
        # Safely get the title
        title = case.get('title', 'No Title Provided')
        full_text = f"Title: {title}. Steps: {steps_text}"
        
        documents.append(full_text)
        metadatas.append({"case_id": case.get('id'), "title": title})
        ids.append(str(case.get('id')))

    print(f"Generating embeddings for {len(documents)} test cases...")
    embeddings = model.encode(documents).tolist()

    print(f"Adding {len(ids)} documents to collection '{COLLECTION_NAME}'...")
    collection.add(embeddings=embeddings, documents=documents, metadatas=metadatas, ids=ids)
    print("✅ TestRail database pre-processing complete!")

if __name__ == "__main__":
    create_testrail_embeddings()