# 🔧 Azure OpenAI Setup Guide

## Required Environment Variables

Set these environment variables to use Azure OpenAI:

```bash
export AZURE_OPENAI_KEY="your-azure-openai-key"
export AZURE_OPENAI_ENDPOINT="https://your-resource-name.openai.azure.com/"
export AZURE_OPENAI_VERSION="2024-02-15-preview"
export AZURE_OPENAI_DEPLOYMENT="gpt-35-turbo"  # Your deployment name
```

## Step-by-Step Setup

### 1. Get Your Azure OpenAI Credentials

From your Azure Portal:
- **Key**: Go to your Azure OpenAI resource → Keys and Endpoint → Copy Key 1
- **Endpoint**: Copy the endpoint URL (e.g., `https://your-resource-name.openai.azure.com/`)
- **Deployment**: Note your model deployment name (e.g., `gpt-35-turbo`, `gpt-4`)

### 2. Set Environment Variables

**Option A: Using .env file (recommended)**
```bash
cp .env.example .env
# Edit .env with your actual values
```

**Option B: Export directly**
```bash
export AZURE_OPENAI_KEY="your-actual-key"
export AZURE_OPENAI_ENDPOINT="https://your-resource-name.openai.azure.com/"
export AZURE_OPENAI_DEPLOYMENT="your-deployment-name"
```

### 3. Test the Setup

```bash
python3 -c "
import os
from test_selector import evaluate_test_relevance_with_llm

# Test with sample data
code = 'function validateLogin(username, password) { return auth.login(username, password); }'
test = {'id': 1, 'title': 'Valid Login Test', 'description': 'Test user login functionality'}

result = evaluate_test_relevance_with_llm(code, test)
print(f'Result: {result}')
"
```

## Common Deployment Names

Your Azure deployment name might be:
- `gpt-35-turbo` (for GPT-3.5 Turbo)
- `gpt-4` (for GPT-4)
- `gpt-4-turbo` (for GPT-4 Turbo)
- Or whatever custom name you gave your deployment

## Troubleshooting

### Error: "No AZURE_OPENAI_KEY found"
- Check that environment variables are set correctly
- Verify the key is valid and not expired

### Error: "The API deployment for this resource does not exist"
- Check your `AZURE_OPENAI_DEPLOYMENT` matches your actual deployment name
- Verify the deployment is active in Azure Portal

### Error: "Invalid API version"
- Try different API versions: `2024-02-15-preview`, `2023-12-01-preview`
- Check Azure OpenAI documentation for latest versions

## Fallback Behavior

If Azure OpenAI is not configured, the system automatically falls back to heuristic evaluation:
- Still works without LLM
- Uses keyword matching for relevance scoring
- Less accurate but functional

## Cost Optimization

To minimize Azure OpenAI costs:
- Adjust `NUM_CANDIDATES` (default: 10) to evaluate fewer tests
- Increase `RELEVANCE_THRESHOLD` (default: 35) to be more selective
- Use GPT-3.5 Turbo instead of GPT-4 for lower costs

```python
# In test_selector.py, adjust these values:
NUM_CANDIDATES = 5        # Evaluate fewer candidates
RELEVANCE_THRESHOLD = 50  # Higher threshold = fewer suggestions
```