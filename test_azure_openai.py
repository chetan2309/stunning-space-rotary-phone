#!/usr/bin/env python3
"""
Test Azure OpenAI integration for the LLM-enhanced test suggestion system.
"""

import os
from test_selector import evaluate_test_relevance_with_llm

def test_azure_openai_connection():
    """Test if Azure OpenAI is properly configured and working"""
    
    print("🔧 Testing Azure OpenAI Configuration")
    print("=" * 50)
    
    # Check environment variables
    required_vars = ["AZURE_OPENAI_KEY", "AZURE_OPENAI_ENDPOINT"]
    optional_vars = ["AZURE_OPENAI_VERSION", "AZURE_OPENAI_DEPLOYMENT"]
    
    print("📋 Environment Variables:")
    for var in required_vars:
        value = os.getenv(var)
        if value:
            # Mask the key for security
            display_value = value[:8] + "..." if var == "AZURE_OPENAI_KEY" else value
            print(f"  ✅ {var}: {display_value}")
        else:
            print(f"  ❌ {var}: Not set")
            return False
    
    for var in optional_vars:
        value = os.getenv(var)
        default = "2024-02-15-preview" if var == "AZURE_OPENAI_VERSION" else "gpt-35-turbo"
        print(f"  📝 {var}: {value or f'Using default: {default}'}")
    
    print("\n🧪 Testing LLM Evaluation...")
    
    # Test cases
    test_cases = [
        {
            "name": "Authentication Code → Login Test",
            "code": "function validateLogin(username, password) { return authenticateUser(username, password); }",
            "test": {"id": 1, "title": "Valid Login Test", "description": "Test user login functionality"},
            "expected_high_relevance": True
        },
        {
            "name": "CSS Code → Login Test", 
            "code": ".navbar { background-color: blue; padding: 10px; }",
            "test": {"id": 2, "title": "Valid Login Test", "description": "Test user login functionality"},
            "expected_high_relevance": False
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{i}. {test_case['name']}")
        print(f"   Code: {test_case['code'][:50]}...")
        print(f"   Test: {test_case['test']['title']}")
        
        try:
            result = evaluate_test_relevance_with_llm(test_case['code'], test_case['test'])
            
            print(f"   ✅ Relevance: {result['relevance_score']}%")
            print(f"   💭 Reasoning: {result['reasoning']}")
            print(f"   📊 Recommendation: {result['recommendation']}")
            
            # Validate expectation
            is_high_relevance = result['relevance_score'] >= 70
            if is_high_relevance == test_case['expected_high_relevance']:
                print(f"   🎯 Result matches expectation!")
            else:
                print(f"   ⚠️  Unexpected result (expected high relevance: {test_case['expected_high_relevance']})")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
            return False
    
    print(f"\n{'=' * 50}")
    print("🎉 Azure OpenAI integration test completed successfully!")
    print("✅ Your Azure OpenAI setup is working correctly")
    print("✅ LLM-enhanced test suggestions are ready to use")
    
    return True

def test_cost_estimation():
    """Provide cost estimation for Azure OpenAI usage"""
    
    print(f"\n💰 Cost Estimation")
    print("=" * 30)
    
    # Rough estimates (prices vary by region and model)
    gpt35_cost_per_1k_tokens = 0.002  # USD
    gpt4_cost_per_1k_tokens = 0.03    # USD
    
    avg_tokens_per_evaluation = 400   # Prompt + response
    evaluations_per_pr = 10          # NUM_CANDIDATES
    
    print(f"📊 Usage per PR analysis:")
    print(f"   • Evaluations: ~{evaluations_per_pr}")
    print(f"   • Tokens per evaluation: ~{avg_tokens_per_evaluation}")
    print(f"   • Total tokens per PR: ~{evaluations_per_pr * avg_tokens_per_evaluation}")
    
    print(f"\n💵 Estimated cost per PR:")
    gpt35_cost = (evaluations_per_pr * avg_tokens_per_evaluation / 1000) * gpt35_cost_per_1k_tokens
    gpt4_cost = (evaluations_per_pr * avg_tokens_per_evaluation / 1000) * gpt4_cost_per_1k_tokens
    
    print(f"   • GPT-3.5 Turbo: ~${gpt35_cost:.4f}")
    print(f"   • GPT-4: ~${gpt4_cost:.4f}")
    
    print(f"\n📈 Monthly estimates (100 PRs):")
    print(f"   • GPT-3.5 Turbo: ~${gpt35_cost * 100:.2f}")
    print(f"   • GPT-4: ~${gpt4_cost * 100:.2f}")
    
    print(f"\n💡 Cost optimization tips:")
    print(f"   • Use GPT-3.5 Turbo for lower costs")
    print(f"   • Reduce NUM_CANDIDATES from 10 to 5")
    print(f"   • Increase RELEVANCE_THRESHOLD to be more selective")

if __name__ == "__main__":
    success = test_azure_openai_connection()
    
    if success:
        test_cost_estimation()
    else:
        print("\n❌ Azure OpenAI setup failed. Please check your configuration.")
        print("📖 See AZURE_SETUP.md for detailed setup instructions.")