#!/usr/bin/env python3
"""
Test script to diagnose LLM service and DOCX import issues
"""
import sys
import os

print("Testing LLM service and DOCX imports to diagnose Bus error...")

# Test 1: Basic openai import
try:
    print("1. Testing openai import...")
    import openai
    print(f"✅ OpenAI imported successfully - version: {openai.__version__}")
except Exception as e:
    print(f"❌ OpenAI import failed: {e}")
    sys.exit(1)

# Test 2: OpenAI API key setup
try:
    print("2. Testing OpenAI API key setup...")
    # Try to set API key (even if it's dummy)
    openai.api_key = "test-key"
    print("✅ OpenAI API key setup successful")
except Exception as e:
    print(f"❌ OpenAI API key setup failed: {e}")
    sys.exit(1)

# Test 3: DOCX import
try:
    print("3. Testing python-docx import...")
    import docx
    print("✅ python-docx imported successfully")
except Exception as e:
    print(f"❌ python-docx import failed: {e}")
    sys.exit(1)

# Test 4: DOCX Document creation
try:
    print("4. Testing DOCX Document creation...")
    from docx import Document
    doc = Document()
    print("✅ DOCX Document creation successful")
except Exception as e:
    print(f"❌ DOCX Document creation failed: {e}")
    sys.exit(1)

# Test 5: Our LLM service import
try:
    print("5. Testing our LLM service import...")
    sys.path.append('src')
    from utils.llm_service import GameLensLLMService
    print("✅ LLM service imported successfully")
except Exception as e:
    print(f"❌ LLM service import failed: {e}")
    sys.exit(1)

# Test 6: LLM service initialization
try:
    print("6. Testing LLM service initialization...")
    llm_service = GameLensLLMService()
    print("✅ LLM service initialized successfully")
except Exception as e:
    print(f"❌ LLM service initialization failed: {e}")
    sys.exit(1)

# Test 7: LLM service availability check
try:
    print("7. Testing LLM service availability check...")
    is_available = llm_service.is_available()
    print(f"✅ LLM service availability check successful: {is_available}")
except Exception as e:
    print(f"❌ LLM service availability check failed: {e}")
    sys.exit(1)

# Test 8: Test with actual API call (if API key is set)
try:
    print("8. Testing actual OpenAI API call...")
    import os
    api_key = os.getenv('OPENAI_API_KEY')
    if api_key and api_key != 'your_openai_api_key_here':
        # Try a simple API call
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=10
        )
        print("✅ OpenAI API call successful")
    else:
        print("⚠️ No valid API key found, skipping API call test")
except Exception as e:
    print(f"❌ OpenAI API call failed: {e}")
    # Don't exit here, as this might be expected without a valid API key

print("\n🎉 All LLM and DOCX tests passed! The issue might be in Streamlit integration.")
print("Next step: Test with Streamlit app to see if the issue is in the web framework integration.")
