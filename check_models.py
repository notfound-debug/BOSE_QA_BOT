import sys
import os
import google.generativeai as genai

# --- FIX: Ensure we can import from src ---
# This adds the current folder to the path, so "from src.config" works
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

try:
    from src.config import GOOGLE_API_KEY
    print("✅ Successfully imported API Key from src.config")
except ImportError:
    print("❌ Error: Could not import settings. Make sure src/config.py exists.")
    sys.exit(1)

# Configure the SDK
genai.configure(api_key=GOOGLE_API_KEY)

print("\n🔍 Checking available models for your API Key...")
print("------------------------------------------------")

try:
    available_models = []
    # List all models and filter for those that generate content (chat)
    for m in genai.list_models():
        if 'generateContent' in m.supported_generation_methods:
            # We only care about "gemini" models for this task
            if "gemini" in m.name:
                print(f"   - Found: {m.name}")
                available_models.append(m.name)
            
    if not available_models:
        print("\n❌ No models found! Your API Key might be invalid.")
    else:
        print(f"\n✅ Success! Found {len(available_models)} usable models.")
        print("\n👇 RECOMMENDATION: Update src/config.py with one of these exact names:")
        # Suggest the best one found
        best_model = next((m for m in available_models if "flash" in m and "1.5" in m), available_models[0])
        # Remove "models/" prefix if present for the config
        clean_name = best_model.replace("models/", "")
        print(f'LLM_MODEL_NAME = "{clean_name}"')
        
except Exception as e:
    print(f"❌ Error contacting Google API: {e}")