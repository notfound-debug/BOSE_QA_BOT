import sys
import os
import pandas as pd

# --- PATH FIX ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# ----------------

from src.bot import get_qa_chain

def run_evaluation():
    print(" Starting Automated Evaluation...")
    
    # 1. Initialize Bot
    bot = get_qa_chain()
    if not bot:
        return

    # 2. Define the Test Set (From AIML.pdf)
    # We use "Keyword Matching" to check if the answer is roughly correct.
    test_cases = [
        {
            "question": "What is the maximum power consumption of the processor?",
            "expected_keywords": ["35 W", "35W", "35 watts"], 
            "doc_source": "EX-1280C"
        },
        {
            "question": "What is the IP rating of the DM8SE loudspeaker?",
            "expected_keywords": ["IP55"],
            "doc_source": "DM8SE"
        },
        {
            "question": "What is the Net Weight of a single DM8SE loudspeaker?",
            "expected_keywords": ["10.3 kg", "22.8 lb"],
            "doc_source": "DM8SE"
        },
        {
            "question": "What is the Dynamic Range of the analog signal path?",
            "expected_keywords": ["115 dB"],
            "doc_source": "EX-1280C"
        },
        {
            "question": "What is the length of AEC tail in milliseconds?",
            "expected_keywords": ["480 ms", "480ms"],
            "doc_source": "EX-1280C"
        }
    ]

    results = []
    correct_count = 0

    print(f" Testing {len(test_cases)} questions...\n")

    for i, test in enumerate(test_cases):
        print(f"🔹 Q{i+1}: {test['question']}")
        
        # Ask the bot
        try:
            response_payload = bot.invoke({"query": test['question']})
            generated_answer = response_payload["result"]
            
            # Check correctness (Simple Keyword Match)
            # If ANY of the expected keywords appear in the answer, we mark it PASS
            is_correct = any(keyword.lower() in generated_answer.lower() for keyword in test['expected_keywords'])
            
            status = " PASS" if is_correct else "❌ FAIL"
            if is_correct:
                correct_count += 1
            
            print(f"   Answer: {generated_answer}")
            print(f"   Result: {status}")
            print("-" * 30)
            
            results.append({
                "Question": test['question'],
                "Bot Answer": generated_answer,
                "Status": status
            })
            
        except Exception as e:
            print(f"    Error: {e}")

    # 3. Final Report
    accuracy = (correct_count / len(test_cases)) * 100
    print(f"\n🏆 Final Accuracy: {accuracy}% ({correct_count}/{len(test_cases)})")
    
    return results

if __name__ == "__main__":
    run_evaluation()