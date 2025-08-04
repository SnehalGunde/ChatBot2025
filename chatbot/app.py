
import sys
import os

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from dialogue_manager.flow_manager import get_response
from nlp_engine.intent_classifier import IntentClassifier


def main():
    print("Welcome to the Pain Support Chatbot (TF-IDF Version)")
    classifier = IntentClassifier()

    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Bot: Take care! Talk to you soon.")
            break

        intent = classifier.predict_intent(user_input)
        print(f"[Debug] Predicted Intent: {intent}")

        response = get_response(intent)
        print(f"Bot: {response}")

if __name__ == "__main__":
    main()
