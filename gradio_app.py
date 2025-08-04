import sys
import os
import json
from datetime import datetime
import gradio as gr

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from dialogue_manager.flow_manager import get_response
from nlp_engine.intent_classifier import IntentClassifier

# File path to store conversation history
HISTORY_FILE = "conversation_history.json"

# Initialize classifier
classifier = IntentClassifier()


from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch
import pickle

class DistilBertIntentClassifier:
    def __init__(self, model_path="distilbert_intent_model"):
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_path)
        self.model = DistilBertForSequenceClassification.from_pretrained(model_path)
        self.model.eval()

        # Load label encoder to get intent names
        with open(os.path.join(model_path, "label_encoder.pkl"), "rb") as f:
            self.label_encoder = pickle.load(f)

    def predict_intent(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
        with torch.no_grad():
            logits = self.model(**inputs).logits
        predicted_class_id = logits.argmax().item()
        return self.label_encoder.inverse_transform([predicted_class_id])[0]


# Load or initialize history
def load_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, 'r') as f:
            return json.load(f)
    return []

# Save updated history
def save_history(history):
    with open(HISTORY_FILE, 'w') as f:
        json.dump(history, f, indent=4)

# Append current interaction
def log_interaction(user_input, intent, response):
    history = load_history()
    history.append({
        "timestamp": datetime.now().isoformat(),
        "user_input": user_input,
        "predicted_intent": intent,
        "bot_response": response
    })
    save_history(history)

# Process user input and generate response
def process_user_input(user_input, history=[]):
    if user_input.lower() in ["exit", "quit"]:
        response = "Take care! Talk to you soon."
        history.append((user_input, response))
        return history, ""

    intent = classifier.predict_intent(user_input)
    print(f"[Debug] Predicted Intent: {intent}")
    

    response = get_response(intent)

    # Optional: add continuity prompt
    past_history = load_history()
    if len(past_history) > 1:
        last_entry = past_history[-2]
        if "back pain" in last_entry["user_input"].lower():
            response += " By the way, has your back pain improved since we last talked?"

    log_interaction(user_input, intent, response)
    history.append((user_input, response))
    return history, ""

# Show conversation history

def show_history():
    history = load_history()
    history_text = "\n".join([
        f"{entry['timestamp']} - You: {entry['user_input']} | Bot: {entry['bot_response']}"
        for entry in history[-5:]  # show last 5 only
    ])
    return history_text

# Gradio UI
with gr.Blocks(theme=gr.themes.Base(), css=".gr-button {background-color:#6E9DCB !important; color:white;} .gr-textbox {font-size:16px;}") as demo:
    gr.Markdown("## 🤖 <span style='color:#4464AD'>Pain Support Chatbot</span>")

    chatbot = gr.Chatbot(label="Your Conversation", elem_id="chat-window", show_label=False)
    user_input = gr.Textbox(label="💬 Type your message", placeholder="e.g., I have lower back pain today", scale=1)
    
    with gr.Row():
        send_btn = gr.Button("🚀 Send", variant="primary")
        hist_btn = gr.Button("🕓 Show Last 5 Messages")
        clear_btn = gr.Button("🧹 Clear Chat")

    send_btn.click(fn=process_user_input, inputs=[user_input, chatbot], outputs=[chatbot, user_input])
    hist_btn.click(fn=show_history, outputs=chatbot)
    clear_btn.click(lambda: [], None, chatbot)


if __name__ == "__main__":
    demo.launch()
