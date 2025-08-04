import os
import sys
import json
import random
import torch
import numpy as np
import nltk
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from datasets import Dataset
from nltk.corpus import wordnet
from nltk import pos_tag, word_tokenize
from transformers import RobertaTokenizerFast, RobertaForSequenceClassification, Trainer, TrainingArguments
import evaluate
import pickle

# ------------------------------
# Setup & Reproducibility
# ------------------------------
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)

# ------------------------------
# File Paths
# ------------------------------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(BASE_DIR)
DATA_PATH = os.path.join(BASE_DIR, "data", "training_data.json")

# ------------------------------
# Load and Prepare Data
# ------------------------------
with open(DATA_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

texts = [entry["user_input"] for entry in data]
labels_raw = [entry["intent"] for entry in data]

# Label encoding
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels_raw)
num_labels = len(label_encoder.classes_)

# ------------------------------
# Synonym Replacement for Low-Sample Classes
# ------------------------------
label_counts = dict(zip(*np.unique(labels, return_counts=True)))
low_sample_classes = [label for label, count in label_counts.items() if count < 20]

def get_synonyms(word):
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonym = lemma.name().replace('_', ' ').lower()
            if synonym != word:
                synonyms.add(synonym)
    return list(synonyms)

def synonym_replacement(text):
    words = word_tokenize(text)
    tagged = pos_tag(words)
    new_words = []
    for word, tag in tagged:
        if tag.startswith('NN') or tag.startswith('VB') or tag.startswith('JJ'):
            synonyms = get_synonyms(word)
            if synonyms:
                word = random.choice(synonyms)
        new_words.append(word)
    return ' '.join(new_words)

def augment_low_sample(texts, labels):
    new_texts, new_labels = [], []
    for x, y in zip(texts, labels):
        if y in low_sample_classes:
            new_texts.append(synonym_replacement(x))
            new_labels.append(y)
    # Ensure all components are lists before concatenation
    return list(texts) + new_texts, list(labels) + new_labels


texts, labels = augment_low_sample(texts, labels)

# ------------------------------
# Train-Validation Split
# ------------------------------
train_texts, val_texts, train_labels, val_labels = train_test_split(
    texts, labels, test_size=0.2, stratify=labels, random_state=SEED
)

# ------------------------------
# Tokenization
# ------------------------------
tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")

def tokenize_function(texts, labels):
    encodings = tokenizer(texts, truncation=True, padding=True, max_length=128)
    encodings['labels'] = labels
    return encodings

train_dataset = Dataset.from_dict(tokenize_function(train_texts, train_labels))
val_dataset = Dataset.from_dict(tokenize_function(val_texts, val_labels))

# ------------------------------
# Load Model
# ------------------------------
model = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=num_labels)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# ------------------------------
# Training Configuration
# ------------------------------
training_args = TrainingArguments(
    output_dir=os.path.join(BASE_DIR, "results"),
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=5,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    logging_dir=os.path.join(BASE_DIR, "logs"),
    save_total_limit=1
)

# ------------------------------
# Evaluation Metrics
# ------------------------------
accuracy = evaluate.load("accuracy")
f1 = evaluate.load("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy.compute(predictions=predictions, references=labels)["accuracy"],
        "f1": f1.compute(predictions=predictions, references=labels, average="weighted")["f1"]
    }

# ------------------------------
# Train the Model
# ------------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

trainer.train()

# ------------------------------
# Save the Model
# ------------------------------
SAVE_DIR = os.path.join(BASE_DIR, "saved_model")
model.save_pretrained(SAVE_DIR)
tokenizer.save_pretrained(SAVE_DIR)

with open(os.path.join(SAVE_DIR, "label_encoder.pkl"), "wb") as f:
    pickle.dump(label_encoder, f)
