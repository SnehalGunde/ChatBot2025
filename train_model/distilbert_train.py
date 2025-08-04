import os
import json
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from datasets import Dataset
import evaluate

# ========== Load and Preprocess Data ==========

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_PATH = os.path.join(BASE_DIR, "data", "training_data.json")

with open(DATA_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

texts = [entry["user_input"] for entry in data]
intents = [entry["intent"] for entry in data]

label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(intents)
num_classes = len(label_encoder.classes_)

# Stratified train-test split
X_train, X_val, y_train, y_val = train_test_split(
    texts, labels, test_size=0.2, random_state=42, stratify=labels
)

# Tokenization
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

def tokenize(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=128)

train_dataset = Dataset.from_dict({"text": X_train, "label": y_train})
val_dataset = Dataset.from_dict({"text": X_val, "label": y_val})

train_dataset = train_dataset.map(tokenize, batched=True)
val_dataset = val_dataset.map(tokenize, batched=True)

train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
val_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

# ========== Compute Class Weights ==========

def compute_class_weights(labels, num_classes):
    class_weights = compute_class_weight("balanced", classes=np.arange(num_classes), y=labels)
    return torch.tensor(class_weights, dtype=torch.float)

class_weights = compute_class_weights(y_train, num_classes)

# ========== Custom Trainer ==========

from transformers import Trainer

class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights.to(model.device))
        loss = loss_fn(logits, labels)
        return (loss, outputs) if return_outputs else loss

# ========== Model and Training Setup ==========

model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", num_labels=num_classes
)

training_args = TrainingArguments(
    output_dir=os.path.join(BASE_DIR, "distilbert_results"),
    num_train_epochs=5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir=os.path.join(BASE_DIR, "logs"),
    logging_steps=50,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
)

accuracy = evaluate.load("accuracy")
precision = evaluate.load("precision")
recall = evaluate.load("recall")
f1 = evaluate.load("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {
        "accuracy": accuracy.compute(predictions=preds, references=labels)["accuracy"],
        "precision": precision.compute(predictions=preds, references=labels, average="weighted")["precision"],
        "recall": recall.compute(predictions=preds, references=labels, average="weighted")["recall"],
        "f1": f1.compute(predictions=preds, references=labels, average="weighted")["f1"],
    }

# ========== Train ==========

trainer = WeightedTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

trainer.train()

# ========== Evaluation and Confusion Matrix ==========

preds_output = trainer.predict(val_dataset)
preds = np.argmax(preds_output.predictions, axis=1)
true = preds_output.label_ids

print("\nClassification Report:\n")
print(classification_report(true, preds, target_names=label_encoder.classes_))

from sklearn.metrics import accuracy_score, precision_recall_fscore_support
# Accuracy, Precision, Recall, F1
accuracy = accuracy_score(true, preds)
precision, recall, f1, _ = precision_recall_fscore_support(
    true, preds, average="weighted", zero_division=0
)
print("\nMetrics for Distil-Bert:\n")
print(f"\n Accuracy:  {accuracy:.4f}")
print(f" Precision: {precision:.4f}")
print(f" Recall:    {recall:.4f}")
print(f" F1 Score:  {f1:.4f}")

cm = confusion_matrix(true, preds)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_, cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "confusion_matrix.png"))
plt.show()
#########################################################################
# ====== Save Fine-Tuned Model & Tokenizer ======
SAVE_DIR = os.path.join(BASE_DIR, "distilbert_intent_model")

# Create the directory if it doesn't exist
os.makedirs(SAVE_DIR, exist_ok=True)

# Save model and tokenizer
model.save_pretrained(SAVE_DIR)
tokenizer.save_pretrained(SAVE_DIR)

# Save Label Encoder (for inference)
import pickle
with open(os.path.join(SAVE_DIR, "label_encoder.pkl"), "wb") as f:
    pickle.dump(label_encoder, f)

print(f"\n✅ Model, tokenizer, and label encoder saved at: {SAVE_DIR}")
