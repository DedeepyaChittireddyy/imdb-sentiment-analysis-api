# training/train.py

from datasets import load_dataset
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np
import os

# Load IMDb dataset
print("📥 Loading IMDb dataset...")
dataset = load_dataset("imdb")

# Load tokenizer
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

# Tokenization function
def tokenize(batch):
    return tokenizer(batch["text"], truncation=True, padding="max_length")

# Apply tokenization
print("🔄 Tokenizing dataset...")
dataset = dataset.map(tokenize, batched=True)
dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])

# Load pre-trained model
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")

# Define evaluation metrics
def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary")
    acc = accuracy_score(labels, preds)
    return {
        "accuracy": acc,
        "f1": f1,
        "precision": precision,
        "recall": recall
    }

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=2,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    evaluation_strategy="steps",
    save_strategy="steps",
    eval_steps=100,
    save_steps=100,
    logging_dir="./logs",
    logging_steps=10,
    load_best_model_at_end=True,
    save_total_limit=1,
    report_to="none"  # disables WandB if not configured
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"].shuffle(seed=42).select(range(10000)),
    eval_dataset=dataset["test"].shuffle(seed=42).select(range(2000)),
    compute_metrics=compute_metrics
)

# Train the model
print("🚀 Training model...")
trainer.train()

# Save fine-tuned model and tokenizer
output_dir = "./model"
os.makedirs(output_dir, exist_ok=True)
print("💾 Saving model to ./model/")
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

print("✅ Fine-tuning complete. Model saved to 'model/' directory.")
