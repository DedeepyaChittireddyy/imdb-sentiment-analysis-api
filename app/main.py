from fastapi import FastAPI
from pydantic import BaseModel
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import torch
import os

# Get model path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "model")

# Load tokenizer and model
tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_DIR)
model = DistilBertForSequenceClassification.from_pretrained(MODEL_DIR)
model.eval()

app = FastAPI()

# Input schema
class TextInput(BaseModel):
    text: str

@app.get("/")
def home():
    return {"message": "Sentiment Analysis API is running!"}

@app.post("/predict")
def predict(input: TextInput):
    inputs = tokenizer(input.text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=1)
        confidence, predicted_class = torch.max(probs, dim=1)
        sentiment = "Positive" if predicted_class.item() == 1 else "Negative"
        return {
            "sentiment": sentiment,
            "confidence": round(confidence.item(), 3)
        }
