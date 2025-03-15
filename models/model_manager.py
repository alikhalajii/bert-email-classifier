import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class ModelManager:
    def __init__(self, model_name="distilbert-base-uncased", label_map=None):
        """Initialize model & tokenizer with dynamic labels."""
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
            )

        if label_map is None:
            raise ValueError("❌ Label map is required.")

        self.label_map = label_map
        self.inverse_label_map = {v: k for k, v in label_map.items()}

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=len(label_map)
        ).to(self.device)
        self.model.eval()

    def predict(self, email_text):
        """Predict the category of a single email."""
        inputs = self.tokenizer(
            email_text, return_tensors="pt",
            padding=True,
            truncation=True).to(self.device)
        
        # Get the predicted label
        with torch.no_grad():
            logits = self.model(**inputs).logits
            predicted_label = torch.argmax(logits, dim=-1).item()
        return self.inverse_label_map[predicted_label]

    @classmethod
    def load_model(cls, load_path, label_map):
        """Load a trained model with label mapping."""
        instance = cls(label_map=label_map)
        instance.model = AutoModelForSequenceClassification.from_pretrained(
            load_path).to(instance.device)
        instance.tokenizer = AutoTokenizer.from_pretrained(load_path)
        return instance

    def save_model(self, save_path="./saved_models/email-classifier"):
        """Save trained model and tokenizer."""
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        print(f"✅ Model saved at {save_path}")
