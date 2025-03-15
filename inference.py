from models.model_manager import ModelManager
from datasets import load_dataset
from data.data_processor import DataProcessor


# Load dataset to get label mapping
ds = load_dataset("Tobi-Bueck/customer-support-tickets")
data_processor = DataProcessor()
data_processor.create_label_map(ds)

# Load trained model
classifier = ModelManager.load_model(
    "./saved_models/email-classifier",
    label_map=data_processor.label_map
    )

# Predict single email
email_text = "Your package will be delivered tomorrow."
predicted_category = classifier.predict(email_text)
print(f"Predicted category: {predicted_category}")
