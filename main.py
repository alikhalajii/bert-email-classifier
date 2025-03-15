from datasets import load_dataset
from models.model_manager import ModelManager
from data.data_processor import DataProcessor
from training.trainer_manager import TrainerManager

# Load dataset
ds = load_dataset("ale-dp/german-english-email-ticket-classification")
ds = ds["train"]

# Split dataset into train/test (80/20)
ds_split = ds.train_test_split(test_size=0.2, seed=42)
ds_test_valid = ds_split["test"].train_test_split(test_size=0.5, seed=42)

# Combine train, validation, and test splits
dataset_splits = {
    "train": ds_split["train"],
    "validation": ds_test_valid["train"],
    "test": ds_test_valid["test"]
}

# Process dataset (tokenization, encoding labels, etc.)
data_processor = DataProcessor()
dataset = data_processor.prepare_data(dataset_splits)

# Initialize model
classifier = ModelManager(
    model_name="distilbert-base-uncased",
    label_map=data_processor.label_map
    )

# Train the model
trainer_manager = TrainerManager(classifier.model, dataset)
trainer_manager.train()

# Save the trained model
classifier.save_model()
