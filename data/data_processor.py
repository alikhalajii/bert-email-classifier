from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer


class DataProcessor:
    def __init__(self, model_name="distilbert-base-uncased"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.label_map = {}

    def create_label_map(self, dataset):
        unique_labels = sorted(set(dataset["train"]["queue"]))
        self.label_map = {label: idx for idx, label in enumerate(unique_labels)}

    def encode_labels(self, labels):
        return [self.label_map[label] for label in labels]

    def prepare_data(self, dataset):
        self.create_label_map(dataset)

        train_data = {
            "email": dataset["train"]["body"],
            "label": self.encode_labels(dataset["train"]["queue"])
        }
        valid_data = {
            "email": dataset["validation"]["body"],
            "label": self.encode_labels(dataset["validation"]["queue"])
        }
        test_data = {
            "email": dataset["test"]["body"],
            "label": self.encode_labels(dataset["test"]["queue"])
        }

        dataset = DatasetDict({
            "train": Dataset.from_dict(train_data),
            "validation": Dataset.from_dict(valid_data),
            "test": Dataset.from_dict(test_data)
        })

        def tokenize_function(examples):
            return self.tokenizer(
                examples["email"],
                padding="max_length",
                truncation=True,
                max_length=512
            )

        return dataset.map(tokenize_function, batched=True)
