from transformers import TrainingArguments, Trainer


class TrainerManager:
    def __init__(self, model, dataset):
        self.model = model
        self.dataset = dataset

    def train(self):
        training_args = TrainingArguments(
            output_dir='./results',
            evaluation_strategy="epoch",
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            logging_dir='./logs',
            save_strategy="epoch",
            num_train_epochs=3,
            logging_steps=200,
            load_best_model_at_end=True
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.dataset['train'],
            eval_dataset=self.dataset['validation'],
            compute_metrics=self.compute_metrics
        )

        trainer.train()

    def compute_metrics(self, p):
        predictions, labels = p
        preds = predictions.argmax(axis=-1)
        accuracy = (preds == labels).mean()
        return {"accuracy": accuracy}
