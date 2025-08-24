from sklearn.base import BaseEstimator, ClassifierMixin
from datasets import Dataset
from transformers import Trainer, TrainingArguments, DataCollatorWithPadding
import torch

class BERTSklearnWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, model, tokenizer, num_labels=4):
        self.num_labels = num_labels
        self.tokenizer = tokenizer
        self.model = model

    def fit(self, X, y):
        # Create a dataset from the input data and labels
        dataset = Dataset.from_dict({"text": list(X), "labels": list(y)})

        def tokenize(example):
            # Ensure padding is handled by the data collator
            return self.tokenizer(example["text"], truncation=True)

        # Tokenize the dataset
        tokenized_dataset = dataset.map(tokenize, batched=True)

        # Prepare the dataset for the Trainer with explicit columns and tensor format
        train_dataset = Dataset.from_dict({
            'input_ids': tokenized_dataset['input_ids'],
            'attention_mask': tokenized_dataset['attention_mask'],
            'labels': tokenized_dataset['labels']
        })
        train_dataset.set_format("torch")

        # Use DataCollatorWithPadding to handle padding dynamically per batch
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer, return_tensors="pt")


        training_args = TrainingArguments(
            output_dir="./results",
            num_train_epochs=1,
            per_device_train_batch_size=8,
            logging_steps=10,
            save_steps=10,
            report_to="none", # Disable reporting to wandb
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=data_collator, # Add the data collator
        )

        trainer.train()
        return self

    def predict_proba(self, X):
        self.model.eval()
        # Tokenize the input data
        inputs = self.tokenizer(list(X), return_tensors="pt", padding=True, truncation=True)

        # Create a Dataset from the tokenized inputs
        predict_dataset = Dataset.from_dict(inputs)
        predict_dataset.set_format("torch")

        # Use Trainer for prediction to handle batching and device placement
        trainer = Trainer(model=self.model)
        predictions = trainer.predict(predict_dataset)
        probs = torch.nn.functional.softmax(torch.tensor(predictions.predictions), dim=1)
        return probs.numpy()


    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)
