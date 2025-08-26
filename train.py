from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.metrics import precision_recall_fscore_support
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    EarlyStoppingCallback,
    Trainer,
)
import datetime
import csv
import pandas as pd
from datasets import Dataset
from sklearn.preprocessing import LabelEncoder


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    return {"precision": precision, "recall": recall, "f1": f1}


def encode_labels(labels):
   encoder = LabelEncoder()
   encoder.fit(labels)

   encoded_labels = encoder.transform(labels)
   return encoded_labels

def get_dataset(path):
    data = []
    labels = []
    with open(path, 'r') as f:
        reader = csv.reader(f, delimiter='\t')

        i = 0
        for row in reader:
            # skip header row
            if i == 0:
                i+=1
                continue

            data.append(row[0])
            labels.append(row[1])
            i+=1 
    
    encoded_labels = encode_labels(labels)
    df = pd.DataFrame({
        "text": data,
        "label": encoded_labels
    })

    dataset = Dataset.from_pandas(df)
    return dataset

model_name = "EMBEDDIA/crosloengual-bert"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)

def tokenize(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True)

training_args = TrainingArguments(
    eval_steps=500,
    overwrite_output_dir=True,
    per_device_train_batch_size=16,
    do_train=True,
    do_eval=True,
    warmup_steps=0,  # 500,
    learning_rate=1e-05,  # 1e-5,
    weight_decay=0.02,
    num_train_epochs=1,
    save_total_limit=1,
    seed=10,
)

train_dataset = get_dataset('train_suggested_final.tsv')
test_dataset = get_dataset('test_suggested_final.tsv')
eval_dataset = get_dataset('eval_suggested_final.tsv')

train_dataset = train_dataset.map(tokenize, batched=True)
test_dataset = test_dataset.map(tokenize, batched=True)
eval_dataset = eval_dataset.map(tokenize, batched=True)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

total_epochs = 10
# Train for multiple epochs manually and evaluate after each epoch
for epoch in range(int(total_epochs)):
    print(f"Starting epoch {epoch+1}/{int(total_epochs)}")
    
    # Train for one epoch
    trainer.train(resume_from_checkpoint=False)  # resume_from_checkpoint can be used if interrupted
    
    # Evaluate manually
    metrics = trainer.evaluate(eval_dataset)
    print(f"Metrics after epoch {epoch+1}: {metrics}")

test_metrics = trainer.evaluate(test_dataset)
print(f"Test metrics: {test_metrics}")