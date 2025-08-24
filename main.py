import csv
import numpy as np 
from sklearn.preprocessing import LabelEncoder

from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.linear_model import LogisticRegression

import torch
from cleanlab.classification import CleanLearning
from bert_wraper import BERTSklearnWrapper

train_data = []
train_labels = []

print("Opening train.tsv")
with open('projekt/mt/train.tsv', 'r') as f:
    reader = csv.reader(f, delimiter='\t')

    i = 0
    for row in reader:
      # skip header row
      if i == 0:
        i+=1
        continue

      if row[1] == '3':
        continue
      
      train_data.append(row[0])
      train_labels.append(row[1])
      i+=1


test_data = []
test_labels = []

print("Opening test.tsv")
with open('projekt/mt/test.tsv', 'r') as f:
    reader = csv.reader(f, delimiter='\t')

    i = 0
    for row in reader:
      # skip header row
      if i == 0:
        i+=1
        continue

      if row[1] == '3':
        continue
      
      test_data.append(row[0])
      test_labels.append(row[1])
      i+=1

eval_data = []
eval_labels = []

print("Opening eval.tsv")
with open('projekt/mt/eval.tsv', 'r') as f:
    reader = csv.reader(f, delimiter='\t')

    i = 0
    for row in reader:
      # skip header row
      if i == 0:
        i+=1
        continue

      if row[1] == '3':
        continue
      
      eval_data.append(row[0])
      eval_labels.append(row[1])
      i+=1

# cleanlab and scikit-learn all use numpy so I need to convert regular lists to numpy list
train_data = np.array(train_data)
train_labels = np.array(train_labels)

test_data = np.array(test_data)
test_labels = np.array(test_labels)

eval_data = np.array(eval_data)
eval_labels = np.array(eval_labels)



print(f"Loaded train data -> lenght={len(train_data)}")
print(f"Loaded train labels -> lenght={len(train_data)}")


print(f"Loaded test data -> lenght={len(test_data)}")
print(f"Loaded test labels -> lenght={len(test_labels)}")


print(f"Loaded eval data -> lenght={len(eval_data)}")
print(f"Loaded eval labels -> lenght={len(eval_labels)}")

def encode_labels(labels):
   encoder = LabelEncoder()
   encoder.fit(labels)

   encoded_labels = encoder.transform(labels)
   return encoded_labels

print("Encoding labels")

encoded_train_labels = encode_labels(train_labels)
encoded_test_labels = encode_labels(test_labels)
encoded_eval_labels = encode_labels(eval_labels)

print(f"Encoded train {len(encoded_train_labels)} labels.")
print(f"Encoded train {len(encoded_test_labels)} labels.")
print(f"Encoded train {len(encoded_eval_labels)} labels.")


print(f"Current torch version - {torch.__version__}")
print("Trying the pipe ...")

pipe = pipeline(
    "sentiment-analysis", model="thak123/Cro-Frida", tokenizer="EMBEDDIA/crosloengual-bert"
)

print("Pipe loaded, loading tokenizer and model ...")

tokenizer = AutoTokenizer.from_pretrained("EMBEDDIA/crosloengual-bert")
model = AutoModelForSequenceClassification.from_pretrained("thak123/Cro-Frida", num_labels=3)

print("Tokenizer and model loaded.")
print("Loading sklearn wraper model around cro-frida ....")

wrapper = BERTSklearnWrapper(num_labels=4, model=model, tokenizer=tokenizer)

def find_test_and_eval_issues(model, test_data, test_labels, eval_data, eval_labels):
    cl = CleanLearning(model)

    pred_test = model.predict_proba(test_data)
    pred_eval = model.predict_proba(eval_data)
    print("Got Cro-Frida predictions for test and eval")

    print("Finding label issues for test and eval ...")
    test_label_issues = cl.find_label_issues(labels=test_labels, pred_probs=pred_test)
    eval_label_issues = cl.find_label_issues(labels=eval_labels, pred_probs=pred_eval)


    print("found label issues, saving to file test_issues.csv and eval_issues.csv")

    test_label_issues.to_csv('test_issues.csv', index=True)
    eval_label_issues.to_csv('eval_issues.csv', index=True)

def find_train_issues(train_data, train_labels):
   cl = CleanLearning(LogisticRegression(max_iter=1000))
   train_embedings = tokenizer(train_data.tolist(), padding="max_length", truncation=True)
   cl.fit(np.array(train_embedings['input_ids']), train_labels)
   label_issues = cl.get_label_issues()
   
   label_issues.to_csv('train_issues.csv', index=True)

import argparse

parser = argparse.ArgumentParser(description="Movie review analysis with cleanlab")

parser.add_argument("--fix-train", type=bool, required=True)

args = parser.parse_args()

if args.fix_train:
   print("Finding train issues")
   find_train_issues(train_data, encoded_train_labels)
else:
   print("Finding test and eval issues")
   find_test_and_eval_issues(wrapper, test_data=test_data, test_labels=encoded_test_labels, eval_data=eval_data, eval_labels=encoded_eval_labels)
