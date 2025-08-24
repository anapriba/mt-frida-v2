import csv

train_issues = []
test_issues = []
eval_issues = []

with open('train_issues.csv', 'r') as f:
    reader = csv.reader(f, delimiter=',')
    i = 0 
    for row in reader:
        # skip header row
        if i == 0:
            i += 1
            continue

        train_issues.append(row)
        i += 1

with open('test_issues.csv', 'r') as f:
    reader = csv.reader(f, delimiter=',')
    i = 0 
    for row in reader:
        # skip header row
        if i == 0:
            i += 1
            continue

        test_issues.append(row)
        i += 1

with open('eval_issues.csv', 'r') as f:
    reader = csv.reader(f, delimiter=',')
    i = 0 
    for row in reader:
        # skip header row
        if i == 0:
            i += 1
            continue

        eval_issues.append(row)
        i += 1


train_data = []
test_data = []
eval_data = []

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
      
      train_data.append(row)
      i+=1

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
      
      test_data.append(row)
      i+=1

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
      
      eval_data.append(row)
      i+=1

print("Loaded data ... ")
print(f"Train data {len(train_data)}")
print(f"Test data {len(test_data)}")
print(f"Eval data {len(eval_data)}")
print(f"Train issues {len(train_issues)}")
print(f"Test issues {len(test_issues)}")
print(f"Eval issues {len(eval_issues)}")

print("Correct data")

def generate_correction_data(original, issues):
   suggestions = []
   suggestions.append(['text', 'label', 'proposed_label', 'confidence'])

   for i in range(len(original)):
      original_row = original[i]
      issues_row = issues[i]

      new_row = []
      new_row.append(original_row[0])
      new_row.append(original_row[1])

      if issues_row[1] == 'True':
         new_row.append(issues_row[4])
         new_row.append(issues_row[2])
      
      suggestions.append(new_row)

   return suggestions

def correct_issues(original, issues):
    final = []
    final.append(['text', 'label']) # append header
    for i in range(len(original)):
        original_row = original[i]
        issues_row = issues[i]

        if issues_row[1] == 'True':
            original_row[1] = issues_row[4] # if issue was detected, change the label value
   
        new_data_row = original_row
        final.append(new_data_row)
    return final

train_final = correct_issues(train_data, train_issues)
test_final = correct_issues(test_data, test_issues)
eval_final = correct_issues(eval_data, eval_issues)

train_suggestions = generate_correction_data(train_data, train_issues)
test_suggestions = generate_correction_data(test_data, test_issues)
eval_suggestions = generate_correction_data(eval_data, eval_issues)


def write_tsv(data, filename):
    with open(filename, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file, delimiter='\t')
        writer.writerows(data)

write_tsv(train_final, 'train_final.tsv')
print("Corrected train")
write_tsv(test_final, 'test_final.tsv')
print("Corrected test")
write_tsv(eval_final, 'eval_final.tsv')
print("Corrected eval")


write_tsv(train_suggestions, 'train_suggestions.tsv')
write_tsv(test_suggestions, 'test_suggestions.tsv')
write_tsv(eval_suggestions, 'eval_suggestions.tsv')