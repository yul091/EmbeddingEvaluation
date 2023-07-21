
import json
import csv
from utils import parse_statement


with open('se_tasks/comment_generate/dataset/train.json') as f:
    dataset = f.readlines()
train_dataset = [json.loads(data) for data in dataset]
with open('se_tasks/comment_generate/dataset/train.tsv',  'w', encoding='utf-8') as f:
    writer = csv.writer(f, delimiter='\t')
    writer.writerow(['label', 'body'])
    for i, data in enumerate(train_dataset):
        x, y = parse_statement(data['code']), data['nl']
        try:
            writer.writerow([x, y])
        except:
            writer.writerow([data['code'], data['nl']])

with open('se_tasks/comment_generate/dataset/test.json') as f:
    dataset = f.readlines()
test_dataset = [json.loads(data) for data in dataset]
with open('se_tasks/comment_generate/dataset/test.tsv',  'w', encoding='utf-8') as f:
    writer = csv.writer(f, delimiter='\t')
    writer.writerow(['label', 'body'])
    for i, data in enumerate(test_dataset):
        x, y = parse_statement(data['code']), data['nl']
        try:
            writer.writerow([x, y])
        except:
            writer.writerow([data['code'], data['nl']])

print('perpeare the dataset successful')