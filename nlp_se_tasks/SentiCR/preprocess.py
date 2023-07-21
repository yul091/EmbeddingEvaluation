from openpyxl import load_workbook
from sklearn.model_selection import train_test_split
import joblib
import re
import csv


def clean_data(data_str):
    tokens = [x.lower() for x in re.findall(r'\w+|\W+', data_str) if not x.isspace()]
    body = ' '.join(tokens)
    return body


workbook = load_workbook('./dataset/oracle.xlsx')
sheets = workbook.get_sheet_names()  # 从名称获取sheet
booksheet = workbook.get_sheet_by_name(sheets[0])

rows = booksheet.rows
columns = booksheet.columns

for i, m in enumerate(columns):
    if i == 0:
        dataset = m
    else:
        label = m

dataset = [clean_data(data.value) for data in dataset]
label = [abs(y.value) for y in label]


x_train, x_test, y_train, y_test = train_test_split(dataset, label, test_size=0.2)


with open('./dataset/train.tsv', 'w', encoding='utf-8') as f:
    writer = csv.writer(f, delimiter='\t')
    writer.writerow(['label', 'body'])
    for i, x in enumerate(x_train):
        y = y_train[i]
        writer.writerow([y, x])
    print('training dataset size is ', len(x_train))


with open('./dataset/test.tsv',  'w', encoding='utf-8') as f:
    writer = csv.writer(f, delimiter='\t')
    writer.writerow(['label', 'body'])
    for i, x in enumerate(x_test):
        y = y_test[i]
        writer.writerow([y, x])
    print('training dataset size is ', len(x_test))

print('perpare the training and testing dataset successful')