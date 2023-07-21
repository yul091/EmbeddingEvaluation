import pickle

with open('dataset/java-small/java-small.dict.c2s', 'br') as f:
    dict = pickle.load(f)

with open('dataset/java-small/java-small.test.c2s', 'r') as f:
    data = f.readlines()
print()