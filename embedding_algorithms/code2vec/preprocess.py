import pickle
import json
import os
from utils import BASEDICT


def build_dict(dataset):
    base_dict = BASEDICT.copy()
    token2index, path2index, func2index = base_dict.copy(), base_dict.copy(), base_dict.copy()
    for i, data in enumerate(dataset):
        data = data.strip().lower()
        target, code_context = data.split()[0], data.split()[1:]
        func_name = target
        if func_name not in func2index:
            func2index[func_name] = len(func2index)
        for context in code_context:
            st, path, ed = context.split(',')
            if st not in token2index:
                token2index[st] = len(token2index)
            if ed not in token2index:
                token2index[ed] = len(token2index)
            if path not in path2index:
                path2index[path] = len(path2index)
    with open(DIR + 'tk.pkl', 'wb') as f:
        pickle.dump([token2index, path2index, func2index], f)
    print("finish dictionary build", len(token2index), len(path2index), len(func2index))


def tk2index(tk_dict, k):
    if k not in tk_dict:
        return tk_dict['____UNKNOW____']
    return tk_dict[k]


def norm_data(data_type):
    file_name = 'dataset/java-small/java-small.' + data_type + '.c2s'
    with open(file_name, 'r') as f:
        dataset = f.readlines()
    with open(DIR + 'tk.pkl', 'rb') as f:
        token2index, path2index, func2index = pickle.load(f)
    newdataset = []
    for i, data in enumerate(dataset):
        data = data.strip().lower()
        target, code_context = data.split()[0], data.split()[1:]
        func_name = target
        label = tk2index(func2index, func_name)
        newdata = []
        for context in code_context:
            st, path, ed = context.split(',')
            newdata.append(
                [tk2index(token2index, st), tk2index(path2index, path), tk2index(token2index, ed)]
            )
        newdataset.append([newdata, label])
    save_file = DIR + data_type + '.pkl'
    with open(save_file, 'wb') as f:
        pickle.dump(newdataset, f)
    print("finish normalize dataset", data_type)


def main():
    with open(DATA_DIR + 'java-small.train.c2s', 'r') as f:
        dataset = f.readlines()
        print('dataset number is ', len(dataset))
    if not os.path.isdir(DIR):
        os.mkdir(DIR)
    build_dict(dataset)
    norm_data('train')
    norm_data('val')
    norm_data('test')


if __name__ == '__main__':
    DIR = 'dataset/java-small-preprocess/'
    DATA_DIR = 'dataset/java-small/'
    main()
