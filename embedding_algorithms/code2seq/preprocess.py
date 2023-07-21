import pickle
import json
import os
from multiprocessing import Process
from utils import BASEDICT


def get_tk_list(token_str):
    return token_str.split('|')


def update_dict(old_dict, tk_list):
    for tk in tk_list:
        if tk not in old_dict:
            old_dict[tk] = len(old_dict)
    return old_dict


def build_dict(dataset):
    base_dict = BASEDICT.copy()
    token2index, path2index, func2index = base_dict.copy(), base_dict.copy(), base_dict.copy()
    for i, data in enumerate(dataset):
        data = data.strip().lower()
        target, code_context = data.split()[0], data.split()[1:]
        target_list = target.split('|')
        func2index = update_dict(func2index, target_list)
        for context in code_context:
            st, path, ed = context.split(',')
            st_list = get_tk_list(st)
            path_list = get_tk_list(path)
            ed_list = get_tk_list(ed)
            token2index = update_dict(token2index, st_list)
            token2index = update_dict(token2index, ed_list)
            path2index = update_dict(path2index, path_list)
    with open(DIR + 'tk.pkl', 'wb') as f:
        pickle.dump([token2index, path2index, func2index], f)
    print("finish dictionary build", len(token2index), len(path2index), len(func2index))


def tklist2index(tk_dict, k_list):
    res = []
    for k in k_list:
        if k not in tk_dict:
            res.append(tk_dict['____UNKNOW____'])
        else:
            res.append(tk_dict[k])
    return res


def norm_data(data_type):
    file_name = DATA_DIR + '/java-small.' + data_type + '.c2s'
    with open(file_name, 'r') as f:
        dataset = f.readlines()
    with open(DIR + 'tk.pkl', 'rb') as f:
        token2index, path2index, func2index = pickle.load(f)
    newdataset = []
    for i, data in enumerate(dataset):
        data = data.strip().lower()
        target, code_context = data.split()[0], data.split()[1:]
        target_list = get_tk_list(target)
        label = tklist2index(func2index, target_list)

        newdata = []
        for context in code_context:
            st, path, ed = context.split(',')
            st_list = get_tk_list(st)
            path_list = get_tk_list(path)
            ed_list = get_tk_list(ed)
            newdata.append(
                [
                    tklist2index(token2index, st_list),
                    tklist2index(path2index, path_list),
                    tklist2index(token2index, ed_list)
                ]
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
    DIR = '../../dataset/java-small-code2seq/'
    DATA_DIR = '../../dataset/java-small/'
    main()
