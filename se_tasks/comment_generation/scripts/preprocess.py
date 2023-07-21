import pickle

from utils import BASEDICT


def build_vocab(file_name):
    with open(file_name, 'r') as f:
        dataset = f.readlines()
    word2index = BASEDICT.copy()
    for data in dataset:
        tk_list = data.split()
        for tk in tk_list:
            if tk not in word2index:
                word2index[tk] = len(word2index)
    return word2index


def token_dataset(dataset, tk2index):
    def token_data():
        for iii, tk in enumerate(tk_list):
            tk_list[iii] = tk2index[tk] if tk in tk2index else tk2index['____UNKNOW____']
        return tk_list

    for i, data in enumerate(dataset):
        tk_list = data.split()
        dataset[i] = token_data()
    return dataset


def build_dataset(code_file, comment_file, tk2index, word2index, out_file):
    data_dir = '../dataset/'
    code_file = data_dir + code_file
    comment_file = data_dir + comment_file
    out_file = data_dir + out_file

    with open(code_file, 'r') as f:
        code = f.readlines()
    with open(comment_file, 'r') as f:
        comment = f.readlines()
    code = token_dataset(code, tk2index)
    comment = token_dataset(comment, word2index)
    res = zip(code, comment)
    with open(out_file, 'wb') as f:
        pickle.dump(list(res), f)
    print('successful build', out_file)


def main():
    tk2index = build_vocab('../dataset/train.code')
    word2index = build_vocab('../dataset/train.comment')
    with open('../dataset/tk.pkl', 'wb') as f:
        pickle.dump([tk2index, word2index], f)

    print('successful build dictionary')
    build_dataset(
        'valid.code', 'valid.comment',
        tk2index, word2index, 'valid.pkl'
    )
    build_dataset(
        'test.code', 'test.comment',
        tk2index, word2index, 'test.pkl'
    )
    build_dataset(
        'train.code', 'train.comment',
        tk2index, word2index, 'train.pkl'
    )


if __name__ == '__main__':
    main()