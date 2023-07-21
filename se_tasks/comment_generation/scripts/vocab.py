import pickle
from utils import BASEDICT
from tqdm import tqdm
import torch
from utils import update_tk2index






def _test_func():
    with open('../dataset/train.code', 'r') as f:
        dataset = f.readlines()
    tk2index, _ = torch.load('../../../vec/100_2/code2vec.vec')
    num, tk2index = update_tk2index(tk2index)
    a, b = 0, 0
    unknow_set = set()
    for data in dataset[:10000]:
        tk_list = data.lower().split()
        for tk in tk_list:
            tk_seq = tk.split('_')
            for tk in tk_seq:
                if tk not in tk2index:
                    a += 1
                    unknow_set.add(tk)
                else:
                    b += 1
    print(a, b)


if __name__ == '__main__':
    _test_func()