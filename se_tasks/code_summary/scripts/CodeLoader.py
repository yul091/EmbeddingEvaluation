from torch.utils.data import Dataset
import pickle
import random


class CodeLoader(Dataset):
    def dict2list(self, token2index):
        res = {}
        for k in token2index:
            res[token2index[k]] = k
        return res

    def transferdata(self, index2token, tk2num):
        def transfer(context):
            tk = ''.join(index2token[context].split('|')).lower()
            return tk2num[tk] if tk in tk2num else tk2num['____UNKNOW____']
        for i, data in enumerate(self.dataset):
            code_context, target = data
            code_context = [
                [transfer(context[0]), context[1], transfer(context[2])]
                for context in code_context
            ]
            self.dataset[i] = [code_context, target]

    def __init__(self, file_name, max_size, token2index, tk2num):
        index2token = self.dict2list(token2index)
        with open(file_name, 'rb') as f:
            dataset = pickle.load(f)
            self.dataset = dataset
            random.shuffle(self.dataset)
        if max_size is not None:
            self.dataset = self.dataset[:max_size]
        if tk2num is not None:
            self.transferdata(index2token, tk2num)

    def __getitem__(self, index):
        data = self.dataset[index]
        code_context, target = data
        return code_context, target

    def __len__(self):
        return len(self.dataset)
