from torch.utils.data import Dataset
import torch
from numpy import random as rnd
import pickle
import random


class C2SDataSet(Dataset):
    def get_dataset(self, file_name, max_size):
        with open(file_name, 'rb') as f:
            dataset = pickle.load(f)
            dataset = dataset
            random.shuffle(dataset)
        if max_size is not None:
            dataset = dataset[:max_size]
        return dataset

    @staticmethod
    def transfer_dict(word2index):
        res = {}
        for tk in word2index:
            res[word2index[tk]] = tk
        return res

    def __init__(self, args, file_name, terminal_dict, path_dict,
                   target_dict, max_size, device):
        super(Dataset, self).__init__()
        self.data_set = self.get_dataset(file_name, max_size)
        self.size = len(self.data_set)

        self.target_dict = target_dict
        self.path_dict = path_dict
        self.terminal_dict = terminal_dict

        self.terminal_list = self.transfer_dict(terminal_dict)
        self.path_list = self.transfer_dict(path_dict)
        self.target_list = self.transfer_dict(target_dict)

        self.device = device
        self.max_context_length = args.context_length
        self.max_terminal_length = args.terminal_length
        self.max_path_length = args.path_length
        self.max_target_length = args.target_length

    def __len__(self):
        return self.size

    def make_up(self, max_len, now_len, PAD = 1):
        return [PAD] * (max_len - now_len)

    def __getitem__(self, index):
        data = self.data_set[index]
        sss, y = data

        sss_shuffled_index = [i for i in range(len(sss))]
        rnd.shuffle(sss_shuffled_index)
        sss_shuffled_index = sss_shuffled_index[:self.max_context_length]
        starts = [
            sss[i][0][:self.max_terminal_length] + self.make_up(self.max_terminal_length, len(sss[i][0]))
            for i in sss_shuffled_index
        ]
        paths = [
            sss[i][1][:self.max_path_length] + self.make_up(self.max_path_length, len(sss[i][1]))
            for i in sss_shuffled_index
        ]
        ends = [
            sss[i][2][:self.max_terminal_length] + self.make_up(self.max_terminal_length, len(sss[i][2]))
            for i in sss_shuffled_index
        ]

        start_mask = [
            [1] * len(sss[i][0][:self.max_terminal_length]) + [0] * (self.max_terminal_length - len(sss[i][0]))
            for i in sss_shuffled_index
        ]
        end_mask = [
            [1] * len(sss[i][2][:self.max_terminal_length]) + [0] * (self.max_terminal_length - len(sss[i][2]))
            for i in sss_shuffled_index
        ]

        context_mask = [1] * len(sss_shuffled_index)
        target = y[:self.max_target_length - 2]
        path_length = [
            len(sss[i][1][:self.max_path_length]) for i in sss_shuffled_index
        ]
        pad_length = self.max_context_length - len(context_mask)
        paths += [
            [1 for _ in range(self.max_path_length)]
            for j in range(pad_length)
        ]
        path_length += [1] * pad_length
        starts += [
            [1 for i in range(self.max_terminal_length)]
            for j in range(pad_length)
        ]
        start_mask += [
            [0 for i in range(self.max_terminal_length)]
            for j in range(pad_length)
        ]
        ends += [
            [1 for i in range(self.max_terminal_length)]
            for j in range(pad_length)
        ]
        end_mask += [
            [0 for i in range(self.max_terminal_length)]
             for j in range(pad_length)
        ]
        context_mask += [0] * pad_length

        target = [self.target_dict["____ST____"]] + target + [self.target_dict["____ED____"]]

        target_mask = [1] * (len(target)-1)  # sos
        target_mask += [0] * (self.max_target_length - len(target))
        target += [1] * (self.max_target_length - len(target))

        return torch.tensor(starts, dtype=torch.long).to(self.device),\
            torch.tensor(paths, dtype=torch.long).to(self.device),\
            torch.tensor(ends, dtype=torch.long).to(self.device),\
            torch.tensor(target, dtype=torch.long).to(self.device),\
            torch.tensor(context_mask, dtype=torch.float).to(self.device),\
            torch.tensor(start_mask, dtype=torch.float).to(self.device),\
            torch.tensor(path_length, dtype=torch.int64).to(self.device),\
            torch.tensor(end_mask, dtype=torch.float).to(self.device),\
            torch.tensor(target_mask, dtype=torch.float).to(self.device)
