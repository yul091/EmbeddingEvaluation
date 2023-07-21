import pickle

from torch.utils.data import Dataset


class CommentData(Dataset):
    def __init__(self, data_path, tk2index, word2index, d_word2index, embed_vec, max_size):
        with open(data_path, 'rb') as f:
            ori_data = pickle.load(f)
            if max_size is not None:
                ori_data = ori_data[:max_size]
        self.tk2index = tk2index
        self.word2index = word2index
        self.d_word2index = d_word2index
        self.embed_vec = embed_vec

        if d_word2index is not None:
            return  #todo add token transformation
        self.dataset = ori_data

    def __getitem__(self, index):
        return self.dataset[index]

    def __len__(self):
        return len(self.dataset)


