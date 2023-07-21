import torch
from sklearn.model_selection import train_test_split
import numpy as np
import pickle

def transfer_edge(node_num, edge_list):
    new_edge = np.zeros([node_num, node_num])
    for (st, ed) in edge_list:
        new_edge[st, ed] = 1
    return new_edge


def bsf_search(root):
    node_list, edge_list = [], []
    candidate = [root]
    candidate_count = 0
    now_id = 0
    while len(candidate):
        ast_node = candidate[0]
        candidate = candidate[1:]
        node_list.append(ast_node['node'].lower())
        for son in ast_node['children']:
            candidate_count += 1
            candidate.append(son)
            edge_list.append([now_id, candidate_count])
        now_id += 1
    return node_list, transfer_edge(len(node_list), edge_list)


def transfer_data(dataset):
    res = []
    for data in dataset:
        ast_1, ast_2, label = data
        node_1, graph_1 = bsf_search(ast_1)
        node_2, graph_2 = bsf_search(ast_2)
        res.append([node_1, graph_1, node_2, graph_2, label])
    return res


def main():
    data = torch.load('se_tasks/clone_detection/dataset/clone.dataset')
    data = transfer_data(data)
    train_data, test_data = train_test_split(data, test_size=0.2,)
    with open('se_tasks/clone_detection/dataset/train.pkl', 'wb') as f:
        pickle.dump(train_data, f)
    with open('se_tasks/clone_detection/dataset/test.pkl', 'wb') as f:
        pickle.dump(test_data, f)

    print('successful get the dataset')


if __name__ == '__main__':
    main()