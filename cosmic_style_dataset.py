import torch
import numpy as np
import pickle
from torch.utils.data import Dataset


class BaseDataset(Dataset):
    def __init__(self, dataset_name, hop, dataset_type):
        super(BaseDataset, self).__init__()
        self.dataset_name = dataset_name
        self.dataset_type = dataset_type
        self.hop = hop
        dataset_path = 'cosmic_data/' + dataset_name + '/data.pkl'
        graph_path = 'cosmic_data/' + dataset_name + '/' + dataset_name + '_graph_hop' + str(hop) + '.pkl'
        edge_attr_path = 'cosmic_data/' + dataset_name + '/' + dataset_name + '_edge_attr_' + dataset_type + '.pkl'
        data = pickle.load(open(dataset_path, 'rb'), encoding='latin1')
        data = data[dataset_type]
        self.r = data[0]
        self.label = data[1]
        self.spk = data[2]
        graph = pickle.load(open(graph_path, 'rb'), encoding='latin1')[dataset_type]
        self.edge_index = graph['edge_index']
        self.edge_type = graph['edge_type']
        self.cmsk = pickle.load(open(edge_attr_path, 'rb'), encoding='latin1')

    def __getitem__(self, item):
        selected_r = np.vstack(self.r[item])
        selected_label = self.label[item]
        selected_spk = self.spk[item]
        selected_edge_index = self.edge_index[item]
        selected_edge_type = self.edge_type[item]
        selected_cmsk = self.cmsk[item]
        selected_edge_attr = []
        selected_edge_relation = []
        for i in range(selected_edge_index.shape[1]):
            edge_i = selected_edge_index[0, i]
            eg_tp = selected_edge_type[i]
            selected_edge_attr.append(torch.tensor(selected_cmsk[edge_i][eg_tp], dtype=torch.float))

            edge_j = selected_edge_index[1, i]
            if edge_j <= edge_i:
                selected_edge_relation.append(2)
            else:
                if eg_tp == 'xWant':
                    selected_edge_relation.append(0)
                else:
                    selected_edge_relation.append(1)
        selected_r = torch.tensor(selected_r, dtype=torch.float)
        selected_label = torch.tensor(selected_label, dtype=torch.long)
        selected_spk = torch.tensor(selected_spk, dtype=torch.float)
        selected_edge_index = torch.tensor(selected_edge_index, dtype=torch.long)
        selected_edge_attr = torch.stack(selected_edge_attr, dim=0)
        selected_edge_relation = torch.tensor(selected_edge_relation, dtype=torch.long)

        return selected_r, selected_label, selected_edge_index, \
               selected_edge_attr, selected_edge_relation, selected_spk

    def __len__(self):
        return len(self.label)


def collate_fn(data):
    r = []
    label = []
    edge_index = []
    edge_attr = []
    edge_relation = []
    spk = []
    seq_len = []
    for d in data:
        r.append(d[0])
        label.append(d[1])
        edge_index.append(d[2])
        edge_attr.append(d[3])
        edge_relation.append(d[4])
        spk.append(d[5])
        seq_len.append(d[0].shape[0])
    return r, label, seq_len, edge_index, edge_attr, edge_relation, spk
