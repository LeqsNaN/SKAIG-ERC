import numpy as np
import pickle


def get_graph_multi_speaker(spk_list, max_hip=1):
    edge_index = []
    edge_types = []
    for conv in spk_list:
        edge_idx = []
        edge_type = []
        num_spk = np.unique(conv)
        for i in range(len(conv)):
            xwant_hip = 0
            owant_hip = [0 for spk in range(np.max(num_spk) + 1)]
            xintent_hip = 0
            s_i = conv[i]
            j = i - 1
            pre_edge_idx = []
            pre_edge_type = []
            while j >= 0:
                if conv[j] == s_i and xintent_hip < max_hip:
                    pre_edge_idx.append([i, j])
                    pre_edge_type.append('xIntent')
                    xintent_hip += 1
                if xintent_hip == max_hip:
                    break
                j -= 1
            for k in range(len(pre_edge_idx) - 1, -1, -1):
                edge_idx.append(pre_edge_idx[k])
                edge_type.append(pre_edge_type[k])
            edge_idx.append([i, i])
            edge_type.append('xEffect')
            j = i + 1
            while j < len(conv):
                if conv[j] == s_i and xwant_hip < max_hip:
                    edge_idx.append([i, j])
                    edge_type.append('xWant')
                    xwant_hip += 1
                if conv[j] != s_i:
                    if owant_hip[conv[j]] < max_hip:
                        edge_idx.append([i, j])
                        edge_type.append('oWant')
                        owant_hip[conv[j]] += 1
                j += 1
        edge_index.append(np.transpose(np.array(edge_idx)))
        edge_types.append(edge_type)
    graph = {'edge_index': edge_index, 'edge_type': edge_types}
    return graph


data = pickle.load(open('cosmic_data/meld/data.pkl', 'rb'), encoding='latin1')
train_data = data['train']
dev_data = data['dev']
test_data = data['test']

train_spk = train_data[5]
dev_spk = dev_data[5]
test_spk = test_data[5]

train_graph = get_graph_multi_speaker(train_spk, 1)
dev_graph = get_graph_multi_speaker(dev_spk, 1)
test_graph = get_graph_multi_speaker(test_spk, 1)

graph = {'train': train_graph, 'dev': dev_graph, 'test': test_graph}
pickle.dump(graph, open('cosmic_data/meld/meld_graph_hop1.pkl', 'wb'))
