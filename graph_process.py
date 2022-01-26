import pickle
import numpy as np

# data = pickle.load(open('bert_data/IEMOCAP/IEMOCAP_data.pkl', 'rb'), encoding='utf-8')
# train_data, test_data = data['train'], data['test']
#
# train_text, train_label = train_data[0], train_data[1]
# test_text, test_label = test_data[0], test_data[1]
#
# train_spk = train_data[2]
# test_spk = test_data[2]


def get_graph(spk_list, max_hip=1):
    edge_index = []
    edge_types = []
    for conv in spk_list:
        edge_idx = []
        edge_type = []
        for i in range(len(conv)):
            xwant_hip = 0
            owant_hip = 0
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
                if conv[j] != s_i and owant_hip < max_hip:
                    edge_idx.append([i, j])
                    edge_type.append('oWant')
                    owant_hip += 1
                if xwant_hip == max_hip and owant_hip == max_hip:
                    break
                j += 1
        edge_index.append(np.transpose(np.array(edge_idx)))
        edge_types.append(edge_type)
    graph = {'edge_index': edge_index, 'edge_type': edge_types}
    return graph


# train_graph = get_graph(train_spk, max_hip=10)
# test_graph = get_graph(test_spk, max_hip=10)
# data = {'train': [train_text, train_label, train_spk, train_graph],
#         'test': [test_text, test_label, test_spk, test_graph]}
# pickle.dump(data, open('bert_data/IEMOCAP/IEMOCAP_graph_hip10_new.pkl', 'wb'))


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


data = pickle.load(open('bert_data/MELD/MELD_data.pkl', 'rb'), encoding='utf-8')
train_data, dev_data, test_data = data
train_utt, train_spk, train_label = train_data
dev_utt, dev_spk, dev_label = dev_data
test_utt, test_spk, test_label = test_data

train_graph = get_graph_multi_speaker(train_spk, 7)
dev_graph = get_graph_multi_speaker(dev_spk, 7)
test_graph = get_graph_multi_speaker(test_spk, 7)

data = {'train': [train_utt, train_label, train_spk, train_graph],
        'dev': [dev_utt, dev_label, dev_spk, dev_graph],
        'test': [test_utt, test_label, test_spk, test_graph]}

pickle.dump(data, open('bert_data/MELD/MELD_graph_hip7_new.pkl', 'wb'))


# data = pickle.load(open('bert_data/EmoryNLP/EmoryNLP_feature.pkl', 'rb'), encoding='utf-8')
# train_data, test_data, dev_data = data
# train_utt, train_spk, train_label = train_data
# dev_utt, dev_spk, dev_label = dev_data
# test_utt, test_spk, test_label = test_data

# train_graph = get_graph_multi_speaker(train_spk, 7)
# dev_graph = get_graph_multi_speaker(dev_spk, 7)
# test_graph = get_graph_multi_speaker(test_spk, 7)

# data = {'train': [train_utt, train_label, train_spk, train_graph],
#        'dev': [dev_utt, dev_label, dev_spk, dev_graph],
#        'test': [test_utt, test_label, test_spk, test_graph]}

# pickle.dump(data, open('bert_data/EmoryNLP/EmoryNLP_graph_hip7_new.pkl', 'wb'))

# data = pickle.load(open('bert_data/DailyDialog/DailyDialog_feature.pkl', 'rb'), encoding='utf-8')
# train_data, dev_data, test_data = data
# train_utt, train_label, train_spk = train_data
# dev_utt, dev_label, dev_spk = dev_data
# test_utt, test_label, test_spk = test_data
#
# train_graph = get_graph(train_spk, 9)
# dev_graph = get_graph(dev_spk, 9)
# test_graph = get_graph(test_spk, 9)
#
# data = {'train': [train_utt, train_label, train_spk, train_graph],
#         'dev': [dev_utt, dev_label, dev_spk, dev_graph],
#         'test': [test_utt, test_label, test_spk, test_graph]}
#
# pickle.dump(data, open('bert_data/DailyDialog/DailyDialog_graph_hip9_new.pkl', 'wb'))
