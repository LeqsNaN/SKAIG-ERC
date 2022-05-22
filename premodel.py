import os
import sys
import argparse
import torch
import pickle

sys.path.append(os.getcwd())

import src.data.data as data
import src.data.config as cfg
import src.models.utils as model_utils
import src.train.batch as batch_utils

import src.interactive.functions as interactive


def set_inputs(input_event, data_loader, text_encoder):
    categories = ['xWant', 'oWant', 'xIntent', 'xEffect']
    prefix, suffix = data.atomic_data.do_example(text_encoder, input_event, None, True, None)
    prefix_len = len(prefix) if len(prefix) <= data_loader.max_event else data_loader.max_event
    
    batch = {'sequences': [], 'attention_mask': []}
    for cate in categories:
        XMB = torch.zeros(1, data_loader.max_event + 1).long().to(cfg.device)
        XMB[:, :prefix_len] = torch.LongTensor(prefix[:prefix_len])
        XMB[:, -1] = torch.LongTensor([text_encoder.encoder["<{}>".format(cate)]])
        batch['sequences'].append(XMB)
        batch['attention_mask'].append(data.atomic_data.make_attention_mask(XMB))
    return batch


def get_atomic_embs(input_event, model, opt, data_loader, text_encoder):
    with torch.no_grad():
        batch = set_inputs(input_event, data_loader, text_encoder)
        output = get_embs(batch, opt, model, data_loader, data_loader.max_event + 
                          data.atomic_data.num_delimiter_tokens['category'], 
                          data_loader.max_effect - 
                          data.atomic_data.num_delimiter_tokens['category'])
    return output


def get_embs(batch, opt, model, data_loader, start_idx, end_len):
    output4 = []
    for XMB, MMB in zip(batch['sequences'], batch['attention_mask']):
        XMB = XMB[:, :start_idx]
        MMB = MMB[:, :start_idx]
    
        XMB = model_utils.prepare_position_embeddings(opt, data_loader.vocab_encoder, XMB.unsqueeze(-1))
    
        output = model.transformer(XMB.unsqueeze(1), sequence_mask=MMB)
        output4.append(output.squeeze(0)[-1])

    return output4


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--model_file", type=str, default="pretrained_models/atomic_pretrained_model.pickle")
    parser.add_argument("--sampling_algorithm", type=str, default="help")
    parser.add_argument("--dataset", type=str, default='IEMOCAP')

    args = parser.parse_args()
    print(args)
    
    if args.dataset == "IEMOCAP":
        data1 = pickle.load(open('/data2/ljn/TRMSM/bert_data/IEMOCAP/IEMOCAP_data.pkl', 'rb'), encoding='utf-8')
        train_data = data1['train']
        test_data = data1['test']
        train_utt = train_data[0]
        test_utt = test_data[0]
    elif args.dataset == "MELD":
        data1 = pickle.load(open('/data2/ljn/TRMSM/bert_data/MELD/MELD_data.pkl', 'rb'), encoding='utf-8')
        train_data, dev_data, test_data = data1[0], data1[1], data1[2]
        train_utt = train_data[0]
        dev_utt = dev_data[0]
        test_utt = test_data[0]
    elif args.dataset == "EmoryNLP":
        data1 = pickle.load(open('/data2/ljn/TRMSM/bert_data/EmoryNLP/EmoryNLP_feature.pkl', 'rb'), encoding='utf-8')
        train_data, dev_data, test_data = data1[0], data1[2], data1[1]
        train_utt = train_data[0]
        dev_utt = dev_data[0]
        test_utt = test_data[0]
    elif args.dataset == "DailyDialog":
        data1 = pickle.load(open('/data2/ljn/TRMSM/bert_data/DailyDialog/DailyDialog_feature.pkl', 'rb'), encoding='utf-8')
        train_data, dev_data, test_data = data1[0], data1[1], data1[2]
        train_utt = train_data[0]
        dev_utt = dev_data[0]
        test_utt = test_data[0]

    opt, state_dict = interactive.load_model_file(args.model_file)

    data_loader, text_encoder = interactive.load_data("atomic", opt)

    n_ctx = data_loader.max_event + data_loader.max_effect
    n_vocab = len(text_encoder.encoder) + n_ctx
    model = interactive.make_model(opt, n_vocab, n_ctx, state_dict)

    if args.device != "cpu":
        cfg.device = int(args.device)
        cfg.do_gpu = True
        torch.cuda.set_device(cfg.device)
        model.cuda(cfg.device)
    else:
        cfg.device = "cpu"

    def all_process(utt):
        all_features = []
        for conv in utt:
            conv_feature = []
            for u in conv:
                u_feature = {}
                emb_output = get_atomic_embs(u, model, opt, data_loader, text_encoder)
                u_feature['xWant'] = emb_output[0]
                u_feature['oWant'] = emb_output[1]
                u_feature['xIntent'] = emb_output[2]
                u_feature['xEffect'] = emb_output[3]
                conv_feature.append(u_feature)
            all_features.append(conv_feature)
        return all_features
    
    train_feature = all_process(train_utt)
    test_feature = all_process(test_utt)
    if args.dataset == 'MELD' or args.dataset == 'EmoryNLP' or args.dataset == 'DailyDialog':
        dev_feature = all_process(dev_utt)
        pickle.dump(dev_feature, open('/data2/ljn/TRMSM/bert_data/'+ args.dataset + '/' + args.dataset + '_edge_attr_dev.pkl', 'wb'))
    pickle.dump(train_feature, open('/data2/ljn/TRMSM/bert_data/'+ args.dataset + '/' + args.dataset + '_edge_attr_train.pkl', 'wb'))
    pickle.dump(test_feature, open('/data2/ljn/TRMSM/bert_data/' + args.dataset + '/' + args.dataset + '_edge_attr_test.pkl', 'wb'))
