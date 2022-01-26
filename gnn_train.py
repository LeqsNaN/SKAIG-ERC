import random
import time
import argparse
import torch
import pickle
import numpy as np
import torch.nn as nn

from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torch.nn.utils import clip_grad_norm_

from gnn_dataset import BaseDataset, collate_fn, collate_fn_batch
from gnn import MentalModel, BatchMentalModel
from gnn_for_meld_emorynlp import BatchMentalModelResidual

from transformers import AutoTokenizer
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import get_constant_schedule, get_constant_schedule_with_warmup

from sklearn.metrics import accuracy_score, f1_score, classification_report


def get_train_valid_sampler(trainset, valid=0.1):
    size = len(trainset)
    idx = list(range(size))
    split = int(valid * size)
    return SubsetRandomSampler(idx[split:]), SubsetRandomSampler(idx[:split])


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_loaders(dataset_name, hip, batch_size=8, pretrained_model='bert-base-uncased', valid=0.1, shuffle=True):
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
    trainset = BaseDataset(dataset_name, hip, 'train', tokenizer)
    testset = BaseDataset(dataset_name, hip, 'test', tokenizer)

    if dataset_name == 'IEMOCAP':
        train_sampler, valid_sampler = get_train_valid_sampler(trainset, valid)
        train_loader = DataLoader(trainset,
                                  batch_size=1,
                                  sampler=train_sampler,
                                  num_workers=0,
                                  collate_fn=collate_fn,
                                  worker_init_fn=seed_worker)
        dev_loader = DataLoader(trainset, batch_size=1, sampler=valid_sampler, num_workers=0, collate_fn=collate_fn, worker_init_fn=seed_worker)
        test_loader = DataLoader(testset, batch_size=1, shuffle=shuffle, num_workers=0, collate_fn=collate_fn, worker_init_fn=seed_worker)
    else:
        devset = BaseDataset(dataset_name, hip, 'dev', tokenizer)
        train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=shuffle, num_workers=0, collate_fn=collate_fn_batch, worker_init_fn=seed_worker)
        dev_loader = DataLoader(devset, batch_size=batch_size, shuffle=shuffle, num_workers=0, collate_fn=collate_fn_batch, worker_init_fn=seed_worker)
        test_loader = DataLoader(testset, batch_size=batch_size, shuffle=shuffle, num_workers=0, collate_fn=collate_fn_batch, worker_init_fn=seed_worker)
    return train_loader, dev_loader, test_loader


def train(dataset_name, model, loss_func, trainloader, devloader, testloader, n_epochs, optimizer,
          scheduler, training_step, model_path, log_path, metric_path, use_gpu, window, mode):
    model.train()

    f = open(log_path, 'a+', encoding='utf-8')

    train_loss_list = []
    train_f1_list = []
    dev_f1_list = []
    test_f1_list = []

    best_fscore = 0
    best_mf1 = 0
    best_accuracy = 0
    best_report = None

    best_test_fscore = 0
    best_test_mf1 = 0
    best_test_accuracy = 0
    best_test_report = None

    step = 0
    early_stopping_step = 0

    for epoch in range(n_epochs):
        losses = []
        preds = []
        labels = []
        masks = []
        num_utt = 0
        print('Epoch {} start: '.format(epoch + 1))
        if step > training_step:
            break
        start_time = time.time()
        for data in trainloader:
            # (clen, slen), (clen)
            textf, wrdm, label, uttm, spkm, edge_index, edge_attr, _, _ = data
            if dataset_name == 'IEMOCAP':
                if use_gpu:
                    textf = textf.cuda()
                    wrdm = wrdm.cuda()
                    uttm = uttm.cuda()
                    spkm = spkm.cuda()
                    edge_index = edge_index.cuda()
                    edge_attr = edge_attr.cuda()
                logits = model(textf, wrdm, uttm, spkm, window, mode, edge_index, edge_attr, residual=False)
            else:
                conv_len = [int(torch.sum(um).item()) for um in uttm]  # torch.sum(uttm, dim=1).numpy().tolist()
                if dataset_name == 'DailyDialog':
                    logits = model(textf, wrdm, conv_len, uttm, spkm, window, mode, edge_index, edge_attr, use_gpu, residual=False)
                else:
                    logits = model(textf, wrdm, conv_len, edge_index, edge_attr, use_gpu)
                # logits = model(textf, wrdm, conv_len, edge_index, edge_attr, use_gpu)
                label = torch.cat(label, dim=0)
            if use_gpu:
                label = label.cuda()
            # loss = loss_func(torch.log_softmax(logits, dim=-1), label)
            loss = loss_func(logits, label)

            model.zero_grad()
            loss.backward()
            clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            pred_ = torch.argmax(torch.softmax(logits, dim=-1), dim=1)
            preds.append(pred_.cpu().numpy())
            labels.append(label.data.cpu().numpy())
            losses.append(loss.item() * label.size(0))
            num_utt += label.size(0)
            step += 1
            if step > training_step:
                break

        preds = np.concatenate(preds)
        labels = np.concatenate(labels)

        avg_loss = np.round(np.sum(losses) / num_utt, 4)
        avg_accuracy = round(accuracy_score(labels, preds) * 100, 5)
        if dataset_name == 'DailyDialog':
            avg_fscore = round(f1_score(labels, preds, labels=[0, 2, 3, 4, 5, 6], average='micro') * 100, 5)
        else:
            avg_fscore = round(f1_score(labels, preds, average='weighted') * 100, 5)
        train_mf1 = round(f1_score(labels, preds, average='macro') * 100, 5)
        dev_accuracy, dev_fscore, dev_mf, dev_reports = evaluate(dataset_name, model, devloader, use_gpu, window=window, mode=mode)
        test_accuracy, test_fscore, test_mf, test_reports = evaluate(dataset_name, model, testloader, use_gpu, window=window, mode=mode)

        if dev_fscore > best_fscore:
            best_fscore = dev_fscore
            best_mf1 = dev_mf
            best_accuracy = dev_accuracy
            best_report = dev_reports

            best_test_fscore = test_fscore
            best_test_mf1 = test_mf
            best_test_accuracy = test_accuracy
            best_test_report = test_reports

            early_stopping_step = 0
            # torch.save(model, model_path)
        else:
            early_stopping_step += 1

        train_loss_list.append(avg_loss)
        train_f1_list.append(avg_fscore)
        dev_f1_list.append(dev_fscore)
        test_f1_list.append(test_fscore)

        log = 'Train: Epoch {} Loss {}, ACC {}, F1 {}, mF {}'.format(epoch + 1, avg_loss,
                                                                     avg_accuracy, avg_fscore, train_mf1)
        print(log)
        f.write(log + '\n')
        log = 'Validation: ACC {}, F1 {}, mF {}'.format(dev_accuracy, dev_fscore, dev_mf)
        print(log)
        f.write(log + '\n')
        log = 'Test: ACC {}, F1 {}, mF {}'.format(test_accuracy, test_fscore, test_mf)
        print(log)
        f.write(log + '\n')
        print('Epoch {} finished. Elapse {}'.format(epoch + 1, round(time.time() - start_time, 4)))
        if early_stopping_step == 10:
            break
    print('----------------------------------------------')
    f.write('----------------------------------------------')
    log = '\n\n[DEV] best ACC {}, F1 {}, mF {}'.format(best_accuracy, best_fscore, best_mf1)
    f.write(log + '\n')
    print(log)
    f.write(best_report)
    log = '[TEST] best ACC {}, F1 {}, mF {}'.format(best_test_accuracy, best_test_fscore, best_test_mf1)
    f.write(log + '\n')
    print(log)
    f.write(best_test_report)
    f.write('----------------------------------------------\n')
    f.close()
    dump_data = [train_loss_list, train_f1_list, dev_f1_list, test_f1_list]
    pickle.dump(dump_data, open(metric_path, 'wb'))


def evaluate(dataset_name, model, dataloader, use_gpu, window, mode):
    model.eval()
    preds = []
    labels = []

    with torch.no_grad():
        for data in dataloader:
            textf, wrdm, label, uttm, spkm, edge_index, edge_attr, _, _ = data
            if dataset_name == 'IEMOCAP':
                if use_gpu:
                    textf = textf.cuda()
                    wrdm = wrdm.cuda()
                    uttm = uttm.cuda()
                    spkm = spkm.cuda()
                    edge_index = edge_index.cuda()
                    edge_attr = edge_attr.cuda()
                logits = model(textf, wrdm, uttm, spkm, window, mode, edge_index, edge_attr, residual=False)
            else:
                conv_len = [int(torch.sum(um).item()) for um in uttm]  # torch.sum(uttm, dim=1).numpy().tolist()
                if dataset_name == 'DailyDialog':
                    logits = model(textf, wrdm, conv_len, uttm, spkm, window, mode, edge_index, edge_attr, use_gpu, residual=False)
                else:
                    logits = model(textf, wrdm, conv_len, edge_index, edge_attr, use_gpu)
                # logits = model(textf, wrdm, conv_len, edge_index, edge_attr, use_gpu)
                label = torch.cat(label, dim=0)
            pred_ = torch.argmax(torch.softmax(logits, dim=1), dim=1)
            preds.append(pred_.cpu().numpy())
            labels.append(label.data.numpy())
    preds = np.concatenate(preds)
    labels = np.concatenate(labels)

    avg_accuracy = round(accuracy_score(labels, preds) * 100, 5)

    if dataset_name == 'DailyDialog':
        avg_fscore = round(f1_score(labels, preds, labels=[0, 2, 3, 4, 5, 6], average='micro') * 100, 5)
        report_classes = classification_report(labels, preds, labels=[0, 2, 3, 4, 5, 6], digits=4)
    else:
        avg_fscore = round(f1_score(labels, preds, average='weighted') * 100, 5)
        report_classes = classification_report(labels, preds, digits=4)
    mf1 = round(f1_score(labels, preds, average='macro') * 100, 5)

    model.train()

    return avg_accuracy, avg_fscore, mf1, report_classes


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('-dataset_name', type=str, default='IEMOCAP')
    parser.add_argument('-hip', type=int, default=1)
    parser.add_argument('-valid', type=float, default=0.1)

    parser.add_argument('-batch_size', type=int, default=8)
    parser.add_argument('-lr', type=float, default=1e-5)
    parser.add_argument('-l2', type=float, default=0.)

    parser.add_argument('-finetune_lr', type=float, default=1e-5)
    parser.add_argument('-normal_lr', type=float, default=1e-4)
    parser.add_argument('-n_epochs', type=int, default=50)
    parser.add_argument('-warmup_step', type=int, default=1000)
    parser.add_argument('-training_step', type=int, default=10000)
    parser.add_argument('-use_gpu', type=bool, default=True)
    parser.add_argument('-schedule', type=str, default='linear')
    parser.add_argument('-seed', type=int, default=7)

    parser.add_argument('-pretrain', type=str, default='roberta-large')
    parser.add_argument('-encoder_mode', type=str, default='maxpooling')
    parser.add_argument('-sent_dim', type=int, default=300)
    parser.add_argument('-tr_ff_dim', type=int, default=300)
    parser.add_argument('-tr_nhead', type=int, default=6)
    parser.add_argument('-tr_dropout', type=float, default=0.1)
    parser.add_argument('-tr_num_layer', type=int, default=6)
    parser.add_argument('-num_block', type=int, default=3)
    parser.add_argument('-max_len', type=int, default=120)
    parser.add_argument('-attn_mask', type=int, default=1)
    parser.add_argument('-bidirectional', type=int, default=1)

    parser.add_argument('-window', type=int, default=10)
    parser.add_argument('-mode', type=str, default='uso')
    parser.add_argument('-att_type', type=str, default='par')

    parser.add_argument('-cn_nhead', type=int, default=6)
    parser.add_argument('-cn_ff_dim', type=int, default=600)
    parser.add_argument('-cn_dropout', type=float, default=0.1)
    parser.add_argument('-edge_dim', type=int, default=300)
    parser.add_argument('-bias', type=int, default=0)
    parser.add_argument('-cn_num_layer', type=int, default=1)
    parser.add_argument('-edge_mapping', type=int, default=1)
    parser.add_argument('-beta', type=int, default=1)
    parser.add_argument('-root_weight', type=int, default=1)
    parser.add_argument('-residual_type', type=str, default='none')

    parser.add_argument('-choice', type=str, default='cn')
    parser.add_argument('-index', type=int, default=1)

    args = parser.parse_args()
    print(args)

    dataset_name = args.dataset_name
    model_index = str(args.index)
    model_dir = dataset_name + '_C/model'
    log_dir = dataset_name + '_C/logs/log'
    metric_dir = dataset_name + '_C/logs/metric'

    model_path = model_dir + model_index + '.pkl'
    log_path = log_dir + model_index + '.txt'
    metric_path = metric_dir + model_index + '.pkl'

    f = open(log_path, 'a+', encoding='utf-8')
    f.write(str(args) + '\n\n')
    f.close()

    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.enabled = False
    # torch.use_deterministic_algorithms(True)

    batch_size = args.batch_size
    n_epochs = args.n_epochs
    lr = args.lr
    l2 = args.l2

    use_gpu = args.use_gpu

    if dataset_name == 'IEMOCAP':
        num_class = 6
    else:
        num_class = 7

    if args.bidirectional == 0:
        bidirectional = False
    else:
        bidirectional = True

    if args.attn_mask == 0:
        attn_mask = False
    else:
        attn_mask = True

    if args.bias == 0:
        bias = False
    else:
        bias = True
    if args.edge_mapping == 0:
        edge_mapping = False
    else:
        edge_mapping = True
    if args.beta == 0:
        beta = False
    else:
        beta = True
    if args.root_weight == 0:
        root_weight = False
    else:
        root_weight = True

    choice = args.choice

    if dataset_name == 'IEMOCAP':
        model = MentalModel(args.pretrain, args.encoder_mode, args.sent_dim, args.tr_nhead,
                            args.tr_ff_dim, args.tr_dropout, attn_mask, args.tr_num_layer,
                            args.max_len, num_class, bidirectional, args.num_block,
                            args.cn_nhead, args.cn_ff_dim, args.cn_dropout, args.edge_dim, bias,
                            args.cn_num_layer, edge_mapping, beta, root_weight, choice)
    elif dataset_name == 'DailyDialog':
        model = BatchMentalModel(args.pretrain, args.encoder_mode, args.sent_dim, args.tr_nhead,
                                 args.tr_ff_dim, args.tr_dropout, attn_mask, args.tr_num_layer,
                                 args.max_len, num_class, bidirectional, args.num_block,
                                 args.cn_nhead, args.cn_ff_dim, args.cn_dropout, args.edge_dim, bias,
                                 args.cn_num_layer, edge_mapping, beta, root_weight, choice)
    else:
        model = BatchMentalModelResidual(args.pretrain, args.encoder_mode, args.sent_dim,
                                         args.cn_ff_dim, args.cn_nhead, args.cn_dropout,
                                         args.edge_dim, num_class, bias, args.cn_num_layer,
                                         edge_mapping, beta, root_weight, args.residual_type)
    if use_gpu:
        model = model.cuda()

    train_loader, dev_loader, test_loader = get_loaders(dataset_name, args.hip, batch_size, args.pretrain, args.valid, True)
    loss_func = nn.CrossEntropyLoss(reduction='mean')

    # if choice == 'both':
    #     param_group = [{'params': model.uttrenc.encoder.parameters(), 'lr': args.finetune_lr},
    #                    {'params': list(model.uttrenc.mapping.parameters()) + list(model.tcn.parameters()) +
    #                     list(model.convenc.parameters()) + list(model.classifier.parameters()), 'lr': args.normal_lr}]
    # elif choice == 'tr':
    #     param_group = [{'params': model.uttrenc.encoder.parameters(), 'lr': args.finetune_lr},
    #                    {'params': list(model.uttrenc.mapping.parameters()) + list(model.convenc.parameters()) +
    #                     list(model.classifier.parameters()), 'lr': args.normal_lr}]
    # elif choice == 'cn':
    #     param_group = [{'params': model.uttrenc.encoder.parameters(), 'lr': args.finetune_lr},
    #                    {'params': list(model.uttrenc.mapping.parameters()) + list(model.tcn.parameters()) +
    #                     list(model.classifier.parameters()), 'lr': args.normal_lr}]
    # else:
    #     raise NotImplementedError()
    # optimizer = AdamW(param_group, weight_decay=l2)

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=l2)
    if args.schedule == 'linear':
        scheduler = get_linear_schedule_with_warmup(optimizer, args.warmup_step, args.training_step)
    elif args.schedule == 'warmup':
        scheduler = get_constant_schedule_with_warmup(optimizer, args.warmup_step)
    else:
        scheduler = get_constant_schedule(optimizer)

    train(dataset_name, model, loss_func, train_loader, dev_loader, test_loader, n_epochs, optimizer, scheduler,
          args.training_step, model_path, log_path, metric_path, use_gpu, window=args.window, mode=args.mode)


if __name__ == '__main__':
    main()
