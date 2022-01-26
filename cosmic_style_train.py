import random
import time
import argparse
import torch
import pickle
import numpy as np
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_

from cosmic_style_dataset import BaseDataset, collate_fn
from cosmic_style import MentalModel

from sklearn.metrics import accuracy_score, f1_score, classification_report


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_loaders(dataset_name, hop, batch_size=8, shuffle=True):
    trainset = BaseDataset(dataset_name, hop, 'train')
    devset = BaseDataset(dataset_name, hop, 'dev')
    testset = BaseDataset(dataset_name, hop, 'test')

    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=shuffle, num_workers=0,
                              collate_fn=collate_fn, worker_init_fn=seed_worker)
    dev_loader = DataLoader(devset, batch_size=batch_size, shuffle=shuffle, num_workers=0,
                            collate_fn=collate_fn, worker_init_fn=seed_worker)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=shuffle, num_workers=0,
                             collate_fn=collate_fn, worker_init_fn=seed_worker)
    return train_loader, dev_loader, test_loader


def train(model, loss_func, trainloader, devloader, testloader,
          n_epochs, optimizer, model_path, log_path, use_gpu, residual):
    model.train()

    f = open(log_path, 'a+', encoding='utf-8')

    best_fscore = 0
    best_mf1 = 0
    best_accuracy = 0
    best_report = None

    best_test_fscore = 0
    best_test_mf1 = 0
    best_test_accuracy = 0
    best_test_report = None

    early_stopping_step = 0

    for epoch in range(n_epochs):
        losses = []
        preds = []
        labels = []
        num_utt = 0
        print('Epoch {} start: '.format(epoch + 1))
        start_time = time.time()
        for data in trainloader:
            # (clen, slen), (clen)
            r, label, conv_len, edge_index, edge_attr, edge_relation, spkm = data
            logits = model(r, conv_len, edge_index, edge_attr, use_gpu, residual)
            label = torch.cat(label, dim=0)
            if use_gpu:
                label = label.cuda()

            loss = loss_func(logits, label)

            optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            pred_ = torch.argmax(torch.softmax(logits, dim=-1), dim=1)
            preds.append(pred_.cpu().numpy())
            labels.append(label.data.cpu().numpy())
            losses.append(loss.item() * label.size(0))
            num_utt += label.size(0)

        preds = np.concatenate(preds)
        labels = np.concatenate(labels)

        avg_loss = np.round(np.sum(losses) / num_utt, 4)
        avg_accuracy = round(accuracy_score(labels, preds) * 100, 5)
        avg_fscore = round(f1_score(labels, preds, average='weighted') * 100, 5)
        train_mf1 = round(f1_score(labels, preds, average='macro') * 100, 5)
        dev_accuracy, dev_fscore, dev_mf, dev_reports = evaluate(model, devloader, use_gpu, residual)
        test_accuracy, test_fscore, test_mf, test_reports = evaluate(model, testloader, use_gpu, residual)

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


def evaluate(model, dataloader, use_gpu, residual):
    model.eval()
    preds = []
    labels = []

    with torch.no_grad():
        for data in dataloader:
            r1, r2, r3, r4, label, conv_len, edge_index, edge_attr, edge_relation, spkm = data
            logits = model(r1, r2, r3, r4, conv_len, edge_index, edge_attr, use_gpu, residual)
            label = torch.cat(label, dim=0)
            pred_ = torch.argmax(torch.softmax(logits, dim=1), dim=1)
            preds.append(pred_.cpu().numpy())
            labels.append(label.data.numpy())
    preds = np.concatenate(preds)
    labels = np.concatenate(labels)

    avg_accuracy = round(accuracy_score(labels, preds) * 100, 5)

    avg_fscore = round(f1_score(labels, preds, average='weighted') * 100, 5)
    report_classes = classification_report(labels, preds, digits=4)
    mf1 = round(f1_score(labels, preds, average='macro') * 100, 5)

    model.train()

    return avg_accuracy, avg_fscore, mf1, report_classes


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('-dataset_name', type=str, default='meld')
    parser.add_argument('-hop', type=int, default=2)               # 4

    parser.add_argument('-batch_size', type=int, default=8)        # 16
    parser.add_argument('-lr', type=float, default=1e-4)
    parser.add_argument('-l2', type=float, default=3e-4)

    parser.add_argument('-n_epochs', type=int, default=50)
    parser.add_argument('--use_gpu', action='store_true')
    parser.add_argument('-seed', type=int, default=7)

    parser.add_argument('-sent_dim', type=int, default=200)        # 300
    parser.add_argument('-ff_dim', type=int, default=200)          # 600
    parser.add_argument('-heads', type=int, default=4)             # 6
    parser.add_argument('-at_dropout', type=float, default=0.1)
    parser.add_argument('-dropout', type=float, default=0.5)
    parser.add_argument('-num_layers', type=int, default=2)        # 5
    parser.add_argument('-edge_dim', type=int, default=200)        # 300

    parser.add_argument('--bias', action='store_true')
    parser.add_argument('--edge_mapping', action='store_true')
    parser.add_argument('--beta', action='store_true')
    parser.add_argument('--root_weight', action='store_true')
    parser.add_argument('--residual', action='store_true')
    parser.add_argument('--shuffle', action='store_true')
    parser.add_argument('--reweight', action='store_true')

    parser.add_argument('-index', type=int, default=1)

    args = parser.parse_args()
    print(args)

    dataset_name = args.dataset_name
    model_index = str(args.index)
    model_dir = 'cosmic_style/' + dataset_name + '/model'
    log_dir = 'cosmic_style/' + dataset_name + '/logs/log'

    model_path = model_dir + model_index + '.pkl'
    log_path = log_dir + model_index + '.txt'

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

    bias = args.bias
    edge_mapping = args.edge_mapping
    beta = args.beta
    root_weight = args.root_weight

    model = MentalModel(args.sent_dim, args.ff_dim, args.sent_dim,
                        args.heads, args.edge_dim, bias, args.at_dropout,
                        args.dropout, args.num_layers, edge_mapping,
                        beta, root_weight, num_class)
    if use_gpu:
        model = model.cuda()

    train_loader, dev_loader, test_loader = get_loaders(dataset_name, args.hop, batch_size, args.shuffle)
    if args.reweight:
        loss_weights = torch.tensor([0.304, 1.197, 5.47, 1.954, 0.848, 5.425, 1.219], dtype=torch.float)
        loss_func = nn.CrossEntropyLoss(weight=loss_weights, reduction='mean')
    else:
        loss_func = nn.CrossEntropyLoss(reduction='mean')

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=l2)

    train(model, loss_func, train_loader, dev_loader,
          test_loader, n_epochs, optimizer, model_path,
          log_path, use_gpu, residual=args.residual)


if __name__ == '__main__':
    main()
