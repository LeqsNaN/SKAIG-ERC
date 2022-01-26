import torch
import torch.nn as nn
from gnn import TRMCN
from gnn_uttenc import UtteranceEncoder


# class BatchMentalModelResidual(nn.Module):
#     def __init__(self, encoder_type='roberta-large', encoder_mode='maxpooling',
#                  sent_dim=300, ff_dim=600, nhead=6, dropout=0.1, edge_dim=300,
#                  num_class=7, bias=False, num_layer=2, edge_mapping=True,
#                  beta=True, root_weight=True, residual_type='cat'):
#         super(BatchMentalModelResidual, self).__init__()
#         self.utterenc = UtteranceEncoder(encoder_type, encoder_mode, sent_dim)
#         self.tcn = TRMCN(sent_dim, ff_dim, sent_dim // nhead, nhead, dropout,
#                          edge_dim, bias, num_layer, edge_mapping, beta, root_weight)
#         self.residual_type = residual_type
#         assert residual_type in ['cat', 'sum', 'none'], 'no such residual connection type. '
#         if residual_type == 'cat':
#             self.classifier = nn.Linear(2*sent_dim, num_class)
#         else:
#             self.classifier = nn.Linear(sent_dim, num_class)
#
#     def forward(self, conversations, masks, conv_len=None, edge_indices=None, edge_attrs=None, use_gpu=True):
#         sent_emb = self.utterenc(conversations, masks, use_gpu)
#         edge_attr = torch.cat(edge_attrs, dim=0)
#         graph_input = torch.cat(sent_emb, dim=0)
#         batch = []
#         cumbatch = []
#         count = 0
#         for i, l in enumerate(conv_len):
#             num_edge = int(edge_indices[i].size(1))
#             batch += [i] * num_edge
#             cumbatch += [count] * num_edge
#             count += l
#         batch = torch.tensor(batch, dtype=torch.long)
#         cumbatch = torch.tensor([cumbatch, cumbatch], dtype=torch.long)
#         edge_index = torch.cat(edge_indices, dim=1)
#         edge_index = edge_index + cumbatch
#         if use_gpu:
#             edge_index = edge_index.cuda()
#             edge_attr = edge_attr.cuda()
#         mental_emb = self.tcn(graph_input, edge_index, edge_attr)
#         if self.residual_type == 'cat':
#             mental_emb = torch.cat([mental_emb, graph_input], dim=1)
#         elif self.residual_type == 'sum':
#             mental_emb = mental_emb + graph_input
#         logits = self.classifier(mental_emb)
#         return logits


class BatchMentalModelResidual(nn.Module):
    def __init__(self, encoder_type='roberta-large', encoder_mode='maxpooling',
                 sent_dim=300, ff_dim=600, nhead=6, dropout=0.1, edge_dim=300,
                 num_class=7, bias=False, num_layer=2, edge_mapping=True,
                 beta=True, root_weight=True, residual_type='cat'):
        super(BatchMentalModelResidual, self).__init__()
        self.utterenc = UtteranceEncoder(encoder_type, encoder_mode, sent_dim)
        self.tcn = TRMCN(sent_dim, ff_dim, sent_dim // nhead, nhead, dropout,
                         edge_dim, bias, num_layer, edge_mapping, beta, root_weight)
        self.residual_type = residual_type
        assert residual_type in ['cat', 'sum', 'none'], 'no such residual connection type. '
        if residual_type == 'cat':
            self.classifier = nn.Linear(2*sent_dim, num_class)
        else:
            self.classifier = nn.Linear(sent_dim, num_class)

    def forward(self, conversations, masks, conv_len=None, edge_indices=None, edge_attrs=None, use_gpu=True):
        sent_emb = self.utterenc(conversations, masks, use_gpu)
        edge_attr = torch.cat(edge_attrs, dim=0)
        batch = []
        cumbatch = []
        count = 0
        for i, l in enumerate(conv_len):
            num_edge = int(edge_indices[i].size(1))
            batch += [i] * num_edge
            cumbatch += [count] * num_edge
            count += l
        batch = torch.tensor(batch, dtype=torch.long)
        cumbatch = torch.tensor([cumbatch, cumbatch], dtype=torch.long)
        edge_index = torch.cat(edge_indices, dim=1)
        edge_index = edge_index + cumbatch
        if use_gpu:
            edge_index = edge_index.cuda()
            edge_attr = edge_attr.cuda()
        mental_emb = self.tcn(sent_emb, edge_index, edge_attr)
        if self.residual_type == 'cat':
            mental_emb = torch.cat([mental_emb, sent_emb], dim=1)
        elif self.residual_type == 'sum':
            mental_emb = mental_emb + sent_emb
        logits = self.classifier(mental_emb)
        return logits
