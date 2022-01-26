import torch
import torch.nn as nn
import torch.nn.functional as F
from gnn import TRMCN


class MentalModel(nn.Module):
    def __init__(self, in_channels, ff_dim, out_channels, heads, edge_dim, bias, at_dropout,
                 dropout, num_layers, edge_mapping, beta, root_weight, num_class):
        super(MentalModel, self).__init__()

        self.sent_map = nn.Linear(1024, in_channels)

        self.trm_conv = TRMCN(in_channels=in_channels,
                              ff_dim=ff_dim,
                              out_channels=out_channels//heads,
                              heads=heads,
                              edge_dim=edge_dim,
                              bias=bias,
                              num_layers=num_layers,
                              edge_mapping=edge_mapping,
                              beta=beta,
                              root_weight=root_weight,
                              dropout=at_dropout)
        self.classifier = nn.Linear(out_channels, num_class)
        self.dropout = nn.Dropout(dropout)

    def forward(self, r, conv_len, edge_indices,
                edge_attrs, use_gpu=True, residual=False):
        # (BA, D)
        edge_attr = torch.cat(edge_attrs, dim=0)

        if use_gpu:
            r = [rt.cuda() for rt in r]
        r = torch.cat(r, dim=0)

        graph_input = self.sent_map(r)

        # mini-batching the graphs
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
        # (2, BA)
        edge_index = torch.cat(edge_indices, dim=1)
        edge_index = edge_index + cumbatch
        if use_gpu:
            edge_index = edge_index.cuda()
            edge_attr = edge_attr.cuda()
        # (BC, D)
        mental_emb = self.trm_conv(graph_input, edge_index, edge_attr)
        mental_emb = self.dropout(mental_emb)
        if residual:
            mental_emb = mental_emb + graph_input
        logits = self.classifier(mental_emb)
        return logits
