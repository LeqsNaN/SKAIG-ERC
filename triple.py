import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.init import xavier_uniform_
from transformers import BertModel

from bert_utils import AbsolutePositionEncoding, _get_clones, TransformerLayerAbs
from mask import build_mixed_mask_local


class FusionAttention(nn.Module):
    def __init__(self, input_dim):
        super(FusionAttention, self).__init__()
        self.wf = nn.Parameter(torch.empty((1, input_dim, 1)), requires_grad=True)

    def forward(self, feat):
        seq_len = feat.size(1)
        sent_dim = feat.size(2)
        # (3, seq_len, sent_dim) -> (seq_len, 3, sent_dim)
        feat = feat.transpose(0, 1)
        # (seq_len, 3, 1)
        alpha = torch.bmm(feat, self.wf.expand(seq_len, sent_dim, 1))
        alpha = F.softmax(alpha, dim=1)
        # (seq_len, 1, 3)*(seq_len, 3, sent_dim) -> (seq_len, 1, sent_dim)
        out = torch.bmm(alpha.transpose(1, 2), feat)

        return out.squeeze(1)


class TripleTransformer(nn.Module):
    def __init__(self,
                 layer,
                 nhead,
                 num_layer,
                 emb_dim,
                 max_len,
                 num_class,
                 bidirectional,
                 num_block,
                 norm=None):
        super(TripleTransformer, self).__init__()
        self.nhead = nhead
        self.bidirectional = bidirectional
        self.num_layer = num_layer
        self.norm = norm
        self.num_block = num_block
        self.pe = AbsolutePositionEncoding(emb_dim, max_len)
        if self.num_block == 1:
            self.layers1 = _get_clones(layer, num_layer)
        elif self.num_block == 2:
            self.layers1 = _get_clones(layer, num_layer)
            self.layers2 = _get_clones(layer, num_layer)
            self.fusion = FusionAttention(emb_dim)
        elif self.num_block == 3:
            self.layers1 = _get_clones(layer, num_layer)
            self.layers2 = _get_clones(layer, num_layer)
            self.layers3 = _get_clones(layer, num_layer)
            self.fusion = FusionAttention(emb_dim)
        else:
            assert 1 <= num_block <= 3, 'ooc'
        # self.layers = _get_clones(layer, num_layer)
        self.classifier = nn.Linear(emb_dim, num_class)
        self._reset_parameter()

    def _reset_parameter(self):
        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)

    def forward(self, src, utt_mask, spk_mask, window=100):
        src_len = src.size(0)

        # ##### make masks
        # (1, src_len, tgt_len)
        # uttm, samm, othm = build_mixed_mask_prior(utt_mask.unsqueeze(0), spk_mask.unsqueeze(0), True)
        uttm, samm, othm = build_mixed_mask_local(utt_mask.unsqueeze(0), spk_mask.unsqueeze(0),
                                                  window, self.bidirectional)
        uttm = uttm.expand(self.nhead, src_len, src_len)
        samm = samm.expand(self.nhead, src_len, src_len)
        othm = othm.expand(self.nhead, src_len, src_len)

        src = self.pe(src)
        if self.num_block == 1:
            output = src
            for i in range(self.num_layer):
                output = self.layers1[i](output, uttm)
        elif self.num_block == 2:
            output1 = src
            output2 = src
            for i in range(self.num_layer):
                output1 = self.layers1[i](output1, samm)
                output2 = self.layers2[i](output2, othm)
            # (2, seq_len, sent_dim)
            output = torch.stack([output1, output2], dim=0)
            output = self.fusion(output)
        elif self.num_block == 3:
            output1 = src
            output2 = src
            output3 = src
            for i in range(self.num_layer):
                output1 = self.layers1[i](output1, uttm)
                output2 = self.layers2[i](output2, samm)
                output3 = self.layers3[i](output3, othm)
            # (3, seq_len, sent_dim)
            output = torch.stack([output1, output2, output3], dim=0)
            output = self.fusion(output)
        else:
            output = None
            assert 1 <= self.num_block <= 2, 'ooc'

        log_prob = F.log_softmax(self.classifier(output), dim=-1)
        return log_prob


class GoodLuck(nn.Module):
    def __init__(self,
                 num_class,
                 num_layer=0,
                 max_len=0,
                 emb_dim=0,
                 nhead=0,
                 num_block=3,
                 ff_dim=0,
                 dropout=0.,
                 activation='relu',
                 bidirectional=False,
                 attn_mask=False):
        super(GoodLuck, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.proj = nn.Linear(768, emb_dim)
        trans = TransformerLayerAbs(emb_dim, nhead, ff_dim, dropout, activation, attn_mask)
        self.transformer = TripleTransformer(trans, nhead, num_layer, emb_dim, max_len, num_class, bidirectional, num_block)

    # def forward(self, conv, attn_mask, utt_mask, spk_mask):
    def forward(self, conv, attn_mask, utt_mask, spk_mask, window=100):
        # (conv_len, sent_len, 768)
        conv_emb = self.bert(conv, attn_mask)[0]
        conv_pooler = torch.max(conv_emb, dim=1)[0]
        conv_pooler = self.proj(conv_pooler)
        log_prob = self.transformer(conv_pooler, utt_mask, spk_mask, window)
        return log_prob
