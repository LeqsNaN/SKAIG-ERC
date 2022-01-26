import math
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.init import xavier_uniform_
from transformers import BertModel
from transformer import _get_activation_fn, _get_clones
from mask import build_mixed_mask_prior, build_mixed_mask_post, build_mixed_mask_local
from rnn import AbsolutePositionEncoding


class MultiHeadAttention(nn.Module):
    def __init__(self, emb_dim, nhead, dropout=0.1, attn_mask=False):
        super(MultiHeadAttention, self).__init__()
        self.attn_mask = attn_mask
        self.nhead = nhead
        self.head_dim = emb_dim // nhead
        self.q_proj_weight = nn.Parameter(torch.empty(emb_dim, emb_dim), requires_grad=True)
        self.k_proj_weight = nn.Parameter(torch.empty(emb_dim, emb_dim), requires_grad=True)
        self.v_proj_weight = nn.Parameter(torch.empty(emb_dim, emb_dim), requires_grad=True)

        self.o_proj = nn.Linear(emb_dim, emb_dim, bias=False)
        self.dropout = dropout
        self._reset_parameter()

    def _reset_parameter(self):
        xavier_uniform_(self.q_proj_weight)
        xavier_uniform_(self.k_proj_weight)
        xavier_uniform_(self.v_proj_weight)
        xavier_uniform_(self.o_proj.weight)

    def forward(self, q, k, v, mask, require_weight=False):
        src_len = q.size(0)
        tgt_len = k.size(0)

        assert src_len == tgt_len, "length of query does not equal length of key"

        scaling = float(self.head_dim) ** -0.5

        query = F.linear(q, self.q_proj_weight)
        key = F.linear(k, self.k_proj_weight)
        value = F.linear(v, self.v_proj_weight)

        # (n_head, s_len, h_dim)
        query = query.contiguous().view(src_len, self.nhead, self.head_dim).transpose(0, 1)
        key = key.contiguous().view(src_len, self.nhead, self.head_dim).transpose(0, 1)
        value = value.contiguous().view(src_len, self.nhead, self.head_dim).transpose(0, 1)

        # q*k
        attn_weight = torch.bmm(query, key.transpose(1, 2))
        attn_weight = attn_weight * scaling

        if mask is not None:
            attn_weight = torch.masked_fill(attn_weight, mask, -1e30)
        # (n_head, src_len, tgt_len)
        attn_score = F.softmax(attn_weight, dim=-1)
        if self.attn_mask:
            attmask = mask.eq(False).to(torch.float)
            attn_score = attn_score * attmask
        attn_score = F.dropout(attn_score, p=self.dropout, training=self.training)
        attn_output = torch.bmm(attn_score, value)
        # (n_head, src_len, h_dim) -> (src_len, n_head, h_dim) -> (src_len, emb_dim)
        attn_output = attn_output.transpose(0, 1).contiguous().view(src_len, -1)
        output = F.linear(attn_output, self.o_proj.weight)
        if require_weight:
            # attn = attn_score.sum(dim=1) / self.nhead
            # return output, attn
            return output, attn_score
        return output, None


class MultiHeadAttention3D(nn.Module):
    def __init__(self, emb_dim, nhead, dropout=0.1, attn_mask=False):
        super(MultiHeadAttention3D, self).__init__()
        self.attn_mask = attn_mask
        self.nhead = nhead
        self.head_dim = emb_dim // nhead
        self.q_proj_weight = nn.Parameter(torch.empty(emb_dim, emb_dim), requires_grad=True)
        self.k_proj_weight = nn.Parameter(torch.empty(emb_dim, emb_dim), requires_grad=True)
        self.v_proj_weight = nn.Parameter(torch.empty(emb_dim, emb_dim), requires_grad=True)
        self.o_proj = nn.Linear(emb_dim, emb_dim, bias=False)
        self.dropout = dropout
        self._reset_parameter()

    def _reset_parameter(self):
        xavier_uniform_(self.q_proj_weight)
        xavier_uniform_(self.k_proj_weight)
        xavier_uniform_(self.v_proj_weight)
        xavier_uniform_(self.o_proj.weight)

    def forward(self, q, k, v, mask, require_weight=False):
        # input size: (slen, bsz, nh*hdim)
        slen = q.size(0)
        bsz = q.size(1)

        scaling = float(self.head_dim) ** -0.5
        query = F.linear(q, self.q_proj_weight)
        key = F.linear(k, self.k_proj_weight)
        value = F.linear(v, self.v_proj_weight)

        # (slen, bsz, nh*hdim) -> (slen, bsz*nh, hdim) -> (bsz*nh, slen, hdim)
        query = query.contiguous().view(slen, bsz * self.nhead, self.head_dim).transpose(0, 1)
        key = key.contiguous().view(slen, bsz * self.nhead, self.head_dim).transpose(0, 1)
        value = value.contiguous().view(slen, bsz * self.nhead, self.head_dim).transpose(0, 1)
        # (bsz*nh, slen, slen)
        attn_weight = torch.bmm(query, key.transpose(1, 2))
        attn_weight = attn_weight * scaling
        if mask is not None:
            attn_weight = torch.masked_fill(attn_weight, mask, -1e30)
        attn_score = F.softmax(attn_weight, dim=-1)
        if self.attn_mask:
            attmask = mask.eq(False).to(torch.float)
            attn_score = attn_score * attmask
        attn_score = F.dropout(attn_score, p=self.dropout, training=self.training)
        # (bsz*nh, slen, slen) * (bsz*nh, slen, hdim) -> (bsz*nh, slen, hdim)
        attn_output = torch.bmm(attn_score, value)
        # (bsz*nh, slen, hdim) -> (slen, bsz*nh, hdim) -> (slen, bsz, nh*hdim)
        attn_output = attn_output.transpose(0, 1).contiguous().view(slen, bsz, -1)
        output = F.linear(attn_output, self.o_proj.weight)
        if require_weight:
            return output, attn_score
        return output, None


class TransformerLayerAbs3D(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout, activation, attn_mask=False):
        super(TransformerLayerAbs3D, self).__init__()
        self.attention = MultiHeadAttention3D(d_model, nhead, dropout, attn_mask)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def forward(self, src, src_mask, require_weight=False):
        if require_weight:
            src2, weight = self.attention(src, src, src, src_mask, require_weight)
        else:
            src2, _ = self.attention(src, src, src, src_mask, require_weight)
        ss = src + self.dropout1(src2)
        ss = self.norm1(ss)
        if hasattr(self, 'activation'):
            ss2 = self.linear2(self.dropout(self.activation(self.linear1(ss))))
        else:
            ss2 = self.linear2(self.dropout(F.relu(self.linear1(ss))))
        ss = ss + self.dropout2(ss2)
        ss = self.norm2(ss)
        if require_weight:
            return ss, weight
        return ss


class TransformerLayerAbs(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout, activation, attn_mask=False):
        super(TransformerLayerAbs, self).__init__()
        self.attention = MultiHeadAttention(d_model, nhead, dropout, attn_mask)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def forward(self, src, src_mask, require_weight=False):
        if require_weight:
            src2, weight = self.attention(src, src, src, src_mask, require_weight)
        else:
            src2, _ = self.attention(src, src, src, src_mask, require_weight)
        ss = src + self.dropout1(src2)
        ss = self.norm1(ss)
        if hasattr(self, "activation"):
            ss2 = self.linear2(self.dropout(self.activation(self.linear1(ss))))
        else:
            ss2 = self.linear2(self.dropout(F.relu(self.linear1(ss))))
        ss = ss + self.dropout2(ss2)
        ss = self.norm2(ss)
        if require_weight:
            return ss, weight
        return ss


class AbsTransformer(nn.Module):
    def __init__(self, layer, emb_dim, max_len, num_layer, num_class, nhead, snhead=0, onhead=0,
                 bidirectional=False, norm=None):
        super(AbsTransformer, self).__init__()
        self.nhead = nhead
        self.snhead = snhead
        self.onhead = onhead
        self.bidirectional = bidirectional
        self.num_layer = num_layer
        self.norm = norm
        self.pe = AbsolutePositionEncoding(emb_dim, max_len)
        self.layers = _get_clones(layer, num_layer)
        self.classifier = nn.Linear(emb_dim, num_class)
        self._reset_parameter()

    def _reset_parameter(self):
        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)

    # def forward(self, src, utt_mask, spk_mask):
    def forward(self, src, utt_mask, spk_mask, window=100):
        src_len = src.size(0)

        # ##### make masks
        # (1, src_len, tgt_len)
        # uttm, samm, othm = build_mixed_mask_prior(utt_mask.unsqueeze(0), spk_mask.unsqueeze(0), True)
        uttm, samm, othm = build_mixed_mask_local(utt_mask.unsqueeze(0), spk_mask.unsqueeze(0),
                                                  window, self.bidirectional)
        # (nhead-snhead-onhead, src_len, tgt_len)
        src_mask = uttm.expand(self.nhead - self.snhead - self.onhead, src_len, src_len)
        if self.snhead > 0:
            mask_attached = samm.expand(self.snhead, src_len, src_len)
            # (nhead-onhead, src_len, tgt_len)
            src_mask = torch.cat((src_mask, mask_attached), dim=0)
        if self.onhead > 0:
            mask_attached = othm.expand(self.onhead, src_len, src_len)
            # (nhead, src_len, tgt_len)
            src_mask = torch.cat((src_mask, mask_attached), dim=0)

        # ##### feed forward
        src = self.pe(src)
        output = src
        for i in range(self.num_layer):
            output = self.layers[i](output, src_mask)
        log_prob = F.log_softmax(self.classifier(output), dim=-1)
        return log_prob


class BertTransformer(nn.Module):
    def __init__(self,
                 num_class,
                 num_layer=0,
                 utt_encoder='transformer',
                 max_len=0,
                 emb_dim=0,
                 nhead=0,
                 snhead=0,
                 onhead=0,
                 ff_dim=0,
                 dropout=0.,
                 activation='relu',
                 bidirectional=False,
                 attn_mask=False):
        super(BertTransformer, self).__init__()
        self.utt_encoder = utt_encoder
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        if utt_encoder == 'linear':
            self.classifier = nn.Linear(768, num_class)
        elif utt_encoder == 'transformer':
            self.proj = nn.Linear(768, emb_dim)
            trans = TransformerLayerAbs(emb_dim, nhead, ff_dim, dropout, activation, attn_mask)
            self.transformer = AbsTransformer(trans, emb_dim, max_len, num_layer, num_class,
                                              nhead, snhead, onhead, bidirectional)
        else:
            assert 1 < 0, 'No such utterance-level encoder'

    # def forward(self, conv, attn_mask, utt_mask, spk_mask):
    def forward(self, conv, attn_mask, utt_mask, spk_mask, window=10):
        # (conv_len, sent_len, 768)
        conv_emb = self.bert(conv, attn_mask)[0]
        conv_pooler = torch.max(conv_emb, dim=1)[0]
        if self.utt_encoder == 'transformer':
            conv_pooler = self.proj(conv_pooler)
            log_prob = self.transformer(conv_pooler, utt_mask, spk_mask)
        else:
            log_prob = F.log_softmax(self.classifier(conv_pooler), dim=-1)
        return log_prob
