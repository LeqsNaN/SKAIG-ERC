import math
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np

from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence, pad_sequence
from torch.nn.init import xavier_uniform_
from torch.utils.data import Dataset
from transformer import build_mask, _get_activation_fn, _get_clones
from mask import build_mixed_mask_prior, build_mixed_mask_post


class GRUEncoder(nn.Module):
    def __init__(self, emb_dim, rnn_hidden_dim, sent_dim, dropout=0.3):
        super(GRUEncoder, self).__init__()
        self.gru = nn.GRU(emb_dim, rnn_hidden_dim, num_layers=1, bidirectional=True)
        self.proj = nn.Linear(2*rnn_hidden_dim, sent_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, sent, sent_len):
        """
        :param sent: torch tensor, N x L x D_in
        :param sent_len: torch tensor, N
        :return:
        """
        # (N, L, D_w) -> (L, N, D_w)
        sent_embs = sent.transpose(0, 1)

        # padding
        # (L, N, D_w) -> (L, N, 2*D_h)
        sent_packed = pack_padded_sequence(sent_embs, sent_len, enforce_sorted=False)
        sent_output = self.gru(sent_packed)[0]
        sent_output = pad_packed_sequence(sent_output, total_length=sent.size(1))[0]

        # (L, N, 2*D_h) -> (N, L, 2*D_h)
        sent_output = sent_output.transpose(0, 1)

        # max pooling
        # (N, L, 2*D_h) -> (N, 2*D_h, L) ->
        # (N, 2*D_h, 1) -> (N, 1, 2*D_h)
        maxpout = F.max_pool1d(sent_output.transpose(2, 1), sent_output.size(1))
        maxpout = maxpout.transpose(2, 1)

        # (N, 1, 2*D_h) -> (N, 1, D_s) -> (N, D_s)
        sent_rep = self.dropout(F.relu(self.proj(maxpout)))
        sent_rep = sent_rep.squeeze(1)

        return sent_rep


class MultiHeadAttention(nn.Module):
    def __init__(self, emb_dim, nhead, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
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
            attn_weight = torch.masked_fill(attn_weight, mask, 1e-30)
        # (n_head, src_len, tgt_len)
        attn_score = F.softmax(attn_weight, dim=-1)
        attn_score = F.dropout(attn_score, p=self.dropout, training=self.training)
        attn_output = torch.bmm(attn_score, value)
        # (n_head, src_len, h_dim) -> (src_len, n_head, h_dim) -> (src_len, emb_dim)
        attn_output = attn_output.transpose(0, 1).contiguous().view(src_len, -1)
        output = F.linear(attn_output, self.o_proj.weight)
        if require_weight:
            attn = attn_score.sum(dim=1) / self.nhead
            return output, attn
        return output, None


class AttentionType1(MultiHeadAttention):
    def __init__(self, emb_dim, nhead, dropout, qk=False, share=False):
        super(AttentionType1, self).__init__(emb_dim, nhead, dropout)
        self.nhead = nhead
        self.head_dim = emb_dim // nhead
        self.qk = qk
        self.share = share
        if share:
            self.k_encoding = nn.Embedding(2, self.head_dim)
            if qk:
                self.q_encoding = nn.Parameter(torch.empty(1, 1, self.head_dim), requires_grad=True)
            else:
                self.q_encoding = None
        else:
            self.k_encoding = nn.Embedding(2, emb_dim)
            if qk:
                self.q_encoding = nn.Parameter(torch.empty(1, emb_dim), requires_grad=True)
            else:
                self.q_encoding = None
        self._reset_parameter_type1()

    def _reset_parameter_type1(self):
        xavier_uniform_(self.q_proj_weight)
        xavier_uniform_(self.k_proj_weight)
        xavier_uniform_(self.v_proj_weight)
        xavier_uniform_(self.o_proj.weight)
        xavier_uniform_(self.k_encoding.weight)
        if self.qk:
            xavier_uniform_(self.q_encoding)

    def forward(self, q, k, v, mask, require_weight=False, utt_idx=None, spk_idx=None):
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

        if self.qk:
            if self.share:
                new_q = query + self.q_encoding
            else:
                q_emb = self.q_encoding
                # (nhead, 1, head_dim)
                q_emb = q_emb.contiguous().view(self.nhead, self.head_dim).unsqueeze(1)
                new_q = query + q_emb
        else:
            if self.share:
                idx = torch.zeros((self.nhead, src_len), dtype=torch.long, device=query.device)
                q_emb = self.k_encoding(idx)
            else:
                idx = torch.zeros(src_len, dtype=torch.long, device=query.device)
                q_emb = self.k_encoding(idx)
                q_emb = q_emb.contiguous().view(src_len, self.nhead, self.head_dim).transpose(0, 1)
            new_q = query + q_emb

        attn_score1 = torch.bmm(new_q, key.transpose(1, 2))

        # utt_idx, spk_idx: [src_len, tgt_len]
        # (src_len, tgt_len, dim)
        k_emb = self.k_encoding(spk_idx)
        utt_idx = utt_idx.unsqueeze(2)
        k_emb = k_emb * utt_idx
        if self.share:
            k_emb = k_emb.unsqueeze(0).expand(self.nhead, src_len, tgt_len, self.head_dim)
        else:
            k_emb = k_emb.contiguous().view(src_len*tgt_len, self.nhead, self.head_dim).transpose(0, 1)
            k_emb = k_emb.contiguous().view(self.nhead, src_len, tgt_len, self.head_dim)
        attn_score2 = torch.einsum('hsd,hstd->hst', (new_q, k_emb))
        # q*k
        attn_weight = attn_score1 + attn_score2
        # q*s_k
        attn_weight = attn_weight * scaling

        if mask is not None:
            attn_weight = torch.masked_fill(attn_weight, mask, 1e-30)
        # (n_head, src_len, tgt_len)
        attn_score = F.softmax(attn_weight, dim=-1)
        attn_score = F.dropout(attn_score, p=self.dropout, training=self.training)
        attn_output = torch.bmm(attn_score, value)
        # (n_head, src_len, h_dim) -> (src_len, n_head, h_dim) -> (src_len, emb_dim)
        attn_output = attn_output.transpose(0, 1).contiguous().view(src_len, -1)
        output = F.linear(attn_output, self.o_proj.weight)
        if require_weight:
            attn = attn_score.sum(dim=1) / self.nhead
            return output, attn
        return output, None


class AttentionType2(MultiHeadAttention):
    def __init__(self, emb_dim, nhead, dropout, share=False):
        super(AttentionType2, self).__init__(emb_dim, nhead, dropout)
        self.nhead = nhead
        self.head_dim = emb_dim // nhead
        self.share = share
        self._reset_parameter()

    def forward(self, q, k, v, mask, require_weight=False, s_q=None, s_k=None):
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
        if self.share:
            s_q = s_q.unsqueeze(0)
            s_k = s_k.unsqueeze(0).expand(self.nhead, src_len, tgt_len, self.head_dim)
        else:
            s_q = s_q.contiguous().view(self.nhead, self.head_dim).unsqueeze(1)
            s_k = s_k.contiguous().view(src_len*tgt_len, self.nhead, self.head_dim).transpose(0, 1)
            s_k = s_k.contiguous().view(self.nhead, src_len, tgt_len, self.head_dim)
        new_q = query + s_q
        attn_weight1 = torch.bmm(new_q, key.transpose(1, 2))
        attn_weight2 = torch.einsum('hsd,hstd->hst', (new_q, s_k))
        attn_weight = attn_weight1 + attn_weight2
        attn_weight = attn_weight * scaling

        if mask is not None:
            attn_weight = torch.masked_fill(attn_weight, mask, 1e-30)
        # (n_head, src_len, tgt_len)
        attn_score = F.softmax(attn_weight, dim=-1)
        attn_score = F.dropout(attn_score, p=self.dropout, training=self.training)
        attn_output = torch.bmm(attn_score, value)
        # (n_head, src_len, h_dim) -> (src_len, n_head, h_dim) -> (src_len, emb_dim)
        attn_output = attn_output.transpose(0, 1).contiguous().view(src_len, -1)
        output = F.linear(attn_output, self.o_proj.weight)
        if require_weight:
            attn = attn_score.sum(dim=1) / self.nhead
            return output, attn
        return output, None


class TransformerLayerAbs(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout, activation, attn_type=1, qk=False, share=False):
        super(TransformerLayerAbs, self).__init__()
        self.attn_type = attn_type
        if attn_type == 1:
            self.attention = AttentionType1(d_model, nhead, dropout, qk, share)
        else:
            self.attention = AttentionType2(d_model, nhead, dropout, share)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def forward(self, src, src_mask, utt_idx=None, spk_idx=None, s_q=None, s_k=None):
        if self.attn_type == 1:
            src2, _ = self.attention(src, src, src, src_mask, utt_idx=utt_idx, spk_idx=spk_idx)
        else:
            src2, _ = self.attention(src, src, src, src_mask, s_q=s_q, s_k=s_k)
        ss = src + self.dropout1(src2)
        ss = self.norm1(ss)
        if hasattr(self, "activation"):
            ss2 = self.linear2(self.dropout(self.activation(self.linear1(ss))))
        else:
            ss2 = self.linear2(self.dropout(F.relu(self.linear1(ss))))
        ss = ss + self.dropout2(ss2)
        ss = self.norm2(ss)
        return ss


class AbsTransformer(nn.Module):
    def __init__(self, layer, emb_dim, max_len, num_layer, num_class, nhead, snhead=0,
                 attn_type=1, mlt=False, qk=False, share=False, norm=None):
        super(AbsTransformer, self).__init__()
        self.nhead = nhead
        self.snhead = snhead
        self.qk = qk
        self.share = share
        self.attn_type = attn_type
        self.layers = _get_clones(layer, num_layer)
        self.num_layer = num_layer
        self.norm = norm
        self.pe = AbsolutePositionEncoding(emb_dim, max_len)
        if attn_type == 2:
            if share:
                if qk:
                    self.q_encoding = nn.Parameter(torch.empty(1, emb_dim // nhead), requires_grad=True)
                else:
                    self.q_encoding = None
                self.k_encoding = nn.Embedding(2, emb_dim // nhead)
            else:
                if qk:
                    self.q_encoding = nn.Parameter(torch.empty(1, emb_dim), requires_grad=True)
                else:
                    self.q_encoding = None
                self.k_encoding = nn.Embedding(2, emb_dim)
        self.classifier = nn.Linear(emb_dim, num_class)
        if mlt:
            self.sclassifier = nn.Linear(emb_dim, 1)
        self.mlt = mlt
        self._reset_parameter()

    def _reset_parameter(self):
        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)

    def forward(self, src, utt_mask, spk_mask):
        src_len = src.size(0)

        # ##### make masks
        # (1, src_len, tgt_len)
        uttm, spkm = build_mask(utt_mask.unsqueeze(0), spk_mask.unsqueeze(0), False)
        src_mask = uttm.expand(self.nhead - self.snhead, src_len, src_len)
        if self.snhead > 0:
            mask_attached = spkm.expand(self.snhead, src_len, src_len)
            # (nhead, src_len, tgt_len)
            src_mask = torch.cat((src_mask, mask_attached), dim=0)

        # ##### feed forward
        src = self.pe(src)
        # (src_len, tgt_len)
        spk_idx = spkm.squeeze(0).to(torch.long)
        if self.attn_type == 1:
            utt_idx = uttm.squeeze(0).eq(False).to(torch.float)
            output = src
            for i in range(self.num_layer):
                output = self.layers[i](output, src_mask, utt_idx=utt_idx, spk_idx=spk_idx)
        else:
            if self.qk:
                # (1, dim)
                s_q = self.q_encoding
            else:
                idx = torch.zeros(1, device=src.device, dtype=torch.long)
                s_q = self.k_encoding(idx)
            # (src_len, tgt_len, dim)
            s_k = self.k_encoding(spk_idx)
            output = src
            for i in range(self.num_layer):
                output = self.layers[i](output, src_mask, s_q=s_q, s_k=s_k)
        log_prob = F.log_softmax(self.classifier(output), dim=-1)
        if self.mlt:
            s_log_prob = F.sigmoid(self.sclassifier(output))
            return log_prob, s_log_prob
        return log_prob


class AbsModel(nn.Module):
    def __init__(self, embeddings, word_dim, rnn_hidden_dim, sent_dim, rnn_dropout, nhead, ff_dim, tf_dropout,
                 activation, num_class, num_layer, max_len, snhead=0, attn_type=1, share=True, qk=False, mlt=False):
        super(AbsModel, self).__init__()
        self.mlt = mlt
        self.embedding = nn.Embedding(embeddings.size(0), embeddings.size(1), padding_idx=0, _weight=embeddings)
        self.embedding.weight.requires_grad = False
        self.gru = GRUEncoder(word_dim, rnn_hidden_dim, sent_dim, rnn_dropout)
        trans = TransformerLayerAbs(sent_dim, nhead, ff_dim, tf_dropout, activation,
                                    attn_type=attn_type, qk=qk, share=share)
        self.transformer = AbsTransformer(trans, sent_dim, max_len, num_layer, num_class, nhead,
                                          snhead, attn_type, mlt, qk=qk, share=share)

    def forward(self, w, w_len, utt_mask, spk_mask):
        word_emb = self.embedding(w)
        sent_rep = self.gru(word_emb, w_len)
        if self.mlt:
            log_prob, s_log_prob = self.transformer(sent_rep, utt_mask, spk_mask)
            return log_prob, s_log_prob
        else:
            log_prob = self.transformer(sent_rep, utt_mask, spk_mask)
            return log_prob


class MultiHeadRelativeAttention(MultiHeadAttention):
    def __init__(self, emb_dim, nhead, dropout=0.1):
        super(MultiHeadRelativeAttention, self).__init__(emb_dim, nhead, dropout)
        self.nhead = nhead
        self.head_dim = emb_dim // nhead
        self.q_proj_weight = nn.Parameter(torch.empty(emb_dim, emb_dim), requires_grad=True)
        self.k_proj_weight = nn.Parameter(torch.empty(emb_dim, emb_dim), requires_grad=True)
        self.v_proj_weight = nn.Parameter(torch.empty(emb_dim, emb_dim), requires_grad=True)
        self.r_proj_weight = nn.Parameter(torch.empty(emb_dim, emb_dim), requires_grad=True)

        self.o_proj = nn.Linear(emb_dim, emb_dim, bias=False)
        self.droput = dropout
        self._reset_parameter()

    def _reset_parameter(self):
        xavier_uniform_(self.q_proj_weight)
        xavier_uniform_(self.k_proj_weight)
        xavier_uniform_(self.v_proj_weight)
        xavier_uniform_(self.r_proj_weight)
        xavier_uniform_(self.o_proj.weight)

    def _rel_shift(self, r, zero_tril=False):
        # r: (N, tgt_len, rel_len)
        bsz, tlen, rlen = r.size()
        zero_pad = torch.ones((bsz, tlen, 1), device=r.device, dtype=r.dtype)
        # (N, tgt_len, rel_len+1)
        r_padded = torch.cat((zero_pad, r), dim=2)
        # (N, rel_len+1, tgt_len)
        r_padded = r_padded.contiguous().view(bsz, rlen+1, tlen)
        # (N, tgt_len, rel_len)
        r = r_padded[:, 1:, :].view_as(r)
        if zero_tril:
            r = torch.tril(r, 0)
        return r

    def forward(self, q, k, v, mask, require_weight=False, p=None, s_q=None, s_k=None):
        r"""
        Args:
            q: (src_len, emb_dim)
            k: (tgt_len, emb_dim)
            v: (tgt_len, emb_dim)
            p: (rel_len, emb_dim)
            s_k: (1, emb_dim)
            s_q: (tgt_len, tgt_len, emb_dim)
            mask: (n_head, src_len, tgt_len)
            require_weight: boolean
        """
        src_len = q.size(0)
        tgt_len = k.size(0)
        rel_len = p.size(0)

        assert src_len == tgt_len, "length of query does not equal length of key"
        assert src_len == rel_len, "length of query does not equal length of pe"

        scaling = float(self.head_dim) ** -0.5

        query = F.linear(q, self.q_proj_weight)
        key = F.linear(k, self.k_proj_weight)
        value = F.linear(v, self.v_proj_weight)
        rel = F.linear(p, self.r_proj_weight)

        # (n_head, s_len, h_dim)
        query = query.contiguous().view(src_len, self.nhead, self.head_dim).transpose(0, 1)
        key = key.contiguous().view(src_len, self.nhead, self.head_dim).transpose(0, 1)
        value = value.contiguous().view(src_len, self.nhead, self.head_dim).transpose(0, 1)
        rel = rel.contiguous().view(src_len, self.nhead, self.head_dim).transpose(0, 1)

        # (n_head, 1, h_dim)
        s_q = s_q.contiguous().view(1, self.nhead, self.head_dim).transpose(0, 1)
        # (n_head, s_len, s_len, h_dim)
        s_k = s_k.contiguous().view(-1, self.nhead, self.head_dim).transpose(0, 1)
        s_k = s_k.view(self.nhead, src_len, src_len, self.head_dim)

        new_q = query + s_q
        # q*k
        a1 = torch.bmm(new_q, key.transpose(1, 2))
        # q*s_k
        a2 = torch.einsum('nid,nijd->nij', (new_q, s_k))
        # q*r
        a3 = torch.bmm(new_q, rel.transpose(1, 2))
        a3 = self._rel_shift(a3, True)

        attn_weight = a1 + a2 + a3
        attn_weight = attn_weight * scaling

        if mask is not None:
            attn_weight = torch.masked_fill(attn_weight, mask, 1e-30)
        # (n_head, src_len, tgt_len)
        attn_score = F.softmax(attn_weight, dim=-1)
        attn_score = F.dropout(attn_score, p=self.droput, training=self.training)
        attn_output = torch.bmm(attn_score, value)
        # (n_head, src_len, h_dim) -> (src_len, n_head, h_dim)
        attn_output = attn_output.transpose(0, 1).contiguous().view(src_len, -1)
        output = F.linear(attn_output, self.o_proj.weight)
        if require_weight:
            attn = attn_score.sum(dim=1) / self.nhead
            return output, attn
        return output, None


class TransformerLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout, activation, relative=True):
        super(TransformerLayer, self).__init__()
        self.relative = relative
        if relative:
            self.attention = MultiHeadRelativeAttention(d_model, nhead, dropout)
        else:
            self.attention = MultiHeadAttention(d_model, nhead, dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def forward(self, src, rel_pe=None, s_q=None, s_k=None, src_mask=None):
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            rel_pe: the relatively positional embeddings (optional).
            src_mask: the mask for the src sequence (optional).
            s_q: speaker encoding of self side reused by par layers(optional).
            s_k: speaker encoding of other side reused by par layers(optional).
        Shape:
            ...
        """
        if self.relative:
            src2, _ = self.attention(src, src, src, mask=src_mask, require_weight=False, p=rel_pe, s_q=s_q, s_k=s_k)
        else:
            src2, _ = self.attention(src, src, src, mask=src_mask, require_weight=False)
        ss = src + self.dropout1(src2)
        ss = self.norm1(ss)
        if hasattr(self, "activation"):
            ss2 = self.linear2(self.dropout(self.activation(self.linear1(ss))))
        else:
            ss2 = self.linear2(self.dropout(F.relu(self.linear1(ss))))
        ss = ss + self.dropout2(ss2)
        ss = self.norm2(ss)
        return ss


class Transformer(nn.Module):
    def __init__(self, layer, emb_dim, max_len, num_layer, num_class, nhead, snhead=0, relative=True, kv=False, mlt=False, norm=None):
        super(Transformer, self).__init__()
        self.nhead = nhead
        self.snhead = snhead
        self.relative = relative
        self.kv = kv
        self.layers = _get_clones(layer, num_layer)
        self.num_layer = num_layer
        self.norm = norm
        if relative:
            self.pe = RelativePositionEncoding(emb_dim, max_len)
            if kv:
                self.q_embedding = nn.Parameter(torch.empty(1, emb_dim), requires_grad=True)
            else:
                self.q_embedding = None
            self.spk_embedding = nn.Embedding(2, emb_dim)
        else:
            self.pe = AbsolutePositionEncoding(emb_dim, max_len)
            self.spk_embedding = None
            self.q_embedding = None
        self.classifier = nn.Linear(emb_dim, num_class)
        if mlt:
            self.sclassifier = nn.Linear(emb_dim, 1)
        self.mlt = mlt
        self._reset_parameter()

    def _reset_parameter(self):
        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)

    def forward(self, src, utt_mask, spk_mask):
        r"""
        Args:
            src: (src_len, emb_dim)
            utt_mask: (src_len)
            spk_mask: (src_len)
        """
        src_len = src.size(0)
        # (1, src_len, tgt_len)
        uttm, spkm = build_mask(utt_mask.unsqueeze(0), spk_mask.unsqueeze(0), False)
        src_mask = uttm.expand(self.nhead-self.snhead, src_len, src_len)
        if self.snhead > 0:
            mask_attached = spkm.expand(self.snhead, src_len, src_len)
            # (nhead, src_len, tgt_len)
            src_mask = torch.cat((src_mask, mask_attached), dim=0)
        if self.relative:
            # (src_len, emb_dim)
            rel_pe = self.pe(src)
            if self.kv:
                s_q = self.q_embedding
            else:
                sq_idx = torch.zeros((1, 1), device=src.device(), dtype=torch.long)
                # (1, emb_dim)
                s_q = self.spk_embedding(sq_idx)
            sk_idx = torch.zeros(spkm.size(), device=src.device(), dtype=torch.long)
            sk_idx = torch.masked_fill(sk_idx, spkm, 1)
            # (1, src_len, tgt_len, emb_dim)
            s_k = self.spk_embedding(sk_idx)
            # (src_len, tgt_len, emb_dim)
            s_k = s_k.squeeze(0)
            # (src_len, emb_dim)
            output = src
            for i in range(self.num_layer):
                output = self.layers[i](output, rel_pe, s_q, s_k, src_mask)
        else:
            src = self.pe(src)
            output = src
            for i in range(self.num_layer):
                output = self.layers[i](output, None, None, None, src_mask)
        log_prob = F.log_softmax(self.classifier(output), dim=-1)
        if self.mlt:
            s_log_prob = F.sigmoid(self.sclassifier(output))
            return log_prob, s_log_prob
        return log_prob


class Model(nn.Module):
    def __init__(self, embeddings, word_dim, rnn_hidden_dim, sent_dim, rnn_dropout, nhead, ff_dim, tf_dropout,
                 activation, num_class, num_layer, max_len, snhead=0, relative=True, kv=False, mlt=False):
        super(Model, self).__init__()
        self.mlt = mlt
        self.embedding = nn.Embedding(embeddings.size(0), embeddings.size(1), padding_idx=0, _weight=embeddings)
        self.embedding.weight.requires_grad = False
        self.gru = GRUEncoder(word_dim, rnn_hidden_dim, sent_dim, rnn_dropout)
        trans = TransformerLayer(sent_dim, nhead, ff_dim, tf_dropout, activation, relative=relative)
        self.transformer = Transformer(trans, sent_dim, max_len, num_layer, num_class, nhead,
                                       snhead, relative=relative, kv=kv, mlt=mlt, norm=None)

    def forward(self, w, w_len, utt_mask, spk_mask):
        word_emb = self.embedding(w)
        sent_rep = self.gru(word_emb, w_len)
        if self.mlt:
            log_prob, s_log_prob = self.transformer(sent_rep, utt_mask, spk_mask)
            return log_prob, s_log_prob
        else:
            log_prob = self.transformer(sent_rep, utt_mask, spk_mask)
            return log_prob


class RelativePositionEncoding(nn.Module):
    # Relative Position (abs(i-j)) pe[:, max_len-1] is the embedding of distance 1
    def __init__(self, input_dim, max_len=200):
        super(RelativePositionEncoding, self).__init__()
        self.max_len = max_len
        pe = torch.zeros(max_len, input_dim)
        position = torch.arange(max_len-1, -1., -1.).unsqueeze(1)
        div_term = torch.exp(torch.arange(0., input_dim, 2) * -(math.log(10000.) / input_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # (Max_len, D)
        self.register_buffer('pe', pe)

    def forward(self, x):
        seq_len = x.size(0)
        # x(L, D)   pe_clip(L, D)
        pe_clip = self.pe[self.max_len-seq_len:]
        pemb = pe_clip.expand_as(x)
        return pemb


class AbsolutePositionEncoding(nn.Module):
    def __init__(self, input_dim, max_len=200):
        super(AbsolutePositionEncoding, self).__init__()
        self.max_len = max_len
        pe = torch.zeros(max_len, input_dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0., input_dim, 2) * -(math.log(10000.) / input_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        seq_len = x.size(0)
        pe_clip = self.pe[:seq_len]
        pemb = x + pe_clip
        return pemb


class EsTransformerLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout, activation):
        super(EsTransformerLayer, self).__init__()
        self.attention = MultiHeadAttention(d_model, nhead, dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def forward(self, src, src_mask=None):
        src2, _ = self.attention(src, src, src, mask=src_mask, require_weight=False)
        ss = src + self.dropout1(src2)
        ss = self.norm1(ss)
        if hasattr(self, "activation"):
            ss2 = self.linear2(self.dropout(self.activation(self.linear1(ss))))
        else:
            ss2 = self.linear2(self.dropout(F.relu(self.linear1(ss))))
        ss = ss + self.dropout2(ss2)
        ss = self.norm2(ss)
        return ss


class EsTransformer(nn.Module):
    def __init__(self, layer, emb_dim, max_len, num_layer, nhead, snhead=0, norm=None):
        super(EsTransformer, self).__init__()
        self.nhead = nhead
        self.snhead = snhead
        self.layers = _get_clones(layer, num_layer)
        self.num_layer = num_layer
        self.norm = norm
        self.pe = AbsolutePositionEncoding(emb_dim, max_len)
        self.spk_embedding = None
        self.q_embedding = None
        self.classifier = nn.Linear(emb_dim, 1)
        self._reset_parameter()

    def _reset_parameter(self):
        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)

    def forward(self, src, utt_mask, spk_mask):
        src_len = src.size(0)
        # (1, src_len, tgt_len)
        uttm, spkm = build_mask(utt_mask.unsqueeze(0), spk_mask.unsqueeze(0), False)
        src_mask = uttm.expand(self.nhead-self.snhead, src_len, src_len)
        if self.snhead > 0:
            mask_attached = spkm.expand(self.snhead, src_len, src_len)
            # (nhead, src_len, tgt_len)
            src_mask = torch.cat((src_mask, mask_attached), dim=0)
        src = self.pe(src)
        output = src
        for i in range(self.num_layer):
            output = self.layers[i](output, src_mask)
        output_es = output[1:] + output[0:-1]
        # (src_len-1, 1)
        log_prob = F.sigmoid(self.classifier(output_es))
        return log_prob


class EsModel(nn.Module):
    def __init__(self, embeddings, word_dim, rnn_hidden_dim, sent_dim, rnn_dropout, nhead, ff_dim, tf_dropout,
                 activation, num_layer, max_len, snhead=0):
        super(EsModel, self).__init__()
        self.embedding = nn.Embedding(embeddings.size(0), embeddings.size(1), padding_idx=0, _weight=embeddings)
        self.embedding.weight.requires_grad = False
        self.gru = GRUEncoder(word_dim, rnn_hidden_dim, sent_dim, rnn_dropout)
        trans = EsTransformerLayer(sent_dim, nhead, ff_dim, tf_dropout, activation)
        self.transformer = EsTransformer(trans, sent_dim, max_len, num_layer, nhead, snhead)

    def forward(self, w, w_len, utt_mask, spk_mask):
        word_emb = self.embedding(w)
        sent_rep = self.gru(word_emb, w_len)
        log_prob = self.transformer(sent_rep, utt_mask, spk_mask)
        return log_prob


class IEMOCAPDataset(Dataset):
    def __init__(self, path, train=True):
        train_data, test_data, _ = pickle.load(open(path, 'rb'), encoding='latin1')
        if train:
            self.sentence = train_data[0]
            # (conv_num, conv_len)
            self.word_mask = []
            for conv in train_data[1]:
                w = []
                for sent in conv:
                    w.append(np.sum(sent))
                self.word_mask.append(w)
            self.dlabels = train_data[2]
            self.slabels = train_data[3]
            self.speaker = train_data[4]
        else:
            self.sentence = test_data[0]
            self.word_mask = []
            for conv in train_data[1]:
                w = []
                for sent in conv:
                    w.append(np.sum(sent))
                self.word_mask.append(w)
            self.dlabels = test_data[2]
            self.slabels = test_data[3]
            self.speaker = test_data[4]

        self.len = len(self.dlabels)

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        txt = torch.tensor(self.sentence[index]).long()
        wmask = torch.tensor(self.word_mask[index])
        umask = torch.tensor([1] * len(self.dlabels[index])).float()
        smask = torch.tensor([0 if x == 'M' else 1 for x in self.speaker[index]]).float()
        dlabel = torch.tensor(self.dlabels[index]).long()
        slabel = torch.tensor(self.slabels[index]).long()
        return txt, dlabel, slabel, wmask, umask, smask

    def collate_fn(self, data):
        dat = pd.DataFrame(data)
        return [pad_sequence(dat[i], True) if i < 5 else pad_sequence(dat[i], True, -1) for i in dat]


class MaskedNLLLoss(nn.Module):
    def __init__(self, weight=None):
        super(MaskedNLLLoss, self).__init__()
        self.weight = weight
        self.loss = nn.NLLLoss(weight=weight, reduction='sum')

    def forward(self, pred, target, mask):
        """pred: size(L, C)
           target: size(1, L)
           mask: (1, L) if <pad> the element equals 0"""
        mask1 = mask.view(-1, 1)                 # (L, 1)
        target1 = target.view(-1)                # (L)
        if self.weight is None:
            loss = self.loss(pred*mask1, target1) / torch.sum(mask1)
        else:
            loss = self.loss(pred*mask1, target1) / torch.sum(self.weight[target1]*mask1.squeeze(1))

        return loss


class RnnDataset(Dataset):
    def __init__(self, path, train=True):
        super(RnnDataset, self).__init__()
        data = pickle.load(open(path, 'rb'), encoding='latin1')
        train_data = data['train']
        test_data = data['test']
        if train:
            self.text = train_data[0]
            wordm = train_data[1]
            self.label = train_data[2]
            self.slabel = train_data[3]
            self.speaker = train_data[4]
        else:
            self.text = test_data[0]
            wordm = test_data[1]
            self.label = test_data[2]
            self.slabel = test_data[3]
            self.speaker = test_data[4]
        self.wmask = []
        for conv in wordm:
            conv = np.array(conv)
            clen = np.sum(conv, axis=1)
            self.wmask.append(clen)
        self.len = len(self.label)

    def __len__(self):
        return self.len

    def __getitem__(self, item):
        conv = self.text[item]
        labl = self.label[item]
        slbl = self.slabel[item]
        spkr = [0 if s == 'M' else 1 for s in self.speaker[item]]
        attn = self.wmask[item]
        uttm = [1] * len(labl)

        # (conv_len, num_word)
        conv = torch.tensor(conv, dtype=torch.long)
        # (conv_len)
        clen = torch.tensor(attn, dtype=torch.long)
        labl = torch.tensor(labl, dtype=torch.long)
        slbl = torch.tensor(slbl, dtype=torch.long)
        spkr = torch.tensor(spkr, dtype=torch.float)
        uttm = torch.tensor(uttm, dtype=torch.float)
        return conv, labl, slbl, clen, uttm, spkr


def collate_func(data):
    # conv, attn: [(conv_len, sent_len)]   lbl, spkr: [(conv_len)]
    data = data[0]
    conv, lbl, slbl, clen, uttm, spkr = data
    return conv, lbl, slbl, clen, uttm, spkr
