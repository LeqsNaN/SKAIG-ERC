import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.init import xavier_uniform_

from bert_utils import AbsolutePositionEncoding, _get_clones
from mask import build_mixed_mask_local
from triple import FusionAttention
from bert_utils import TransformerLayerAbs, TransformerLayerAbs3D
from transformers import BertModel
from torch.nn.utils.rnn import pad_sequence


class PositionEncoding(nn.Module):
    def __init__(self, input_dim, max_len=200):
        super(PositionEncoding, self).__init__()
        self.max_len = max_len
        pe = torch.zeros(max_len, input_dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0., input_dim, 2) * -(math.log(10000.) / input_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # (bsz, slen, dim)
        seq_len = x.size(1)
        pe_clip = self.pe[:seq_len]
        pemb = x + pe_clip.unsqueeze(0)
        return pemb


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

    def forward(self, src, utt_mask, spk_mask, window=100, mode='so'):
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
                if mode == 'u':
                    output = self.layers1[i](output, uttm)
                elif mode == 's':
                    output = self.layers1[i](output, samm)
                elif mode == 'o':
                    output = self.layers1[i](output, othm)
                else:
                    raise NotImplementedError
        elif self.num_block == 2:
            output1 = src
            output2 = src
            for i in range(self.num_layer):
                if mode == 'so':
                    output1 = self.layers1[i](output1, samm)
                    output2 = self.layers2[i](output2, othm)
                elif mode == 'us':
                    output1 = self.layers1[i](output1, uttm)
                    output2 = self.layers2[i](output2, samm)
                elif mode == 'uo':
                    output1 = self.layers1[i](output1, uttm)
                    output2 = self.layers2[i](output2, othm)
                else:
                    raise NotImplementedError
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
            assert 1 <= self.num_block <= 3, 'ooc'

        log_prob = F.log_softmax(self.classifier(output), dim=-1)
        return log_prob


class HiTransformer(nn.Module):
    def __init__(self,
                 layer,
                 nhead,
                 num_layer,
                 emb_dim,
                 max_len,
                 num_class,
                 bidirectional,
                 norm=None):
        super(HiTransformer, self).__init__()
        self.nhead = nhead
        self.bidirectional = bidirectional
        self.num_layer = num_layer
        self.norm = norm
        self.pe = AbsolutePositionEncoding(emb_dim, max_len)

        self.layers1 = _get_clones(layer, num_layer)
        self.layers2 = _get_clones(layer, num_layer)
        self.layers3 = _get_clones(layer, num_layer)

        self.fusion1 = FusionAttention(emb_dim)
        self.fusion2 = FusionAttention(emb_dim)

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
        uttm, samm, othm = build_mixed_mask_local(utt_mask.unsqueeze(0), spk_mask.unsqueeze(0),
                                                  window, self.bidirectional)
        uttm = uttm.expand(self.nhead, src_len, src_len)
        samm = samm.expand(self.nhead, src_len, src_len)
        othm = othm.expand(self.nhead, src_len, src_len)

        src = self.pe(src)
        ct_output = src
        sm_output = src
        om_output = src
        for i in range(self.num_layer):
            ct_output = self.layers1[i](ct_output, uttm)
            sm_output = self.layers2[i](sm_output, samm)
            om_output = self.layers3[i](om_output, othm)
        sp_output = torch.stack([sm_output, om_output], dim=0)
        sp_output = self.fusion1(sp_output)
        output = torch.stack([ct_output, sp_output], dim=0)
        output = self.fusion2(output)

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
    def forward(self, conv, attn_mask, utt_mask, spk_mask, window=100, mode='so'):
        # (conv_len, sent_len, 768)
        conv_emb = self.bert(conv, attn_mask)[0]
        conv_pooler = torch.max(conv_emb, dim=1)[0]
        conv_pooler = self.proj(conv_pooler)
        log_prob = self.transformer(conv_pooler, utt_mask, spk_mask, window, mode)
        return log_prob


class GoodLuck2Me(nn.Module):
    def __init__(self,
                 num_class,
                 num_layer=0,
                 max_len=0,
                 emb_dim=0,
                 nhead=0,
                 ff_dim=0,
                 dropout=0.,
                 activation='relu',
                 bidirectional=True,
                 attn_mask=False):
        super(GoodLuck2Me, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.proj = nn.Linear(768, emb_dim)
        trans = TransformerLayerAbs(emb_dim, nhead, ff_dim, dropout, activation, attn_mask)
        self.transformer = HiTransformer(trans, nhead, num_layer, emb_dim, max_len, num_class, bidirectional)

    def forward(self, conv, attn_mask, utt_mask, spk_mask, window=100):
        conv_emb = self.bert(conv, attn_mask)[0]
        conv_pooler = torch.max(conv_emb, dim=1)[0]
        conv_pooler = self.proj(conv_pooler)
        log_prob = self.transformer(conv_pooler, utt_mask, spk_mask, window)
        return log_prob


class FusionAttention3D(nn.Module):
    def __init__(self, input_dim):
        super(FusionAttention3D, self).__init__()
        self.input_dim = input_dim
        self.wf = nn.Parameter(torch.empty((1, 1, input_dim, 1)), requires_grad=True)

    def forward(self, feat, mask):
        # (bsz, 3, seq_len, sent_dim)
        # (bsz, seq_len)
        bsz = feat.size(0)
        seq_len = feat.size(2)
        sent_dim = feat.size(3)
        # (bsz, 3, seq_len, sent_dim) -> (bsz, seq_len, 3, sent_dim)
        feat = feat.transpose(1, 2)
        # (bsz, seq_len, 3, 1)
        alpha = torch.matmul(feat, self.wf.expand(bsz, seq_len, sent_dim, 1))
        alpha = F.softmax(alpha, dim=2)
        mask = mask.unsqueeze(2).unsqueeze(3)
        alpha = alpha * mask
        # (bsz, seq_len, 1, 3)*(bsz, seq_len, 3, sent_dim) -> (bsz, seq_len, 1, sent_dim)
        out = torch.matmul(alpha.transpose(2, 3), feat)

        # (bsz, seq_len, sent_dim)
        return out.squeeze(2)


class HiBatchTransformer(nn.Module):
    def __init__(self,
                 layer,
                 nhead,
                 num_layer,
                 emb_dim,
                 max_len,
                 num_class,
                 bidirectional,
                 norm=None):
        super(HiBatchTransformer, self).__init__()
        self.nhead = nhead
        self.bidirectional = bidirectional
        self.num_layer = num_layer
        self.norm = norm
        self.pe = PositionEncoding(emb_dim, max_len)

        self.layers1 = _get_clones(layer, num_layer)
        self.layers2 = _get_clones(layer, num_layer)
        self.layers3 = _get_clones(layer, num_layer)

        self.fusion1 = FusionAttention3D(emb_dim)
        self.fusion2 = FusionAttention3D(emb_dim)

        self.classifier = nn.Linear(emb_dim, num_class)
        self._reset_parameter()

    def _reset_parameter(self):
        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)

    def forward(self, src, utt_mask, spk_mask, window=100):
        bsz = src.size(0)
        src_len = src.size(1)

        # ##### make masks
        # (bsz, slen, slen)
        uttm, samm, othm = build_mixed_mask_local(utt_mask, spk_mask, window, self.bidirectional)
        # (bsz, 1, slen, slen) -> (bsz*nh, slen, slen)
        uttm = uttm.unsqueeze(1).expand(bsz, self.nhead, src_len, src_len)
        uttm = uttm.contiguous().view(bsz*self.nhead, src_len, src_len)

        samm = samm.unsqueeze(1).expand(bsz, self.nhead, src_len, src_len)
        samm = samm.contiguous().view(bsz*self.nhead, src_len, src_len)

        othm = othm.unsqueeze(1).expand(bsz, self.nhead, src_len, src_len)
        othm = othm.contiguous().view(bsz*self.nhead, src_len, src_len)

        src = self.pe(src)
        # (bsz, slen, dim) -> (slen, bsz, dim)
        src = src.transpose(0, 1)
        ct_output = src
        sm_output = src
        om_output = src
        for i in range(self.num_layer):
            ct_output = self.layers1[i](ct_output, uttm)
            sm_output = self.layers2[i](sm_output, samm)
            om_output = self.layers3[i](om_output, othm)
        # (bsz, slen, dim)
        ct_output = ct_output.transpose(0, 1)
        sm_output = sm_output.transpose(0, 1)
        om_output = om_output.transpose(0, 1)
        # (bsz, 2, slen, dim)
        sp_output = torch.stack([sm_output, om_output], dim=1)
        sp_output = self.fusion1(sp_output, utt_mask)
        output = torch.stack([ct_output, sp_output], dim=1)
        output = self.fusion2(output, utt_mask)

        log_prob = F.log_softmax(self.classifier(output), dim=-1)
        return log_prob


class BatchTransformer(nn.Module):
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
        super(BatchTransformer, self).__init__()
        self.nhead = nhead
        self.bidirectional = bidirectional
        self.num_layer = num_layer
        self.norm = norm
        self.num_block = num_block
        self.pe = PositionEncoding(emb_dim, max_len)
        if self.num_block == 1:
            self.layers1 = _get_clones(layer, num_layer)
        elif self.num_block == 2:
            self.layers1 = _get_clones(layer, num_layer)
            self.layers2 = _get_clones(layer, num_layer)
            self.fusion = FusionAttention3D(emb_dim)
        elif self.num_block == 3:
            self.layers1 = _get_clones(layer, num_layer)
            self.layers2 = _get_clones(layer, num_layer)
            self.layers3 = _get_clones(layer, num_layer)
            self.fusion = FusionAttention3D(emb_dim)
        else:
            assert 1 <= num_block <= 3, 'ooc'
        # self.layers = _get_clones(layer, num_layer)
        self.classifier = nn.Linear(emb_dim, num_class)
        self._reset_parameter()

    def _reset_parameter(self):
        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)

    def forward(self, src, utt_mask, spk_mask, window=100, mode='so'):
        bsz = src.size(0)
        src_len = src.size(1)

        # ##### make masks
        # (1, src_len, tgt_len)
        uttm, samm, othm = build_mixed_mask_local(utt_mask, spk_mask,
                                                  window, self.bidirectional)

        uttm = uttm.unsqueeze(1).expand(bsz, self.nhead, src_len, src_len)
        uttm = uttm.contiguous().view(bsz * self.nhead, src_len, src_len)

        samm = samm.unsqueeze(1).expand(bsz, self.nhead, src_len, src_len)
        samm = samm.contiguous().view(bsz * self.nhead, src_len, src_len)

        othm = othm.unsqueeze(1).expand(bsz, self.nhead, src_len, src_len)
        othm = othm.contiguous().view(bsz * self.nhead, src_len, src_len)

        src = self.pe(src)
        if self.num_block == 1:
            output = src.transpose(0, 1)
            for i in range(self.num_layer):
                if mode == 'u':
                    output = self.layers1[i](output, uttm)
                elif mode == 's':
                    output = self.layers1[i](output, samm)
                elif mode == 'o':
                    output = self.layers1[i](output, othm)
                else:
                    raise NotImplementedError
            output = output.transpose(0, 1)
        elif self.num_block == 2:
            output1 = src.transpose(0, 1)
            output2 = src.transpose(0, 1)
            for i in range(self.num_layer):
                if mode == 'so':
                    output1 = self.layers1[i](output1, samm)
                    output2 = self.layers2[i](output2, othm)
                elif mode == 'us':
                    output1 = self.layers1[i](output1, uttm)
                    output2 = self.layers2[i](output2, samm)
                elif mode == 'uo':
                    output1 = self.layers1[i](output1, uttm)
                    output2 = self.layers2[i](output2, othm)
                else:
                    raise NotImplementedError
            output1 = output1.transpose(0, 1)
            output2 = output2.transpose(0, 1)
            # (bsz, 2, seq_len, sent_dim)
            output = torch.stack([output1, output2], dim=1)
            output = self.fusion(output, utt_mask)
        elif self.num_block == 3:
            output1 = src.transpose(0, 1)
            output2 = src.transpose(0, 1)
            output3 = src.transpose(0, 1)
            for i in range(self.num_layer):
                output1 = self.layers1[i](output1, uttm)
                output2 = self.layers2[i](output2, samm)
                output3 = self.layers3[i](output3, othm)
            output1 = output1.transpose(0, 1)
            output2 = output2.transpose(0, 1)
            output3 = output3.transpose(0, 1)
            # (bsz, 3, seq_len, sent_dim)
            output = torch.stack([output1, output2, output3], dim=1)
            output = self.fusion(output, utt_mask)
        else:
            output = None
            assert 1 <= self.num_block <= 3, 'ooc'

        log_prob = F.log_softmax(self.classifier(output), dim=-1)
        return log_prob


class BatchTRMSM(nn.Module):
    def __init__(self,
                 num_class,
                 num_layer=0,
                 num_block=0,
                 max_len=0,
                 emb_dim=0,
                 nhead=0,
                 ff_dim=0,
                 dropout=0.,
                 activation='relu',
                 bidirectional=True,
                 attn_mask=False,
                 model_type='par'):
        super(BatchTRMSM, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.proj = nn.Linear(768, emb_dim)
        trans = TransformerLayerAbs3D(emb_dim, nhead, ff_dim, dropout, activation, attn_mask)
        self.transformer = HiTransformer(trans, nhead, num_layer, emb_dim, max_len, num_class, bidirectional)
        self.mode = model_type
        if model_type == 'par':
            self.transformer = BatchTransformer(trans, nhead, num_layer, emb_dim, max_len,
                                                num_class, bidirectional, num_block, None)
        elif model_type == 'hi':
            self.transformer = HiBatchTransformer(trans, nhead, num_layer, emb_dim,
                                                  max_len, num_class, bidirectional, None)
        else:
            raise NotImplementedError

    def forward(self, conv, attn_mask, utt_mask, spk_mask, window=100, use_gpu=True, mod='so'):
        conv_emb = []
        for c, a in zip(conv, attn_mask):
            if use_gpu:
                c = c.cuda()
                a = a.cuda()
            # (slen, wlen, dim)
            cemb = self.bert(c, a)[0]
            # (slen, dim)
            cpool = torch.max(cemb, dim=1)[0]
            cpool = self.proj(cpool)
            conv_emb.append(cpool)
        utt_mask = pad_sequence(utt_mask, batch_first=True, padding_value=0)
        spk_mask = pad_sequence(spk_mask, batch_first=True, padding_value=0)
        conv_emb = pad_sequence(conv_emb, batch_first=True, padding_value=0.)
        if use_gpu:
            utt_mask = utt_mask.cuda()
            spk_mask = spk_mask.cuda()
        if self.mode == 'par':
            log_prob = self.transformer(conv_emb, utt_mask, spk_mask, window, mod)
        else:
            log_prob = self.transformer(conv_emb, utt_mask, spk_mask, window)
        return log_prob


class MaskedNLLLoss(nn.Module):
    def __init__(self):
        super(MaskedNLLLoss, self).__init__()
        self.loss_fn = nn.NLLLoss(reduction='sum')

    def forward(self, preds, label, mask):
        # (bsz, slen, ncls); (bsz, slen); (bsz, slen)
        bsz = preds.size(0)
        slen = preds.size(1)
        mask1 = mask.contiguous().view(-1, 1)
        preds1 = preds.contiguous().view(bsz*slen, -1)
        label1 = label.contiguous().view(-1)
        loss = self.loss_fn(preds1*mask1, label1) / torch.sum(mask)
        return loss
