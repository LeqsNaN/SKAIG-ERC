import torch
import torch.nn as nn
import torch.nn.functional as F

from distill import TransformerLayerAbs, TransformerLayerAbs3D
from mask import build_mixed_mask_local
from torch.nn.init import xavier_uniform_
from triple import FusionAttention
from torch.nn.utils.rnn import pad_sequence


from transformer import _get_clones
from bert_utils import AbsolutePositionEncoding
from enhance_triple import PositionEncoding, FusionAttention3D

from transformers import BertModel, RobertaModel, XLNetModel
from torch_geometric.nn.conv import TransformerConv


class UtteranceEncoder(nn.Module):
    def __init__(self, model_type, mode, out_dim):
        super(UtteranceEncoder, self).__init__()
        self.model_type = model_type
        self.mode = mode
        self.out_dim = out_dim
        if model_type == 'bert-base-uncased':
            self.encoder = BertModel.from_pretrained('bert-base-uncased')
            self.hidden_dim = 768
        elif model_type == 'roberta-base':
            self.encoder = RobertaModel.from_pretrained('roberta-base')
            self.hidden_dim = 768
        elif model_type == 'roberta-large':
            self.encoder = RobertaModel.from_pretrained('roberta-large')
            self.hidden_dim = 1024
        elif model_type == 'xlnet-base-cased':
            self.encoder = XLNetModel.from_pretrained('xlnet-base-cased')
            self.hidden_dim = 768
        else:
            assert model_type in ['bert-base-uncased', 'roberta-base', 'roberta-large', 'xlnet-base-cased'], 'not support this pretrained model'
        self.mapping = nn.Linear(self.hidden_dim, out_dim)

    def forward(self, conversations, mask, use_gpu=True):
        if isinstance(conversations, torch.Tensor):
            output = self.encoder(conversations, mask)
            hidden_state = output['last_hidden_state']
            pooler_output = output['pooler_output']
            if self.mode == 'cls':
                # (C, D)
                sent_emb = self.mapping(pooler_output)
            elif self.mode == 'maxpooling':
                sent_emb = self.mapping(torch.max(hidden_state, dim=1)[0])
            else:
                sent_emb = None
                assert self.mode in ['cls', 'maxpooling'], 'not support other operation'
        else:
            hidden_state = []
            pooler_output = []
            sent_emb = []
            for conversation, msk in zip(conversations, mask):
                if use_gpu:
                    conversation = conversation.cuda()
                    msk = msk.cuda()
                # (C, L, D), (C, D)
                output = self.encoder(conversation, msk)
                hidden_state.append(output['last_hidden_state'])
                pooler_output.append(output['pooler_output'])
            if self.mode == 'cls':
                # B * (C, D)
                for p in pooler_output:
                    sent_emb.append(self.mapping(p))
            elif self.mode == 'maxpooling':
                for hd in hidden_state:
                    sent_emb.append(self.mapping(torch.max(hd, dim=1)[0]))
            else:
                sent_emb = None
                assert self.mode in ['cls', 'maxpooling'], 'not support other operation'
        return sent_emb

    def __repr__(self):
        return '{}({}, mode={}, out_dim={})'.format(self.__class__.__name__,
                                                    self.encoder.__class__.__name__,
                                                    self.mode,
                                                    self.out_dim)


class FFMLP(nn.Module):
    def __init__(self, input_dim, ff_dim, dropout):
        super(FFMLP, self).__init__()
        self.input_dim = input_dim
        self.ff_dim = ff_dim
        self.linear1 = nn.Linear(input_dim, ff_dim)
        self.linear2 = nn.Linear(ff_dim, input_dim)
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src):
        ss = self.norm1(src)
        ss2 = self.linear2(self.dropout1(F.relu(self.linear1(ss))))
        ss = ss + self.dropout2(ss2)
        ss = self.norm2(ss)
        return ss


class TRMCN(nn.Module):
    def __init__(self, in_channels, ff_dim, out_channels, heads, dropout, edge_dim, bias, num_layers, edge_mapping, beta, root_weight):
        super(TRMCN, self).__init__()
        assert in_channels == heads * out_channels, 'in_channels must equal heads * out_channels'
        self.edge_mapping = edge_mapping
        self.heads = heads
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dropout = dropout
        self.bias = bias
        self.num_layers = num_layers
        if edge_mapping:
            self.mapping = nn.Linear(768, edge_dim)
            self.edge_dim = edge_dim
        else:
            self.mapping = None
            self.edge_dim = 768

        conv = TransformerConv(in_channels=in_channels,
                               out_channels=out_channels,
                               heads=heads, concat=True,
                               beta=beta, dropout=dropout,
                               edge_dim=edge_dim, bias=bias,
                               root_weight=root_weight)
        ff = FFMLP(in_channels, ff_dim, dropout)
        self.convs = _get_clones(conv, num_layers)
        self.ffnet = _get_clones(ff, num_layers)

    def forward(self, x, edge_index, edge_attr):
        if self.mapping is not None:
            edge_attr = self.mapping(edge_attr)
        for i in range(self.num_layers):
            x = self.convs[i](x=x, edge_index=edge_index, edge_attr=edge_attr)
            x = self.ffnet[i](x)
        return x


class TripleTransformer(nn.Module):
    def __init__(self, layer, nhead, num_layer, emb_dim, max_len, bidirectional, num_block, norm=None):
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

        return output


class BatchTransformer(nn.Module):
    def __init__(self, layer, nhead, num_layer, emb_dim,
                 max_len, bidirectional, num_block, norm=None):
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
        return output


class MentalModel(nn.Module):
    def __init__(self, encoder_type='bert-base-uncased', encoder_mode='maxpooling', sent_dim=300, tr_nhead=6,
                 tr_ff_dim=300, tr_dropout=0.1, attn_mask=True, tr_num_layer=6, max_len=120, num_class=6,
                 bidirectional=True, num_block=3, cn_nhead=6, cn_ff_dim=300, cn_dropout=0.1, edge_dim=300,
                 bias=False, cn_num_layer=3, edge_mapping=True, beta=True, root_weight=True, choice='both'):
        super(MentalModel, self).__init__()
        self.choice = choice
        self.uttrenc = UtteranceEncoder(encoder_type, encoder_mode, sent_dim)
        if choice == 'both':
            trm_layer = TransformerLayerAbs(sent_dim, tr_nhead, tr_ff_dim, tr_dropout, 'relu', attn_mask)
            self.convenc = TripleTransformer(trm_layer, tr_nhead, tr_num_layer,
                                             sent_dim, max_len, bidirectional, num_block)
            self.tcn = TRMCN(sent_dim, cn_ff_dim, sent_dim//cn_nhead, cn_nhead, cn_dropout, edge_dim,
                             bias, cn_num_layer, edge_mapping, beta, root_weight)
            self.classifier = nn.Linear(2 * sent_dim, num_class)
        elif choice == 'tr':
            trm_layer = TransformerLayerAbs(sent_dim, tr_nhead, tr_ff_dim, tr_dropout, 'relu', attn_mask)
            self.convenc = TripleTransformer(trm_layer, tr_nhead, tr_num_layer,
                                             sent_dim, max_len, bidirectional, num_block)
            self.classifier = nn.Linear(sent_dim, num_class)
        elif choice == 'cn':
            self.tcn = TRMCN(sent_dim, cn_ff_dim, sent_dim // cn_nhead, cn_nhead, cn_dropout, edge_dim,
                             bias, cn_num_layer, edge_mapping, beta, root_weight)
            self.classifier = nn.Linear(sent_dim, num_class)
        else:
            raise NotImplementedError()

    def forward(self, conversation, mask, utt_mask=None, spk_mask=None,
                window=None, mode=None, edge_index=None, edge_attr=None, residual=False):
        sent_emb = self.uttrenc(conversation, mask)
        if self.choice == 'both':
            contextualized_emb = self.convenc(sent_emb, utt_mask, spk_mask, window, mode)
            mental_emb = self.tcn(sent_emb, edge_index, edge_attr)
            concat_emb = torch.cat([contextualized_emb, mental_emb], dim=1)
            logits = self.classifier(concat_emb)
        elif self.choice == 'tr':
            contextualized_emb = self.convenc(sent_emb, utt_mask, spk_mask, window, mode)
            logits = self.classifier(contextualized_emb)
        else:
            mental_emb = self.tcn(sent_emb, edge_index, edge_attr)
            if residual:
                mental_emb = mental_emb + sent_emb
            logits = self.classifier(mental_emb)
        return logits


class BatchMentalModel(nn.Module):
    def __init__(self, encoder_type='roberta-base', encoder_mode='maxpooling', sent_dim=200, tr_nhead=4,
                 tr_ff_dim=200, tr_dropout=0.1, attn_mask=True, tr_num_layer=5, max_len=120, num_class=7,
                 bidirectional=True, num_block=3, cn_nhead=6, cn_ff_dim=300, cn_dropout=0.1, edge_dim=300,
                 bias=False, cn_num_layer=3, edge_mapping=True, beta=True, root_weight=True, choice='both'):
        super(BatchMentalModel, self).__init__()
        self.uttrenc = UtteranceEncoder(encoder_type, encoder_mode, sent_dim)
        self.choice = choice
        if choice == 'both':
            trm_layer = TransformerLayerAbs3D(sent_dim, tr_nhead, tr_ff_dim, tr_dropout, 'relu', attn_mask)

            self.convenc = BatchTransformer(trm_layer, tr_nhead, tr_num_layer,
                                            sent_dim, max_len, bidirectional, num_block)

            self.tcn = TRMCN(sent_dim, cn_ff_dim, sent_dim // cn_nhead, cn_nhead, cn_dropout,
                             edge_dim, bias, cn_num_layer, edge_mapping, beta, root_weight)

            self.classifier = nn.Linear(2 * sent_dim, num_class)
        elif choice == 'tr':
            trm_layer = TransformerLayerAbs3D(sent_dim, tr_nhead, tr_ff_dim, tr_dropout, 'relu', attn_mask)

            self.convenc = BatchTransformer(trm_layer, tr_nhead, tr_num_layer,
                                            sent_dim, max_len, bidirectional, num_block)
            self.classifier = nn.Linear(sent_dim, num_class)
        elif choice == 'cn':
            self.tcn = TRMCN(sent_dim, cn_ff_dim, sent_dim // cn_nhead, cn_nhead, cn_dropout,
                             edge_dim, bias, cn_num_layer, edge_mapping, beta, root_weight)

            self.classifier = nn.Linear(sent_dim, num_class)
        else:
            raise NotImplementedError()

    def forward(self, conversations, masks, conv_len=None, utt_mask=None, spk_mask=None,
                window=None, mode=None, edge_indices=None, edge_attrs=None, use_gpu=True, residual=False):
        # B * (C, D)
        sent_emb = self.uttrenc(conversations, masks, use_gpu)
        if self.choice == 'both':
            # (B, C)
            spk_mask = pad_sequence(spk_mask, batch_first=True, padding_value=0)
            utt_mask = pad_sequence(utt_mask, batch_first=True, padding_value=0)
            if use_gpu:
                utt_mask = utt_mask.cuda()
                spk_mask = spk_mask.cuda()
            # (B, C, D)
            conv_input = pad_sequence(sent_emb, batch_first=True, padding_value=0.)
            contextualized_emb = self.convenc(conv_input, utt_mask, spk_mask, window, mode)
            udim = contextualized_emb.size(-1)

            # (BA, D)
            edge_attr = torch.cat(edge_attrs, dim=0)
            # (BC, D)
            graph_input = torch.cat(sent_emb, dim=0)

            # mini-batching the graphs
            batch = []
            cumbatch = []
            count = 0
            for i, l in enumerate(conv_len):
                batch += [i] * l
                cumbatch += [count] * l
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
            mental_emb = self.tcn(graph_input, edge_index, edge_attr)
            umask = utt_mask.eq(1).unsqueeze(2).expand_as(contextualized_emb)
            ctxt_emb = torch.masked_select(contextualized_emb, umask).contiguous().view(-1, udim)
            concat_emb = torch.cat([ctxt_emb, mental_emb], dim=1)
            logits = self.classifier(concat_emb)
        elif self.choice == 'tr':
            # (B, C)
            spk_mask = pad_sequence(spk_mask, batch_first=True, padding_value=0)
            utt_mask = pad_sequence(utt_mask, batch_first=True, padding_value=0)
            if use_gpu:
                utt_mask = utt_mask.cuda()
                spk_mask = spk_mask.cuda()
            # (B, C, D)
            conv_input = pad_sequence(sent_emb, batch_first=True, padding_value=0.)
            contextualized_emb = self.convenc(conv_input, utt_mask, spk_mask, window, mode)
            udim = contextualized_emb.size(-1)
            umask = utt_mask.eq(1).unsqueeze(2).expand_as(contextualized_emb)
            ctxt_emb = torch.masked_select(contextualized_emb, umask).contiguous().view(-1, udim)
            logits = self.classifier(ctxt_emb)
        else:
            # (BA, D)
            edge_attr = torch.cat(edge_attrs, dim=0)
            # (BC, D)
            graph_input = torch.cat(sent_emb, dim=0)

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
            mental_emb = self.tcn(graph_input, edge_index, edge_attr)
            if residual:
                mental_emb = mental_emb + graph_input
            logits = self.classifier(mental_emb)
        return logits
