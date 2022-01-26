import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from transformers import BertModel, RobertaModel, XLNetModel


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
            #
            # concatenating all the utterances in conversations
            #

            # a batch of conversations with size of [(NUM_UTT_1, UTT_LEN_1), (NUM_UTT_2, UTT_LEN_2),
            # ..., (NUM_UTT_b, UTT_LEN_b)] to a tensor with size of (ALL_UTT, MAX_UTT_LEN),
            # where ALL_UTT = sum(NUM_UTT_x); MAX_UTT_LEN = max(NUM_UTT_x)
            max_utt_len = max([int(c.size(1)) for c in conversations])

            conversation = list()
            msk = list()
            for c, m in zip(conversations, mask):
                conversation.append(torch.cat((c, torch.zeros((c.size(0), max_utt_len-c.size(1)), dtype=torch.long)), dim=1))
                msk.append(torch.cat((m, torch.zeros((m.size(0), max_utt_len-m.size(1)))), dim=1))
            # (ALL_UTT, MAX_UTT_LEN)
            conversation = torch.cat(conversation, dim=0)
            msk = torch.cat(msk, dim=0)
            if use_gpu:
                conversation = conversation.cuda()
                msk = msk.cuda()

            #
            # processed with the pretrained model
            #

            output = self.encoder(conversation, msk)
            # (ALL_UTT, MAX_UTT_LEN, OUT_DIM) AND (ALL_UTT, OUT_DIM)
            hidden_state = output['last_hidden_state']
            pooler_output = output['pooler_output']
            if self.mode == 'cls':
                sent_emb = self.mapping(pooler_output)
            elif self.mode == 'maxpooling':
                sent_emb = self.mapping(torch.max(hidden_state, dim=1)[0])
            else:
                sent_emb = None
                assert self.mode in ['cls', 'maxpooling'], 'not support other operations'
        return sent_emb

    def __repr__(self):
        return '{}({}, mode={}, out_dim={})'.format(self.__class__.__name__,
                                                    self.encoder.__class__.__name__,
                                                    self.mode,
                                                    self.out_dim)
