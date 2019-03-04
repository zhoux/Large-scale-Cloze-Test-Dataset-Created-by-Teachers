import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
from torch.autograd import Variable
import Attention
import Constants
import random
import time
import string

def check_decreasing(lengths):
    lens, order = torch.sort(lengths, 0, True)
    if torch.ne(lens, lengths).long().sum() == 0:
        return None
    else:
        _, rev_order = torch.sort(order)
        return lens, Variable(order), Variable(rev_order)

_INF = float('inf')

class InforWeightedCE(nn.Module):
    def __init__(self, weight, infor_softmax_temp):
        super(InforWeightedCE, self).__init__()
        self.weight = weight
        if self.weight is not None:
            #self.weight = Variable(self.weight).cuda()
            self.weight = Variable(self.weight)
        self.infor_softmax_temp = infor_softmax_temp * 1.
        self.sm = nn.Softmax()

    def forward(self, input, target, infor, vocab=None):
        index_offset = torch.arange(0, target.size(0)).long() * input.size(1)
        #index_offset = Variable(index_offset.cuda())
        index_offset = Variable(index_offset)
        scores = input.view(-1).index_select(0, target + index_offset)
        if self.weight is not None:
            mask_0 = self.weight[target.data]
            scores = scores * mask_0
        else:
            mask_0 = None
        if infor is not None:
            entropy_flat = infor.view(-1).contiguous()
            if mask_0 is not None:
                entropy_flat.data.masked_fill_((1 - mask_0).byte().data, -_INF)
            entropy_softmax = self.sm(entropy_flat.unsqueeze(0) / self.infor_softmax_temp)[0]
            scores = scores * entropy_softmax
            final_score = - scores.sum()
        else:
            if mask_0 is not None:
                final_score = - scores.sum() / mask_0.sum()
            else:
                final_score = - scores.mean()
        return final_score

class Model(nn.Module):

    def __init__(self, args, ntoken, embeddings):
        super(Model, self).__init__()
        self.dropout = nn.Dropout(args.dropout)
        self.ntoken = ntoken
        self.emb = nn.Embedding(ntoken, args.emsize, padding_idx=Constants.PAD)
        self.emb.weight.data.copy_(embeddings)
        bidir = True
        assert bidir
        self.other_word_cost = args.other_word_cost
        self.num_directions = 2 if bidir else 1
        self.nhid = args.nhid // 2 if bidir else args.nhid
        self.RNN_type = args.RNN_type
        self.softmax = nn.Softmax()
        self.use_cuda = args.cuda
        if self.RNN_type == "LSTM":
            self.article_rnn = nn.LSTM(args.emsize, self.nhid, args.nlayers, dropout=args.dropout, bidirectional=bidir) #dropout introduces a dropout layer on the outputs of each RNN layer except the last layer
        elif self.RNN_type == "GRU":
            self.article_rnn = nn.GRU(args.emsize, self.nhid, args.nlayers, dropout=args.dropout, bidirectional=bidir)
        else:
            assert False
        self.nlayers = args.nlayers
        self.generator = nn.Linear(args.emsize, ntoken, bias=False)
        if args.tied:
            self.generator.weight = self.emb.weight
        else:
            self.generator.weight.data.copy_(embeddings)
        self.out_emb = self.generator.weight
        self.out_bias = self.generator.bias
        att_hid = args.nhid
        self.output_att = Attention.Attention_wrap(att_hid, args.emsize)
        self.use_cuda = args.cuda

    def _fix_enc_hidden(self, h): #Useful for LSTM
        #  the encoder hidden is  (layers*directions) x batch x dim
        #  we need to convert it to layers x batch x (directions*dim)
        if self.num_directions == 2:
            return h.view(h.size(0) // 2, 2, h.size(1), h.size(2)) \
                .transpose(1, 2).contiguous() \
                .view(h.size(0) // 2, h.size(1), h.size(2) * 2)
        else:
            return h

    def forward_rnn(self, rnn, input):
        emb = self.emb(input)
        emb = self.dropout(emb)
        mask = input.data.ne(Constants.PAD).long()
        lengths = torch.sum(mask, 0).squeeze(0)
        check_res = check_decreasing(lengths)
        if check_res is None:
            packed_emb = rnn_utils.pack_padded_sequence(emb, lengths.tolist())
            packed_out, hidden_t = rnn(packed_emb)
            outputs, srclens = rnn_utils.pad_packed_sequence(packed_out)
        else:
            lens, order, rev_order = check_res
            packed_emb = rnn_utils.pack_padded_sequence(emb.index_select(1, order), lens.tolist())
            packed_out, hidden_t = rnn(packed_emb)
            outputs, srclens = rnn_utils.pad_packed_sequence(packed_out)
            outputs = outputs.index_select(1, rev_order)
            if self.RNN_type == "LSTM":
                hidden_t = (hidden_t[0].index_select(1, rev_order),
                            hidden_t[1].index_select(1, rev_order))
            else:
                hidden_t = hidden_t.index_select(1, rev_order)
        return outputs, hidden_t

    def forward(self, articles, options, placeholder_idx, article_idx):
        c_article, h_article = self.forward_rnn(self.article_rnn, articles)
        option_emb = self.out_emb[options.data.view(-1)].view(options.size(0), options.size(1), -1)
        option_mask = torch.ne(options, Constants.PAD).float()
        option_emb = option_emb * option_mask.unsqueeze(2).expand_as(option_emb)
        option_emb = option_emb.sum(dim=0).squeeze()
        option_length = option_mask.sum(dim=0).squeeze()
        option_emb = option_emb // option_length.unsqueeze(1).expand_as(option_emb)
        h_options = option_emb
        num_que = placeholder_idx.size(0) #the number of questions in this batch
        h_options = h_options.view(num_que, 4, -1)
        c_article = c_article.transpose(0, 1).contiguous()
        c_article_flat = c_article.view(-1, c_article.size()[2])
        forward_c_article_flat = c_article_flat[:, : self.nhid]
        backward_c_article_flat = c_article_flat[:, self.nhid :]
        if self.other_word_cost > 0: #and self.option_mean_pooling:
            aug_forward = c_article[:, :-2, :self.nhid].contiguous().view(-1, self.nhid)
            aug_backward = c_article[:, 2:, self.nhid:].contiguous().view(-1, self.nhid)
            aug_rep = torch.cat([aug_forward, aug_backward], 1)
            proj_rep = self.output_att.attn_core.linear_in(aug_rep)
            score_other_word = self.generator(proj_rep)
            score_other_word_sm = self.softmax(score_other_word)
        else:
            score_other_word = None
            score_other_word_sm = None

        rep_at_token_forward = forward_c_article_flat.index_select(0, placeholder_idx - 1)
        rep_at_token_backward = backward_c_article_flat.index_select(0, placeholder_idx + 1)

        rep_at_token = torch.cat([rep_at_token_forward, rep_at_token_backward], 1)

        rep_article = rep_at_token
        final = rep_article

        _, sm_score, score = self.output_att(final, h_options)
        return sm_score, score_other_word_sm