#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Modified from CS224N 2018-19 Homework 4
nmt_model.py: NMT Model used as text autoencoder
Guoqin Ma <sebsk@stanford.edu>
"""
from collections import namedtuple
import sys
from typing import List, Tuple, Dict, Set, Union
import torch
import torch.nn as nn
import torch.nn.utils
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from model_embeddings import ModelEmbeddings


class Att2inCore(nn.Module):
    '''
    The network structure used in Att2inModel
    '''

    def __init__(self):
        super(Att2inCore, self).__init__()
        self.input_encoding_size = 512
        self.rnn_size = 512
        self.drop_prob_lm = 0
        self.fc_feat_size = 2048
        self.att_feat_size = 2048
        self.att_hid_size = 512

        # Build a LSTM
        self.a2c = nn.Linear(self.att_feat_size, 2 * self.rnn_size)
        self.i2h = nn.Linear(self.input_encoding_size, 5 * self.rnn_size)
        self.h2h = nn.Linear(self.rnn_size, 5 * self.rnn_size)
        self.dropout = nn.Dropout(self.drop_prob_lm)

        self.h2att = nn.Linear(self.rnn_size, self.att_hid_size)
        self.alpha_net = nn.Linear(self.att_hid_size, 1)

    def forward(self, xt, fc_feats, att_feats, p_att_feats, state):
        # The p_att_feats here is already projected
        att_size = att_feats.numel() // att_feats.size(0) // self.att_feat_size
        att = p_att_feats.view(-1, att_size, self.att_hid_size)

        att_h = self.h2att(state[0][-1])                        # batch * att_hid_size
        att_h = att_h.unsqueeze(1).expand_as(att)            # batch * att_size * att_hid_size
        dot = att + att_h                                   # batch * att_size * att_hid_size
        dot = F.tanh(dot)                                # batch * att_size * att_hid_size
        dot = dot.view(-1, self.att_hid_size)               # (batch * att_size) * att_hid_size
        dot = self.alpha_net(dot)                           # (batch * att_size) * 1
        dot = dot.view(-1, att_size)                        # batch * att_size

        weight = F.softmax(dot)                             # batch * att_size
        att_feats_ = att_feats.view(-1, att_size, self.att_feat_size) # batch * att_size * att_feat_size
        att_res = torch.bmm(weight.unsqueeze(1), att_feats_).squeeze(1) # batch * att_feat_size

        all_input_sums = self.i2h(xt) + self.h2h(state[0][-1])
        sigmoid_chunk = all_input_sums.narrow(1, 0, 3 * self.rnn_size)
        sigmoid_chunk = F.sigmoid(sigmoid_chunk)
        in_gate = sigmoid_chunk.narrow(1, 0, self.rnn_size)
        forget_gate = sigmoid_chunk.narrow(1, self.rnn_size, self.rnn_size)
        out_gate = sigmoid_chunk.narrow(1, self.rnn_size * 2, self.rnn_size)

        in_transform = all_input_sums.narrow(1, 3 * self.rnn_size, 2 * self.rnn_size) + \
            self.a2c(att_res)
        in_transform = torch.max(\
            in_transform.narrow(1, 0, self.rnn_size),
            in_transform.narrow(1, self.rnn_size, self.rnn_size))
        next_c = forget_gate * state[1][-1] + in_gate * in_transform
        next_h = out_gate * F.tanh(next_c)

        output = self.dropout(next_h)
        state = (next_h.unsqueeze(0), next_c.unsqueeze(0))
        return output, state

Hypothesis = namedtuple('Hypothesis', ['value', 'score'])

'''
To reconstruct RNN_ENCODER, we need to store:
ntoken: len(vocab.src) OR NMT.model_embeddings.source weight size(0)
ninput: embed_size[0], =300
nhidden: hidden_size[0], =128
RNN_ENCODER.encoder weight: NMT.model_embeddings.source weight
vocab.src.word2id AND vocab.src.id2word

To reconstruct Att2inModel, we need to store:
ntoken: len(vocab.src) OR NMT.model_embeddings.source weight size(0)
input_encoding_size: embed_size[1], =512
rnn_size: hidden_size[1], =512
Attn2inModel.core weight: NMT.decoder weight
Attn2inModel.logit weight: NMT.target_vocab_projection weight
Attn2inModel.embed weight: NMT.model_embeddings.target weight
vocab.tgt.word2id AND vocab.tgt.id2word
'''

class NMT(nn.Module):
    """ Simple Neural Machine Translation Model:
        - Bidrectional LSTM Encoder
        - Unidirection LSTM Decoder
        - Global Attention Model (Luong, et al. 2015)
    """
    def __init__(self, embed_size, hidden_size, vocab, dropout_rate=0.2, use_attention=True):
        """ Init NMT Model.

        @param embed_size (tuple): Embedding size (src, tgt)
        @param hidden_size (tuple): Hidden Size (src, tgt)
        @param vocab (Vocab): Vocabulary object containing src and tgt languages
                              See vocab.py for documentation.
        @param dropout_rate (float): Dropout probability, for attention
        """
        super(NMT, self).__init__()
        self.model_embeddings = ModelEmbeddings(embed_size, vocab)
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate
        self.vocab = vocab
        self.use_attention = use_attention

        if self.use_attention:
            print('Use attention', file=sys.stderr)
        else:
            print('Attention not used', file=sys.stderr)

        self.encoder = nn.LSTM(input_size=self.model_embeddings.embed_size[0], hidden_size=self.hidden_size[0], bidirectional=True)
        self.decoder = Att2inCore()
        self.att_projection = nn.Linear(in_features=self.hidden_size[0]*2, out_features=self.hidden_size[1], bias=False)
        self.h_projection = nn.Linear(in_features=self.hidden_size[0]*2, out_features=self.hidden_size[1], bias=False)
        self.c_projection = nn.Linear(in_features=self.hidden_size[0]*2, out_features=self.hidden_size[1], bias=False)
        self.combined_output_projection = nn.Linear(in_features=self.hidden_size[0]*2 + self.hidden_size[1], out_features=self.hidden_size[1], bias=False)
        self.target_vocab_projection = nn.Linear(in_features=self.hidden_size[1], out_features=len(self.vocab.tgt) + 1, bias=False)
        self.dropout = nn.Dropout(p=self.dropout_rate)

        assert self.decoder.rnn_size == self.hidden_size[1], 'check decoder input dim'

        ### END YOUR CODE

    def forward(self, source: List[List[str]], target: List[List[str]]) -> torch.Tensor:
        """ Take a mini-batch of source and target sentences, compute the log-likelihood of
        target sentences under the language models learned by the NMT system.

        @param source (List[List[str]]): list of source sentence tokens
        @param target (List[List[str]]): list of target sentence tokens, wrapped by `<s>` and `</s>`

        @returns scores (Tensor): a variable/tensor of shape (b, ) representing the
                                    log-likelihood of generating the gold-standard target sentence for
                                    each example in the input batch. Here b = batch size.
        """
        # Compute sentence lengths
        source_lengths = [len(s) for s in source]

        # Convert list of lists into tensors
        source_padded = self.vocab.src.to_input_tensor(source, device=self.device)   # Tensor: (src_len, b)
        target_padded = self.vocab.tgt.to_input_tensor(target, device=self.device)   # Tensor: (tgt_len, b)

        enc_hiddens, dec_init_state = self.encode(source_padded, source_lengths)
        enc_masks = self.generate_sent_masks(enc_hiddens, source_lengths)
        combined_outputs = self.decode(enc_hiddens, enc_masks, dec_init_state, target_padded)
        P = F.log_softmax(self.target_vocab_projection(combined_outputs), dim=-1)

        # Zero out, probabilities for which we have nothing in the target text
        target_masks = (target_padded != self.vocab.tgt['<pad>']).float()

        # Compute log probability of generating true target words
        target_gold_words_log_prob = torch.gather(P, index=target_padded[1:].unsqueeze(-1), dim=-1).squeeze(-1) * target_masks[1:]
        scores = target_gold_words_log_prob.sum(dim=0)
        return scores


    def encode(self, source_padded: torch.Tensor, source_lengths: List[int]) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """ Apply the encoder to source sentences to obtain encoder hidden states.
            Additionally, take the final states of the encoder and project them to obtain initial states for decoder.

        @param source_padded (Tensor): Tensor of padded source sentences with shape (b, src_len), where
                                        b = batch_size, src_len = maximum source sentence length. Note that
                                       these have already been sorted in order of longest to shortest sentence.
        @param source_lengths (List[int]): List of actual lengths for each of the source sentences in the batch
        @returns enc_hiddens (Tensor): Tensor of hidden units with shape (b, src_len, h*2), where
                                        b = batch size, src_len = maximum source sentence length, h = hidden size.
        @returns dec_init_state (tuple(Tensor, Tensor)): Tuple of tensors representing the decoder's initial
                                                hidden state and cell.
        """

        X = self.model_embeddings.source(source_padded)
        enc_hiddens, (last_hidden, last_cell) = self.encoder(pack_padded_sequence(X, source_lengths))
        enc_hiddens, _ = pad_packed_sequence(enc_hiddens, batch_first=True)
        init_decoder_hidden = self.h_projection(torch.cat((last_hidden[0], last_hidden[1]), dim=1))
        init_decoder_cell = self.c_projection(torch.cat((last_cell[0], last_cell[1]), dim=1))
        dec_init_state = (init_decoder_hidden, init_decoder_cell)
        ### END YOUR CODE

        return enc_hiddens, dec_init_state


    def decode(self, enc_hiddens: torch.Tensor, enc_masks: torch.Tensor,
                dec_init_state: Tuple[torch.Tensor, torch.Tensor], target_padded: torch.Tensor) -> torch.Tensor:
        """Compute combined output vectors for a batch.

        @param enc_hiddens (Tensor): Hidden states (b, src_len, h*2), where
                                     b = batch size, src_len = maximum source sentence length, h = hidden size.
        @param enc_masks (Tensor): Tensor of sentence masks (b, src_len), where
                                     b = batch size, src_len = maximum source sentence length.
        @param dec_init_state (tuple(Tensor, Tensor)): Initial state and cell for decoder
        @param target_padded (Tensor): Gold-standard padded target sentences (tgt_len, b), where
                                       tgt_len = maximum target sentence length, b = batch size.

        @returns combined_outputs (Tensor): combined output tensor  (tgt_len, b,  h), where
                                        tgt_len = maximum target sentence length, b = batch_size,  h = hidden size
        """
        # Chop of the <END> token for max length sentences.
        target_padded = target_padded[:-1]

        # Initialize the decoder state (hidden and cell)
        dec_state = dec_init_state

        # Initialize previous combined output vector o_{t-1} as zero
        batch_size = enc_hiddens.size(0)

        # Initialize a list we will use to collect the combined output o_t on each step
        combined_outputs = []

        enc_hiddens_proj = self.att_projection(enc_hiddens)
        Y = self.model_embeddings.target(target_padded)
        Y_ts = torch.split(Y, split_size_or_sections=1, dim=0)
        for Y_t in Y_ts:
            Y_t = torch.squeeze(Y_t, dim=0)
            dec_state, combined_output = self.step(Y_t, dec_state, enc_hiddens, enc_hiddens_proj, enc_masks)
            combined_outputs.append(combined_output)
        combined_outputs = torch.stack(combined_outputs, dim=0)

        ### END YOUR CODE

        return combined_outputs


    def step(self, Y_t: torch.Tensor,
            dec_state: Tuple[torch.Tensor, torch.Tensor],
            enc_hiddens: torch.Tensor,
            enc_hiddens_proj: torch.Tensor,
            enc_masks: torch.Tensor) -> Tuple[Tuple, torch.Tensor, torch.Tensor]:
        """ Compute one forward step of the LSTM decoder, including the attention computation.

        @param Y_t (Tensor): With shape (b, e). The input for the decoder, where b = batch size, e = embedding size.
        @param dec_state (tuple(Tensor, Tensor)): Tuple of tensors both with shape (b, h), where b = batch size, h = hidden size.
                First tensor is decoder's prev hidden state, second tensor is decoder's prev cell.
        @param enc_hiddens (Tensor): Encoder hidden states Tensor, with shape (b, src_len, h * 2), where b = batch size,
                                    src_len = maximum source length, h = hidden size.
        @param enc_hiddens_proj (Tensor): Encoder hidden states Tensor, projected from (h * 2) to h. Tensor is with shape (b, src_len, h),
                                    where b = batch size, src_len = maximum source length, h = hidden size.
        @param enc_masks (Tensor): Tensor of sentence masks shape (b, src_len),
                                    where b = batch size, src_len is maximum source length.

        @returns dec_state (tuple (Tensor, Tensor)): Tuple of tensors both shape (b, h), where b = batch size, h = hidden size.
                First tensor is decoder's new hidden state, second tensor is decoder's new cell.
        @returns combined_output (Tensor): Combined output Tensor at timestep t, shape (b, h), where b = batch size, h = hidden size.
        """
        batch_size = Y_t.size(0)
        fc_feats = torch.zeros([batch_size, 2048], device=self.device)
        att_feats = torch.zeros([batch_size, 2048], device=self.device)
        p_att_feats = torch.zeros([batch_size, self.hidden_size[1]], device=self.device)
        _, (dec_hidden, dec_cell) = self.decoder(Y_t, fc_feats, att_feats, p_att_feats,
            (dec_state[0].view(1, dec_state[0].size(0), dec_state[0].size(1)), dec_state[1].view(1, dec_state[1].size(0), dec_state[1].size(1)))) # xt, fc_feats, att_feats, p_att_feats, state
        dec_hidden = torch.squeeze(dec_hidden, dim=0)
        dec_cell = torch.squeeze(dec_cell, dim=0)
        dec_state = (dec_hidden, dec_cell)

        if self.use_attention:
            e_t = torch.bmm(torch.unsqueeze(dec_hidden, dim=1), enc_hiddens_proj.permute(0, 2, 1))
            e_t = torch.squeeze(e_t, dim=1)

            # Set e_t to -inf where enc_masks has 1
            if enc_masks is not None:
                e_t.data.masked_fill_(enc_masks.byte(), -float('inf'))

            alpha_t = F.softmax(e_t, dim=1)
            a_t = torch.squeeze(torch.bmm(torch.unsqueeze(alpha_t, dim=1), enc_hiddens), dim=1)
            U_t = torch.cat((a_t, dec_hidden), dim=1)
            V_t = self.combined_output_projection(U_t)
            O_t = torch.tanh(V_t)
            combined_output = self.dropout(O_t)

        else:
            combined_output = self.dropout(dec_hidden)
        return dec_state, combined_output

    def generate_sent_masks(self, enc_hiddens: torch.Tensor, source_lengths: List[int]) -> torch.Tensor:
        """ Generate sentence masks for encoder hidden states.

        @param enc_hiddens (Tensor): encodings of shape (b, src_len, 2*h), where b = batch size,
                                     src_len = max source length, h = hidden size.
        @param source_lengths (List[int]): List of actual lengths for each of the sentences in the batch.

        @returns enc_masks (Tensor): Tensor of sentence masks of shape (b, src_len),
                                    where src_len = max source length, h = hidden size.
        """
        enc_masks = torch.zeros(enc_hiddens.size(0), enc_hiddens.size(1), dtype=torch.float)
        for e_id, src_len in enumerate(source_lengths):
            enc_masks[e_id, src_len:] = 1
        return enc_masks.to(self.device)


    def beam_search(self, src_sent: List[str], beam_size: int=5, max_decoding_time_step: int=70) -> List[Hypothesis]:
        """ Given a single source sentence, perform beam search, yielding translations in the target language.
        @param src_sent (List[str]): a single source sentence (words)
        @param beam_size (int): beam size
        @param max_decoding_time_step (int): maximum number of time steps to unroll the decoding RNN
        @returns hypotheses (List[Hypothesis]): a list of hypothesis, each hypothesis has two fields:
                value: List[str]: the decoded target sentence, represented as a list of words
                score: float: the log-likelihood of the target sentence
        """
        src_sents_var = self.vocab.src.to_input_tensor([src_sent], self.device)

        src_encodings, dec_init_vec = self.encode(src_sents_var, [len(src_sent)])
        src_encodings_att_linear = self.att_projection(src_encodings)

        h_tm1 = dec_init_vec
        att_tm1 = torch.zeros(1, self.hidden_size[1], device=self.device)

        eos_id = self.vocab.tgt['<end>']

        hypotheses = [['<s>']]
        hyp_scores = torch.zeros(len(hypotheses), dtype=torch.float, device=self.device)
        completed_hypotheses = []

        t = 0
        while len(completed_hypotheses) < beam_size and t < max_decoding_time_step:
            t += 1
            hyp_num = len(hypotheses)

            exp_src_encodings = src_encodings.expand(hyp_num,
                                                     src_encodings.size(1),
                                                     src_encodings.size(2))

            exp_src_encodings_att_linear = src_encodings_att_linear.expand(hyp_num,
                                                                           src_encodings_att_linear.size(1),
                                                                           src_encodings_att_linear.size(2))

            y_tm1 = torch.tensor([self.vocab.tgt[hyp[-1]] for hyp in hypotheses], dtype=torch.long, device=self.device)
            y_t_embed = self.model_embeddings.target(y_tm1)

            (h_t, cell_t), att_t  = self.step(y_t_embed, h_tm1,
                                                      exp_src_encodings, exp_src_encodings_att_linear, enc_masks=None)

            # log probabilities over target words
            log_p_t = F.log_softmax(self.target_vocab_projection(att_t), dim=-1)

            live_hyp_num = beam_size - len(completed_hypotheses)
            contiuating_hyp_scores = (hyp_scores.unsqueeze(1).expand_as(log_p_t) + log_p_t).view(-1)
            top_cand_hyp_scores, top_cand_hyp_pos = torch.topk(contiuating_hyp_scores, k=live_hyp_num)

            prev_hyp_ids = top_cand_hyp_pos / len(self.vocab.tgt)
            hyp_word_ids = top_cand_hyp_pos % len(self.vocab.tgt)

            new_hypotheses = []
            live_hyp_ids = []
            new_hyp_scores = []

            for prev_hyp_id, hyp_word_id, cand_new_hyp_score in zip(prev_hyp_ids, hyp_word_ids, top_cand_hyp_scores):
                prev_hyp_id = prev_hyp_id.item()
                hyp_word_id = hyp_word_id.item()
                cand_new_hyp_score = cand_new_hyp_score.item()

                hyp_word = self.vocab.tgt.id2word[hyp_word_id]
                new_hyp_sent = hypotheses[prev_hyp_id] + [hyp_word]
                if hyp_word == '<end>':
                    completed_hypotheses.append(Hypothesis(value=new_hyp_sent[1:-1],
                                                           score=cand_new_hyp_score))
                else:
                    new_hypotheses.append(new_hyp_sent)
                    live_hyp_ids.append(prev_hyp_id)
                    new_hyp_scores.append(cand_new_hyp_score)

            if len(completed_hypotheses) == beam_size:
                break

            live_hyp_ids = torch.tensor(live_hyp_ids, dtype=torch.long, device=self.device)
            h_tm1 = (h_t[live_hyp_ids], cell_t[live_hyp_ids])
            att_tm1 = att_t[live_hyp_ids]

            hypotheses = new_hypotheses
            hyp_scores = torch.tensor(new_hyp_scores, dtype=torch.float, device=self.device)

        if len(completed_hypotheses) == 0:
            completed_hypotheses.append(Hypothesis(value=hypotheses[0][1:],
                                                   score=hyp_scores[0].item()))

        completed_hypotheses.sort(key=lambda hyp: hyp.score, reverse=True)

        return completed_hypotheses

    @property
    def device(self) -> torch.device:
        """ Determine which device to place the Tensors upon, CPU or GPU.
        """
        return self.model_embeddings.source.weight.device

    @staticmethod
    def load(model_path: str):
        """ Load the model from a file.
        @param model_path (str): path to model
        """
        params = torch.load(model_path, map_location=lambda storage, loc: storage)
        args = params['args']
        model = NMT(vocab=params['vocab'], **args)
        model.load_state_dict(params['state_dict'])

        return model

    def save(self, path: str):
        """ Save the model to a file.
        @param path (str): path to the model
        """
        print('save model parameters to [%s]' % path, file=sys.stderr)

        params = {
            'args': dict(embed_size=self.model_embeddings.embed_size, hidden_size=self.hidden_size, dropout_rate=self.dropout_rate),
            'vocab': self.vocab,
            'state_dict': self.state_dict()
        }

        if not self.use_attention:
            params['args'].update(use_attention=False)

        torch.save(params, path)

    def save_enc_dec(self, enc_path: str, dec_path: str):
        """ Save the encoder and decoder to a file.
        @param enc_path (str): path to the encoder params
        @param dec_path (str): path to the decoder params
        """
        print('save encoder parameters to [%s]' % enc_path, file=sys.stderr)
        print('save decoder parameters to [%s]' % dec_path, file=sys.stderr)

        enc_params = {
            'args': dict(ntoken=len(self.vocab.src), ninput=self.model_embeddings.embed_size[0], nhidden=self.hidden_size[0]*2),
            'vocab': {'word2id': self.vocab.src.word2id, 'id2word': self.vocab.src.id2word},
            'encoder': self.encoder.state_dict(),
            'embedding': self.model_embeddings.source.state_dict()
        }

        torch.save(enc_params, enc_path)

        dec_params = {
            'vocab': {'word2id': self.vocab.tgt.word2id, 'id2word': self.vocab.tgt.id2word},
            'decoder': self.decoder.state_dict(),
            'logit': self.target_vocab_projection .state_dict(),
            'embedding': self.model_embeddings.target.state_dict()
        }

        torch.save(dec_params, dec_path)
'''
To reconstruct RNN_ENCODER, we need to store:
ntoken: len(vocab.src) OR NMT.model_embeddings.source weight size(0)
ninput: embed_size[0], =300
nhidden: hidden_size[0], =128
RNN_ENCODER.encoder weight: NMT.model_embeddings.source weight
vocab.src.word2id AND vocab.src.id2word

To reconstruct Att2inModel, we need to store:
ntoken: len(vocab.src) OR NMT.model_embeddings.source weight size(0)
input_encoding_size: embed_size[1], =512
rnn_size: hidden_size[1], =512
Attn2inModel.core weight: NMT.decoder weight
Attn2inModel.logit weight: NMT.target_vocab_projection weight
Attn2inModel.embed weight: NMT.model_embeddings.target weight
vocab.tgt.word2id AND vocab.tgt.id2word
'''
