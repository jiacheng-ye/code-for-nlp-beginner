import torch
import torch.nn as nn
from crf import CRF
import torch.nn.functional as F
import math

class CharCNN(nn.Module):
    def __init__(self, num_filters, kernel_sizes, padding):
        super(CharCNN, self).__init__()
        self.conv = nn.Conv2d(1, num_filters, kernel_sizes, padding=padding)

    def forward(self, x):
        '''
        :param x: (batch * seq_len, 1, max_word_len, char_embed_size)
        :return: (batch * seq_len, num_filters)
        '''
        x = self.conv(x).squeeze(-1)  # (batch * seq_len, num_filters, max_word_len)
        x_max = F.max_pool1d(x, x.size(2)).squeeze(-1)  # (batch * seq_len, num_filters)
        return x_max

class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=128, dropout_rate=0.1, layer_num=1):
        super(BiLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.layer_num = layer_num
        if layer_num == 1:
            self.bilstm = nn.LSTM(input_size, hidden_size // 2, layer_num, batch_first=True, bidirectional=True)

        else:
            self.bilstm = nn.LSTM(input_size, hidden_size // 2, layer_num, batch_first=True, dropout=dropout_rate,
                                  bidirectional=True)
        self.init_weights()

    def init_weights(self):
        for name, p in self.bilstm._parameters.items():
            if p.dim() > 1:
                bias = math.sqrt(6 / (p.size(0) / 4 + p.size(1)))
                nn.init.uniform_(p, -bias, bias)
            else:
                p.data.zero_()
                # This is the range of indices for our forget gates for each LSTM cell
                p.data[self.hidden_size // 2: self.hidden_size] = 1

    def forward(self, x, lens):
        '''
        :param x: (batch, seq_len, input_size)
        :param lens: (batch, )
        :return: (batch, seq_len, hidden_size)
        '''
        ordered_lens, index = lens.sort(descending=True)
        ordered_x = x[index]
        packed_x = nn.utils.rnn.pack_padded_sequence(ordered_x, ordered_lens, batch_first=True)
        packed_output, _ = self.bilstm(packed_x)
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        recover_index = index.argsort()
        output = output[recover_index]
        return output


class SoftmaxDecoder(nn.Module):
    def __init__(self, label_size, input_dim):
        super(SoftmaxDecoder, self).__init__()
        self.input_dim = input_dim
        self.label_size = label_size
        self.linear = torch.nn.Linear(input_dim, label_size)
        self.init_weights()

    def init_weights(self):
        bias = math.sqrt(6 / (self.linear.weight.size(0) + self.linear.weight.size(1)))
        nn.init.uniform_(self.linear.weight, -bias, bias)

    def forward_model(self, inputs):
        batch_size, seq_len, input_dim = inputs.size()
        output = inputs.contiguous().view(-1, self.input_dim)
        output = self.linear(output)
        output = output.view(batch_size, seq_len, self.label_size)
        return output

    def forward(self, inputs, lens, label_ids=None):
        logits = self.forward_model(inputs)
        p = torch.nn.functional.softmax(logits, -1)  # (batch_size, max_seq_len, num_labels)
        predict_mask = (torch.arange(inputs.size(1)).expand(len(lens), inputs.size(1))).to(lens.device) < lens.unsqueeze(1)
        if label_ids is not None:
            # cross entropy loss
            p = torch.nn.functional.softmax(logits, -1)  # (batch_size, max_seq_len, num_labels)
            one_hot_labels = torch.eye(self.label_size)[label_ids].type_as(p)
            losses = -torch.log(torch.sum(one_hot_labels * p, -1))  # (batch_size, max_seq_len)
            masked_losses = torch.masked_select(losses, predict_mask)  # (batch_sum_real_len)
            return masked_losses.sum()
        else:
            return torch.argmax(logits, -1), p

class CRFDecoder(nn.Module):
    def __init__(self, label_size, input_dim):
        super(CRFDecoder, self).__init__()
        self.input_dim = input_dim
        self.linear = nn.Linear(in_features=input_dim,
                                out_features=label_size)
        self.crf = CRF(label_size + 2)
        self.label_size = label_size

        self.init_weights()

    def init_weights(self):
        bias = math.sqrt(6 / (self.linear.weight.size(0) + self.linear.weight.size(1)))
        nn.init.uniform_(self.linear.weight, -bias, bias)

    def forward_model(self, inputs):
        batch_size, seq_len, input_dim = inputs.size()
        output = inputs.contiguous().view(-1, self.input_dim)
        output = self.linear(output)
        output = output.view(batch_size, seq_len, self.label_size)
        return output

    def forward(self, inputs, lens, labels=None):
        '''
        :param inputs:(batch_size, max_seq_len, input_dim)
        :param predict_mask:(batch_size, max_seq_len)
        :param labels:(batch_size, max_seq_len)
        :return: if labels is None, return preds(batch_size, max_seq_len) and p(batch_size, max_seq_len, num_labels);
                 else return loss (scalar).
        '''
        logits = self.forward_model(inputs)  # (batch_size, max_seq_len, num_labels)
        p = torch.nn.functional.softmax(logits, -1)  # (batch_size, max_seq_len, num_labels)
        logits = self.crf.pad_logits(logits)
        predict_mask = (torch.arange(inputs.size(1)).expand(len(lens), inputs.size(1))).to(lens.device) < lens.unsqueeze(1)
        if labels is None:
            _, preds = self.crf.viterbi_decode(logits, predict_mask)
            return preds, p
        return self.neg_log_likehood(logits, predict_mask, labels)

    def neg_log_likehood(self, logits, predict_mask, labels):
        norm_score = self.crf.calc_norm_score(logits, predict_mask)
        gold_score = self.crf.calc_gold_score(logits, labels, predict_mask)
        loglik = gold_score - norm_score
        return -loglik.sum()


class NER_Model(nn.Module):
    def __init__(self, word_embed, char_embed,
                 num_labels, hidden_size, dropout_rate=(0.33, 0.5, (0.33, 0.5)),
                 lstm_layer_num=1, kernel_step=3, char_out_size=100, use_char=False,
                 freeze=False, use_crf=True):
        super(NER_Model, self).__init__()
        self.word_embed = nn.Embedding.from_pretrained(word_embed, freeze)
        self.word_embed_size = word_embed.size(-1)
        self.use_char = use_char
        if use_char:
            self.char_embed = nn.Embedding.from_pretrained(char_embed, freeze)
            self.char_embed_size = char_embed.size(-1)
            self.charcnn = CharCNN(char_out_size, (kernel_step, self.char_embed_size), (2, 0))
            self.bilstm = BiLSTM(char_out_size + self.word_embed_size, hidden_size, dropout_rate[2][1], lstm_layer_num)
        else:
            self.bilstm = BiLSTM(self.word_embed_size, hidden_size, dropout_rate[2][1], lstm_layer_num)

        self.embed_dropout = nn.Dropout(dropout_rate[0])
        self.out_dropout = nn.Dropout(dropout_rate[1])
        self.rnn_in_dropout = nn.Dropout(dropout_rate[2][0])

        if use_crf:
            self.decoder = CRFDecoder(num_labels, hidden_size)
        else:
            self.decoder = SoftmaxDecoder(num_labels, hidden_size)


    def forward(self, word_ids, char_ids, lens, label_ids=None):
        '''

        :param word_ids: (batch_size, max_seq_len)
        :param char_ids: (batch_size, max_seq_len, max_word_len)
        :param predict_mask: (batch_size, max_seq_len)
        :param label_ids: (batch_size, max_seq_len, max_word_len)
        :return: if labels is None, return preds(batch_size, max_seq_len) and p(batch_size, max_seq_len, num_labels);
                 else return loss (scalar).
        '''
        word_embed = self.word_embed(word_ids)
        if self.char_embed:
            # reshape char_embed and apply to CNN
            char_embed = self.char_embed(char_ids).reshape(-1, char_ids.size(-1), self.char_embed_size).unsqueeze(1)
            char_embed = self.embed_dropout(
                char_embed)  # a dropout layer applied before character embeddings are input to CNN.
            char_embed = self.charcnn(char_embed)
            char_embed = char_embed.reshape(char_ids.size(0), char_ids.size(1), -1)
            embed = torch.cat([word_embed, char_embed], -1)
        else:
            embed = word_embed
        x = self.rnn_in_dropout(embed)
        hidden = self.bilstm(x, lens)  # (batch_size, max_seq_len, hidden_size)
        hidden = self.out_dropout(hidden)
        return self.decoder(hidden, lens, label_ids)
