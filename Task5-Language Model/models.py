import torch
import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size=128, dropout_rate=0.5, layer_num=1):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.layer_num = layer_num
        if layer_num == 1:
            self.lstm = nn.LSTM(input_size, hidden_size, layer_num, batch_first=True)
        else:
            self.lstm = nn.LSTM(input_size, hidden_size, layer_num, dropout=dropout_rate, batch_first=True)

        self.init_weights()

    def init_weights(self):
        for p in self.lstm.parameters():
            if p.dim() > 1:
                nn.init.xavier_normal_(p)
            else:
                p.data.zero_()

    def init_hidden(self, batch_size):
        weight = next(self.parameters())
        return (weight.new_zeros(self.layer_num, batch_size, self.hidden_size),
                weight.new_zeros(self.layer_num, batch_size, self.hidden_size))

    def forward(self, x, lens, hidden):
        '''
        :param x: (batch, seq_len, input_size)
        :param lens: (batch, ), in descending order
        :param hidden: tuple(h,c), each has shape (num_layer, batch, hidden_size)
        :return: output: (batch, seq_len, hidden_size)
                 tuple(h,c): each has shape (num_layer, batch, hidden_size)
        '''
        packed_x = nn.utils.rnn.pack_padded_sequence(x, lens, batch_first=True)
        packed_output, (h, c) = self.lstm(packed_x, hidden)
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        return output, (h, c)


class LSTM_LM(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size=128, dropout_rate=0.2, layer_num=1, max_seq_len=128):
        super(LSTM_LM, self).__init__()
        self.hidden_size = hidden_size
        self.layer_num = layer_num
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = LSTM(embed_size, hidden_size, dropout_rate, layer_num)
        self.project = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_normal_(self.embed.weight)
        nn.init.xavier_normal_(self.project.weight)

    def forward(self, x, lens, hidden):
        '''
        :param x: (batch, seq_len, input_size)
        :param lens: (batch, ), in descending order
        :param hidden: tuple(h,c), each has shape (num_layer, batch, hidden_size)
        :return: output: (batch, seq_len, hidden_size)
                 tuple(h,c): each has shape (num_layer, batch, hidden_size)
        '''
        embed = self.embed(x)
        hidden, (h, c) = self.lstm(self.dropout(embed), lens, hidden)  # (batch, seq_len, hidden_size)
        out = self.project(self.dropout(hidden))  # (batch, seq_len, vocab_size)
        return out, (h, c)
