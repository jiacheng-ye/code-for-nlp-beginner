import torch
import torch.nn as nn
import torch.nn.functional as F


class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=128, dropout_rate=0.1, layer_num=1):
        super(BiLSTM, self).__init__()
        self.hidden_size = hidden_size
        if layer_num == 1:
            self.bilstm = nn.LSTM(input_size, hidden_size // 2, layer_num, batch_first=True, bidirectional=True)

        else:
            self.bilstm = nn.LSTM(input_size, hidden_size // 2, layer_num, batch_first=True, dropout=dropout_rate,
                                  bidirectional=True)
        self.init_weights()

    def init_weights(self):
        for p in self.bilstm.parameters():
            if p.dim() > 1:
                nn.init.normal_(p)
                p.data.mul_(0.01)
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
        recover_output = output[recover_index]
        return recover_output


class ESIM(nn.Module):
    def __init__(self, vocab_size, num_labels, embed_size, hidden_size, dropout_rate=0.1, layer_num=1,
                 pretrained_embed=None, freeze=False):
        super(ESIM, self).__init__()
        self.pretrained_embed = pretrained_embed
        if pretrained_embed is not None:
            self.embed = nn.Embedding.from_pretrained(pretrained_embed, freeze)
        else:
            self.embed = nn.Embedding(vocab_size, embed_size)
        self.bilstm1 = BiLSTM(embed_size, hidden_size, dropout_rate, layer_num)
        self.bilstm2 = BiLSTM(hidden_size, hidden_size, dropout_rate, layer_num)
        self.fc1 = nn.Linear(4 * hidden_size, hidden_size)
        self.fc2 = nn.Linear(4 * hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_labels)
        self.dropout = nn.Dropout(dropout_rate)

        self.init_weight()

    def init_weight(self):
        if self.pretrained_embed is None:
            nn.init.normal_(self.embed.weight)
            self.embed.weight.data.mul_(0.01)
        nn.init.normal_(self.fc1.weight)
        self.fc1.weight.data.mul_(0.01)
        nn.init.normal_(self.fc2.weight)
        self.fc2.weight.data.mul_(0.01)
        nn.init.normal_(self.fc3.weight)
        self.fc3.weight.data.mul_(0.01)


    def soft_align_attention(self, x1, x1_lens, x2, x2_lens):
        '''
        local inference modeling
        :param x1: (batch, seq1_len, hidden_size)
        :param x1_lens: (batch, )
        :param x2: (batch, seq2_len, hidden_size)
        :param x2_lens: (batch, )
        :return: x1_align (batch, seq1_len, hidden_size)
                 x2_align (batch, seq2_len, hidden_size)
        '''
        seq1_len = x1.size(1)
        seq2_len = x2.size(1)
        batch_size = x1.size(0)

        attention = torch.matmul(x1, x2.transpose(1, 2))  # (batch, seq1_len, seq2_len)
        mask1 = torch.arange(seq1_len).expand(batch_size, seq1_len).to(x1.device) >= x1_lens.unsqueeze(
            1)  # (batch, seq1_len), 1 means <pad>
        mask2 = torch.arange(seq2_len).expand(batch_size, seq2_len).to(x1.device) >= x2_lens.unsqueeze(
            1)  # (batch, seq2_len)
        mask1 = mask1.float().masked_fill_(mask1, float('-inf'))
        mask2 = mask2.float().masked_fill_(mask2, float('-inf'))
        weight2 = F.softmax(attention + mask2.unsqueeze(1), dim=-1)  # (batch, seq1_len, seq2_len)
        x1_align = torch.matmul(weight2, x2)  # (batch, seq1_len, hidden_size)
        weight1 = F.softmax(attention.transpose(1, 2) + mask1.unsqueeze(1), dim=-1)  # (batch, seq2_len, seq1_len)
        x2_align = torch.matmul(weight1, x1)  # (batch, seq2_len, hidden_size)
        return x1_align, x2_align

    def composition(self, x, lens):
        x = F.relu(self.fc1(x))
        x_compose = self.bilstm2(self.dropout(x), lens)  # (batch, seq_len, hidden_size)
        p1 = F.avg_pool1d(x_compose.transpose(1, 2), x.size(1)).squeeze(-1)  # (batch, hidden_size)
        p2 = F.max_pool1d(x_compose.transpose(1, 2), x.size(1)).squeeze(-1)  # (batch, hidden_size)
        return torch.cat([p1, p2], 1)  # (batch, hidden_size*2)

    def forward(self, x1, x1_lens, x2, x2_lens):
        '''
        :param x1: (batch, seq1_len)
        :param x1_lens: (batch,)
        :param x2: (batch, seq2_len)
        :param x2_lens: (batch,)
        :return: (batch, num_class)
        '''
        # Input encoding
        embed1 = self.embed(x1)  # (batch, seq1_len, embed_size)
        embed2 = self.embed(x2)  # (batch, seq2_len, embed_size)
        new_embed1 = self.bilstm1(self.dropout(embed1), x1_lens)  # (batch, seq1_len, hidden_size)
        new_embed2 = self.bilstm1(self.dropout(embed2), x2_lens)  # (batch, seq2_len, hidden_size)

        # Local inference collected over sequence
        x1_align, x2_align = self.soft_align_attention(new_embed1, x1_lens, new_embed2, x2_lens)

        # Enhancement of local inference information
        x1_combined = torch.cat([new_embed1, x1_align, new_embed1 - x1_align, new_embed1 * x1_align],
                                dim=-1)  # (batch, seq1_len, 4*hidden_size)
        x2_combined = torch.cat([new_embed2, x2_align, new_embed2 - x2_align, new_embed2 * x2_align],
                                dim=-1)  # (batch, seq2_len, 4*hidden_size)

        # Inference composition
        x1_composed = self.composition(x1_combined, x1_lens)  # (batch, 2*hidden_size), v=[v_avg; v_max]
        x2_composed = self.composition(x2_combined, x2_lens)  # (batch, 2*hidden_size)
        composed = torch.cat([x1_composed, x2_composed], -1)  # (batch, 4*hidden_size)

        # MLP classifier
        out = self.fc3(self.dropout(torch.tanh(self.fc2(self.dropout(composed)))))
        return out
