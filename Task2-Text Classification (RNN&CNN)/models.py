import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from rnn import RNNModel

class RNN(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers, num_classes, bidirectional=True,
                 dropout_rate=0.3):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embed = nn.Embedding(vocab_size, embed_size)
        # self.rnn = nn.RNN(embed_size, hidden_size, num_layers, batch_first=True, bidirectional=bidirectional)
        self.rnn = RNNModel(embed_size, hidden_size, num_layers, batch_first=True, bidirectional=bidirectional)
        self.bidirectional = bidirectional
        if not bidirectional:
            self.fc = nn.Linear(hidden_size, num_classes)
        else:
            self.fc = nn.Linear(hidden_size * 2, num_classes)
        self.dropout = nn.Dropout(dropout_rate)

        self.init_weights()

    def init_weights(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(self, x, lens):
        embeddings = self.embed(x)
        output, _ = self.rnn(embeddings)
        # get the output specified by length
        real_output = output[range(len(lens)), lens - 1]  # (batch_size, seq_length, hidden_size*num_directions)
        out = self.fc(self.dropout(real_output))
        return out


class CNN(nn.Module):
    def __init__(self, vocab_size, embed_size, num_classes, num_filters=100, kernel_sizes=[3, 4, 5], dropout_rate=0.3):
        super(CNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.convs = nn.ModuleList([
            nn.Conv2d(1, num_filters, (k, embed_size), padding=(k - 1, 0))
            for k in kernel_sizes])
        self.fc = nn.Linear(len(kernel_sizes) * num_filters, num_classes)
        self.dropout = nn.Dropout(dropout_rate)

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x).squeeze(3))  # (batch_size, num_filter, conv_seq_length)
        x_max = F.max_pool1d(x, x.size(2)).squeeze(2)  # (batch_size, num_filter)
        return x_max

    def forward(self, x, lens):
        embed = self.embed(x).unsqueeze(1)  # (batch_size, 1, seq_length, embedding_dim)

        conv_results = [self.conv_and_pool(embed, conv) for conv in self.convs]

        out = torch.cat(conv_results, 1)  # (batch_size, num_filter * len(kernel_sizes))
        return self.fc(self.dropout(out))
