import torch
import torch.nn as nn
from tqdm import tqdm, trange
from torch.optim import Adam
from tensorboardX import SummaryWriter
import pandas as pd
import os
from torchtext import data
from torchtext.data import Iterator, BucketIterator
from torchtext.vocab import Vectors

from models import RNN, CNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_epochs = 5
batch_size = 512
learning_rate = 0.001
max_seq_length = 48
num_classes = 5
dropout_rate = 0.1
data_path = "data"
clip = 5

# embedding
embed_size = 200
# vectors = None
vectors = Vectors('glove.6B.200d.txt', '/home/yjc/embeddings')
freeze = False

use_rnn = True
# parameters for RNN
hidden_size = 256
num_layers = 1
bidirectional = True

# parameters for CNN
num_filters = 200
kernel_sizes = [2, 3, 4]  # n-gram


def load_iters(batch_size=32, device="cpu", data_path='data', vectors=None):
    TEXT = data.Field(lower=True, batch_first=True, include_lengths=True)
    LABEL = data.LabelField(batch_first=True)
    train_fields = [(None, None), (None, None), ('text', TEXT), ('label', LABEL)]
    test_fields = [(None, None), (None, None), ('text', TEXT)]

    train_data = data.TabularDataset.splits(
        path=data_path,
        train='train.tsv',
        format='tsv',
        fields=train_fields,
        skip_header=True
    )[0]  # return is a tuple.

    test_data = data.TabularDataset.splits(
        path='data',
        train='test.tsv',
        format='tsv',
        fields=test_fields,
        skip_header=True
    )[0]

    TEXT.build_vocab(train_data.text, vectors=vectors)
    LABEL.build_vocab(train_data.label)
    train_data, dev_data = train_data.split([0.8, 0.2])

    train_iter, dev_iter = BucketIterator.splits(
        (train_data, dev_data),
        batch_sizes=(batch_size, batch_size),
        device=device,
        sort_key=lambda x: len(x.text),
        sort_within_batch=True,
        repeat=False,
        shuffle=True
    )

    test_iter = Iterator(test_data, batch_size=batch_size, device=device, sort=False, sort_within_batch=False,
                         repeat=False, shuffle=False)
    return train_iter, dev_iter, test_iter, TEXT, LABEL


if __name__ == "__main__":
    train_iter, dev_iter, test_iter, TEXT, LABEL = load_iters(batch_size, device, data_path, vectors)
    vocab_size = len(TEXT.vocab.itos)
    # build model
    if use_rnn:
        model = RNN(vocab_size, embed_size, hidden_size, num_layers, num_classes, bidirectional, dropout_rate)
    else:
        model = CNN(vocab_size, embed_size, num_classes, num_filters, kernel_sizes, dropout_rate)
    if vectors is not None:
        model.embed.from_pretrained(TEXT.vocab.vectors, freeze=freeze)
    model.to(device)

    optimizer = Adam(model.parameters(), lr=learning_rate)
    loss_func = nn.CrossEntropyLoss()
    writer = SummaryWriter('logs', comment="rnn")
    for epoch in trange(train_epochs, desc="Epoch"):
        model.train()
        ep_loss = 0
        for step, batch in enumerate(tqdm(train_iter, desc="Iteration")):
            (inputs, lens), labels = batch.text, batch.label
            outputs = model(inputs, lens)
            loss = loss_func(outputs, labels)
            ep_loss += loss.item()

            model.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()

            writer.add_scalar('Train_Loss', loss, epoch)
            if step % 100 == 0:
                tqdm.write("Epoch %d, Step %d, Loss %.2f" % (epoch, step, loss.item()))

        # evaluating
        model.eval()
        with torch.no_grad():
            corr_num = 0
            err_num = 0
            for batch in dev_iter:
                (inputs, lens), labels = batch.text, batch.label
                outputs = model(inputs, lens)
                corr_num += (outputs.argmax(1) == labels).sum().item()
                err_num += (outputs.argmax(1) != labels).sum().item()
            tqdm.write("Epoch %d, Accuracy %.3f" % (epoch, corr_num / (corr_num + err_num)))

    # predicting
    model.eval()
    with torch.no_grad():
        predicts = []
        for batch in test_iter:
            inputs, lens = batch.text
            outputs = model(inputs, lens)
            predicts.extend(outputs.argmax(1).cpu().numpy())
        test_data = pd.read_csv(os.path.join(data_path, 'test.tsv'), sep='\t')
        test_data["Sentiment"] = predicts
        test_data[['PhraseId', 'Sentiment']].set_index('PhraseId').to_csv('result.csv')
