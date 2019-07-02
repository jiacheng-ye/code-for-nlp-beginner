import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from tqdm import tqdm, trange
from torch.optim import Adam
from tensorboardX import SummaryWriter
import pandas as pd
from collections import OrderedDict, Counter
import os, re

from models import RNN, CNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_epochs = 10
batch_size = 512
learning_rate = 0.001
max_seq_length = 48
num_classes = 5
dropout_rate = 0.1
data_path = "data"
clip = 5
embed_size = 200
use_pretrained_embedding = True
embed_path = '/home/yjc/embeddings/glove.6B.200d.txt'
freeze = False
use_rnn = True  # set True to use RNN, otherwise CNN.

# parameters for RNN
hidden_size = 256
num_layers = 1
bidirectional = True

# parameters for CNN
num_filters = 100
kernel_sizes = [2, 3, 4]  # n-gram


class Tokenizer():
    def __init__(self, datas, vocabulary=None):
        self.data_len = len(datas)
        if vocabulary:
            self.tok2id = vocabulary
        else:
            self.tok2id = self.build_dict(self.tokenize(' '.join(datas)), offset=4)

        self.tok2id['[PAD]'] = 0
        self.tok2id['[UNK]'] = 1

        self.id2tok = OrderedDict([(id, tok) for tok, id in self.tok2id.items()])

    def build_dict(self, words, offset=0, max_words=None, max_df=None):
        cnt = Counter(words)
        if max_df:
            words = dict(filter(lambda x: x[1] < max_df * self.data_len, cnt.items()))
        words = sorted(words.items(), key=lambda x: x[1], reverse=True)
        if max_words:
            words = words[:max_words]  # [(word, count)]
        return {word: offset + i for i, (word, _) in enumerate(words)}

    @staticmethod
    def tokenize(text):
        # return re.compile(r'\b\w\w+\b').findall(text)
        return text.split(" ")

    def convert_ids_to_tokens(self, ids):
        return [self.id2tok[i] for i in ids]

    def convert_tokens_to_ids(self, tokens):
        ids = []
        for token in tokens:
            if not self.tok2id.get(token):
                ids.append(self.tok2id["[UNK]"])
            else:
                ids.append(self.tok2id[token])
        return ids


class MyDataset(Dataset):
    def __init__(self, datas, max_seq_length, tokenizer):
        self.datas = datas
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, item):
        toks = self.tokenizer.tokenize(self.datas['Phrase'][item].lower())
        cur_example = InputExample(uid=item, toks=toks, labels=self.datas['Sentiment'][item])
        cur_features = convert_example_to_features(cur_example, self.max_seq_length, self.tokenizer)
        cur_tensors = (
            torch.LongTensor(cur_features.input_ids),
            torch.tensor(cur_features.label_ids)
        )
        return cur_tensors


class InputExample(object):
    def __init__(self, uid, toks, labels=None):
        self.toks = toks
        self.labels = labels
        self.uid = uid


class InputFeatures(object):
    def __init__(self, eid, input_ids, label_ids):
        self.input_ids = input_ids
        self.label_ids = label_ids
        self.eid = eid


def convert_example_to_features(example, max_seq_length, tokenizer):
    """Convert a raw sample (pair of sentences as tokenized strings) into a proper training sample as ids."""
    input_ids = tokenizer.convert_tokens_to_ids(example.toks)[:max_seq_length]
    # pad up to the sequence length.
    while len(input_ids) < max_seq_length:
        input_ids.append(0)

    if example.uid == 0:
        print("*** Example ***")
        print("uid: %s" % example.uid)
        print("tokens: %s" % " ".join([str(x) for x in example.toks]))
        print("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        print("label: %s " % example.labels)

    features = InputFeatures(input_ids=input_ids,
                             eid=example.uid,
                             label_ids=example.labels,
                             )
    return features


def load_word_vector_mapping(file):
    ret = OrderedDict()
    with open(file, encoding="utf8") as f:
        for line in f.readlines():
            word, vec = line.split(" ", 1)
            ret[word] = list(map(float, vec.split()))
    return ret


if __name__ == "__main__":
    # load data
    datas = pd.read_csv(os.path.join(data_path, 'train.tsv'), sep='\t')
    train_num = int(len(datas) * 0.8)
    train_data = datas[:train_num]
    dev_data = datas[train_num:]
    dev_data.index = range(len(dev_data))
    test_data = pd.read_csv(os.path.join(data_path, 'test.tsv'), sep='\t')
    test_data["Sentiment"] = [0 for _ in range(len(test_data))]

    # build model
    if use_pretrained_embedding:
        word2vec = load_word_vector_mapping(embed_path)
        words = list(word2vec.keys())
        tok2id = dict([(x, i) for i, x in enumerate(words, 2)])
        tokenizer = Tokenizer(train_data['Phrase'], tok2id)
        vecs = list(word2vec.values())
        assert embed_size == len(
            vecs[0]), "Parameter embed_size must be equal to the embed_size of the pretrained embeddings."
        vecs.insert(0, [.0 for _ in range(embed_size)])  # PAD
        vecs.insert(1, [.0 for _ in range(embed_size)])  # UNK

        vocab_size = len(tokenizer.tok2id)
        if use_rnn:
            model = RNN(vocab_size, embed_size, hidden_size, num_layers, num_classes, bidirectional, dropout_rate).to(
                device)
        else:
            model = CNN(vocab_size, embed_size, num_classes, num_filters, kernel_sizes, dropout_rate).to(device)
        weights = torch.tensor(vecs)
        model.embed.from_pretrained(weights, freeze=freeze)
    else:
        tokenizer = Tokenizer(train_data['Phrase'])
        vocab_size = len(tokenizer.tok2id)
        if use_rnn:
            model = RNN(vocab_size, embed_size, hidden_size, num_layers, num_classes, bidirectional, dropout_rate).to(
                device)
        else:
            model = CNN(vocab_size, embed_size, num_classes, num_filters, kernel_sizes, dropout_rate).to(device)

    # build datasets
    train_dataset = MyDataset(train_data, max_seq_length, tokenizer)
    dev_dataset = MyDataset(dev_data, max_seq_length, tokenizer)
    test_dataset = MyDataset(test_data, max_seq_length, tokenizer)

    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
    dev_dataloader = DataLoader(dev_dataset, shuffle=True, batch_size=batch_size)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

    optimizer = Adam(model.parameters(), lr=learning_rate)
    loss_func = nn.CrossEntropyLoss()
    writer = SummaryWriter('logs', comment="rnn")
    for epoch in trange(train_epochs, desc="Epoch"):
        model.train()
        ep_loss = 0
        for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
            inputs, labels = tuple(t.to(device) for t in batch)
            lens = (inputs != 0).sum(-1)  # 0 is the id of [PAD].
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
            for batch in dev_dataloader:
                inputs, labels = tuple(t.to(device) for t in batch)
                lens = (inputs != 0).sum(-1)  # 0 is the id of [PAD].
                outputs = model(inputs, lens)
                corr_num += (outputs.argmax(1) == labels).sum().item()
                err_num += (outputs.argmax(1) != labels).sum().item()
            tqdm.write("Epoch %d, Accuracy %.3f" % (epoch, corr_num / (corr_num + err_num)))

    # predicting
    model.eval()
    with torch.no_grad():
        predicts = []
        for batch in test_dataloader:
            inputs, labels = tuple(t.to(device) for t in batch)
            lens = (inputs != 0).sum(-1)  # 0 is the id of [PAD].
            outputs = model(inputs, lens)
            predicts.extend(outputs.argmax(1).cpu().numpy())
        test_data["Sentiment"] = predicts
        test_data[['PhraseId', 'Sentiment']].set_index('PhraseId').to_csv('result.csv')

# 0.597
