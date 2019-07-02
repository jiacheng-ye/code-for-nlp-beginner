from torchtext import data
from torchtext.data import Iterator, BucketIterator
import os
import re
import math

def read_data(input_file):
    """Reads a BIO data."""
    with open(input_file) as f:
        lines = []
        words = []
        labels = []
        for line in f:
            contends = line.strip()
            if contends.startswith("-DOCSTART-"):
                continue
            if len(contends) == 0:
                if len(words) == 0:
                    continue
                lines.append([words, [list(word) for word in words], labels])
                words = []
                labels = []
                continue
            tokens = line.strip().split(' ')
            assert (len(tokens) == 4)
            word = tokens[0]
            label = tokens[-1]
            words.append(word)
            labels.append(label)
        return lines


class ConllDataset(data.Dataset):

    def __init__(self, word_field, char_field, label_field, datafile, **kwargs):
        fields = [("word", word_field), ("char", char_field), ("label", label_field)]
        datas = read_data(datafile)
        examples = []
        for word, char, label in datas:
            examples.append(data.Example.fromlist([word, char, label], fields))
        super(ConllDataset, self).__init__(examples, fields, **kwargs)


def unk_init(x):
    dim = x.size(-1)
    bias = math.sqrt(3.0 / dim)
    x.uniform_(-bias, bias)
    return x

def load_iters(batch_size=32, device="cpu", data_path='data', vectors=None, word2lower=True):
    zero_char_in_word = lambda ex: [re.sub('\d', '0', w) for w in ex]
    zero_char = lambda w: [re.sub('\d', '0', c) for c in w]
    WORD_TEXT = data.Field(lower=word2lower, batch_first=True,
                           preprocessing=zero_char_in_word)
    CHAR_NESTING = data.Field(tokenize=list, preprocessing=zero_char)  # process a word in char list
    CHAR_TEXT = data.NestedField(CHAR_NESTING)  #
    LABEL = data.Field(unk_token=None, pad_token="O", batch_first=True)
    train_data = ConllDataset(WORD_TEXT, CHAR_TEXT, LABEL, os.path.join(data_path, "train.txt"))
    dev_data = ConllDataset(WORD_TEXT, CHAR_TEXT, LABEL, os.path.join(data_path, "dev.txt"))
    test_data = ConllDataset(WORD_TEXT, CHAR_TEXT, LABEL, os.path.join(data_path, "test.txt"))

    if vectors is not None:
        WORD_TEXT.build_vocab(train_data.word, vectors=vectors,
                              unk_init=unk_init)
    else:
        WORD_TEXT.build_vocab(train_data.word)
    CHAR_TEXT.build_vocab(train_data.char)
    LABEL.build_vocab(train_data.label)

    train_iter, dev_iter = BucketIterator.splits(
        (train_data, dev_data),
        batch_sizes=(batch_size, batch_size),
        device=device,
        sort_key=lambda x: len(x.word),
        sort_within_batch=True,
        repeat=False,
        shuffle=True
    )

    test_iter = Iterator(test_data, batch_size=batch_size, device=device, sort=False, sort_within_batch=False,
                         repeat=False, shuffle=False)
    return train_iter, dev_iter, test_iter, WORD_TEXT, CHAR_TEXT, LABEL

# load_iters(use_char=True)
