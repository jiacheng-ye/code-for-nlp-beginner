from torchtext import data
from torchtext.data import BucketIterator
import os


def read_data(input_file, max_length):
    with open(input_file, encoding="utf8") as f:
        poetries = []
        poetry = []
        for line in f:
            contends = line.strip()
            if len(poetry) + len(contends) <= max_length:
                if contends:
                    poetry.extend(contends)
                else:
                    poetries.append(poetry)
                    poetry = []
            else:
                poetries.append(poetry)
                poetry = list(contends)
        if poetry:
            poetries.append(poetry)
        return poetries


class PoetryDataset(data.Dataset):

    def __init__(self, text_field, datafile, max_length, **kwargs):
        fields = [("text", text_field)]
        datas = read_data(datafile, max_length)
        examples = []
        for text in datas:
            examples.append(data.Example.fromlist([text], fields))
        super(PoetryDataset, self).__init__(examples, fields, **kwargs)


def load_iters(eos_token="[EOS]", batch_size=32, device="cpu", data_path='data', max_length=128):
    TEXT = data.Field(eos_token=eos_token, batch_first=True, include_lengths=True)
    datas = PoetryDataset(TEXT, os.path.join(data_path, "poetryFromTang.txt"), max_length)
    train_data, dev_data, test_data = datas.split([0.8, 0.1, 0.1])

    TEXT.build_vocab(train_data)

    train_iter, dev_iter, test_iter = BucketIterator.splits(
        (train_data, dev_data, test_data),
        batch_sizes=(batch_size, batch_size, batch_size),
        device=device,
        sort_key=lambda x: len(x.text),
        sort_within_batch=True,
        repeat=False,
        shuffle=True
    )
    return train_iter, dev_iter, test_iter, TEXT
