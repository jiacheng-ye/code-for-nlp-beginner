# -*- coding:utf8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm, trange
from tensorboardX import SummaryWriter
from models import LSTM_LM
from util import load_iters
import math

torch.manual_seed(1)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

BATCH_SIZE = 64
HIDDEN_DIM = 512
LAYER_NUM = 1
EPOCHS = 200
DROPOUT_RATE = 0.5
LEARNING_RATE = 0.01
MOMENTUM = 0.9
CLIP = 5
DECAY_RATE = 0.05  # learning rate decay rate
EOS_TOKEN = "[EOS]"
DATA_PATH = 'data'
EMBEDDING_SIZE = 200
TEMPERATURE = 0.8  # Higher temperature means more diversity.
MAX_LEN = 64


def train(train_iter, dev_iter, loss_func, optimizer, epochs, clip):
    for epoch in trange(epochs):
        model.train()
        total_loss = 0
        total_words = 0
        for i, batch in enumerate(tqdm(train_iter)):
            text, lens = batch.text
            if epoch == 0 and i == 0:
                tqdm.write(' '.join([TEXT.vocab.itos[i] for i in text[0]]))
                tqdm.write(' '.join([str(i.item()) for i in text[0]]))
            inputs = text[:, :-1]
            targets = text[:, 1:]
            init_hidden = model.lstm.init_hidden(inputs.size(0))
            logits, _ = model(inputs, lens - 1, init_hidden)  # [EOS] is included in length.
            loss = loss_func(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))

            model.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
            total_loss += loss.item()
            total_words += lens.sum().item()
        tqdm.write("Epoch: %d, Train perplexity: %d" % (epoch + 1, math.exp(total_loss / total_words)))
        writer.add_scalar('Train_Loss', total_loss, epoch)
        eval(dev_iter, True, epoch)

        lr = LEARNING_RATE / (1 + DECAY_RATE * (epoch + 1))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


def eval(data_iter, is_dev=False, epoch=None):
    model.eval()
    with torch.no_grad():
        total_words = 0
        total_loss = 0
        for i, batch in enumerate(data_iter):
            text, lens = batch.text
            inputs = text[:, :-1]
            targets = text[:, 1:]
            model.zero_grad()
            init_hidden = model.lstm.init_hidden(inputs.size(0))
            logits, _ = model(inputs, lens - 1, init_hidden)  # [EOS] is included in length.
            loss = loss_func(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))

            total_loss += loss.item()
            total_words += lens.sum().item()
    if epoch is not None:
        tqdm.write(
            "Epoch: %d, %s perplexity %.3f" % (
                epoch + 1, "Dev" if is_dev else "Test", math.exp(total_loss / total_words)))
        writer.add_scalar('Dev_Loss', total_loss, epoch)
    else:
        tqdm.write(
            "%s perplexity %.3f" % ("Dev" if is_dev else "Test", math.exp(total_loss / total_words)))


def generate(eos_idx, word, temperature=0.8):
    model.eval()
    with torch.no_grad():
        if word in TEXT.vocab.stoi:
            idx = TEXT.vocab.stoi[word]
            inputs = torch.tensor([idx])
        else:
            print("%s is not in vocabulary, choose by random." % word)
            prob = torch.ones(len(TEXT.vocab.stoi))
            inputs = torch.multinomial(prob, 1)
            idx = inputs[0].item()

        inputs = inputs.unsqueeze(1).to(device)  # shape [1, 1]
        lens = torch.tensor([1]).to(device)
        hidden = tuple([h.to(device) for h in model.lstm.init_hidden(1)])
        poetry = [TEXT.vocab.itos[idx]]

        while idx != eos_idx:
            logits, hidden = model(inputs, lens, hidden)  # logits: (1, 1, vocab_size)
            word_weights = logits.squeeze().div(temperature).exp().cpu()
            idx = torch.multinomial(word_weights, 1)[0].item()
            inputs.fill_(idx)
            poetry.append(TEXT.vocab.itos[idx])
        print("".join(poetry[:-1]))


if __name__ == "__main__":
    train_iter, dev_iter, test_iter, TEXT = load_iters(EOS_TOKEN, BATCH_SIZE, device, DATA_PATH, MAX_LEN)
    pad_idx = TEXT.vocab.stoi[TEXT.pad_token]
    eos_idx = TEXT.vocab.stoi[EOS_TOKEN]
    model = LSTM_LM(len(TEXT.vocab), EMBEDDING_SIZE, HIDDEN_DIM, DROPOUT_RATE, LAYER_NUM).to(device)

    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)
    loss_func = torch.nn.CrossEntropyLoss(ignore_index=pad_idx, reduction="sum")
    writer = SummaryWriter("logs")
    train(train_iter, dev_iter, loss_func, optimizer, EPOCHS, CLIP)
    eval(test_iter, is_dev=False)
    try:
        while True:
            word = input("Input the first word or press Ctrl-C to exit: ")
            generate(eos_idx, word.strip(), TEMPERATURE)
    except:
        pass
