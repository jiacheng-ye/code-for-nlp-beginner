from util import load_iters
import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.vocab import Vectors
from models import ESIM
from tqdm import tqdm, trange
torch.manual_seed(1)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

BATCH_SIZE = 32
HIDDEN_SIZE = 300
EPOCHS = 20
DROPOUT_RATE = 0.5
LAYER_NUM = 1
LEARNING_RATE = 3e-4
data_path = 'data'

EMBEDDING_SIZE = 300
vectors = None
# vectors = Vectors('glove.6B.300d.txt', '/home/yjc/embeddings')
freeze = False


def show_example(premise, hypothesis, label, TEXT, LABEL):
    tqdm.write('Label: ' + LABEL.vocab.itos[label])
    tqdm.write('premise: ' + ' '.join([TEXT.vocab.itos[i] for i in premise]))
    tqdm.write('hypothesis: ' + ' '.join([TEXT.vocab.itos[i] for i in hypothesis]))


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def eval(data_iter, name, epoch=None):
    model.eval()
    with torch.no_grad():
        correct_num = 0
        err_num = 0
        total_loss = 0
        for i, batch in enumerate(data_iter):
            premise, premise_lens = batch.premise
            hypothesis, hypothesis_lens = batch.hypothesis
            labels = batch.label

            output = model(premise, premise_lens, hypothesis, hypothesis_lens)
            predicts = output.argmax(-1).reshape(-1)
            loss = loss_func(output, labels)
            total_loss += loss.item()
            correct_num += (predicts == labels).sum().item()
            err_num += (predicts != batch.label).sum().item()

    if epoch is not None:
        tqdm.write(
            "Epoch: %d, %s Acc: %.3f, Loss %.3f" % (epoch + 1, name, correct_num / (correct_num + err_num), total_loss))
    else:
        tqdm.write(
            "%s Acc: %.3f, Loss %.3f" % (name, correct_num / (correct_num + err_num), total_loss))


def train(train_iter, dev_iter, loss_func, optimizer, epochs):
    for epoch in trange(epochs):
        model.train()
        total_loss = 0
        for batch in tqdm(train_iter):
            premise, premise_lens = batch.premise
            hypothesis, hypothesis_lens = batch.hypothesis
            labels = batch.label
            # show_example(premise[0],hypothesis[0], labels[0], TEXT, LABEL)

            model.zero_grad()
            output = model(premise, premise_lens, hypothesis, hypothesis_lens)
            loss = loss_func(output, labels)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
        tqdm.write("Epoch: %d, Train Loss: %d" % (epoch + 1, total_loss))

        eval(dev_iter, "Dev", epoch)

if __name__ == "__main__":
    train_iter, dev_iter, test_iter, TEXT, LABEL, _ = load_iters(BATCH_SIZE, device, data_path, vectors)

    model = ESIM(len(TEXT.vocab), len(LABEL.vocab.stoi), EMBEDDING_SIZE, HIDDEN_SIZE, DROPOUT_RATE, LAYER_NUM)
    if vectors is not None:
        model.embed.from_pretrained(TEXT.vocab.vectors, freeze=freeze)
    model.to(device)
    print(f'The model has {count_parameters(model):,} trainable parameters')

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_func = nn.CrossEntropyLoss()

    train(train_iter, dev_iter, loss_func, optimizer, EPOCHS)
    eval(test_iter, "Test")
