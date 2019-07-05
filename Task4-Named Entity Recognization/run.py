# -*- coding:utf8 -*-
import torch
import torch.optim as optim
from tqdm import tqdm, trange
from torchtext.vocab import Vectors
from models import NER_Model
import codecs
from util import load_iters, get_chunks

torch.manual_seed(1)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data_path = "data"
vectors = None
# vectors = Vectors('glove.6B.100d.txt', '/home/yjc/embeddings')
FREEZE = False
BATCH_SIZE = 10
LOWER_CASE = False
EPOCHS = 100
# SGD parameters
LEARNING_RATE = 0.015
DECAY_RATE = 0.05
MOMENTUM = 0.9
CLIP = 5
PATIENCE = 5
# network parameters
WORD_EMBEDDING_SIZE = 100
HIDDEN_SIZE = 400  # every LSTM's(forward and backward) hidden size is half of HIDDEN_SIZE
LSTM_LAYER_NUM = 1
DROPOUT_RATE = 0.5
USE_CHAR = True  # use char embedding
CHAR_EMBEDDING_SIZE = 30  # the input char embedding to CNN
N_FILTERS = 30  # the output char embedding from CNN
KERNEL_STEP = 3  # n-gram size of CNN


def train(train_iter, dev_iter, optimizer, epochs, clip, patience):
    best_dev_f1 = -1
    patience_counter = 0
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for i, batch in enumerate(tqdm(train_iter)):
            words, lens = batch.word
            labels = batch.label
            if i == 0:
                tqdm.write(' '.join([WORD.vocab.itos[i] for i in words[0]]))
                tqdm.write(' '.join([LABEL.vocab.itos[i] for i in labels[0]]))
            model.zero_grad()
            loss = model(words, batch.char, lens, labels)
            total_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
        tqdm.write("Epoch: %d, Train Loss: %d" % (epoch + 1, total_loss))

        lr = LEARNING_RATE / (1 + DECAY_RATE * (epoch + 1))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        dev_f1 = eval(dev_iter, "Dev", epoch)
        if dev_f1 < best_dev_f1:
            patience_counter += 1
        else:
            best_dev_f1 = dev_f1
            patience_counter = 0
            torch.save(model.state_dict(), 'best_model.ckpt')
        if patience_counter >= patience:
            tqdm.write("Early stopping: patience limit reached, stopping...")
            break

def eval(data_iter, name, epoch=None, use_cache=False):
    if use_cache:
        model.load_state_dict(torch.load('params.ckpt'))
    model.eval()
    with torch.no_grad():
        total_loss = 0
        correct_preds = 0
        total_preds = 0
        total_correct = 0
        for i, batch in enumerate(data_iter):
            words, lens = batch.word
            labels = batch.label
            predicted_seq, _ = model(words, batch.char, lens)  # predicted_seq : (batch_size, seq_len)
            loss = model(words, batch.char, lens, labels)
            total_loss += loss.item()

            for ground_truth_id, predicted_id, len_ in zip(labels.cpu().numpy(), predicted_seq.cpu().numpy(), lens.cpu().numpy()):
                lab_chunks = set(get_chunks(ground_truth_id[:len_], LABEL.vocab.stoi))
                lab_pred_chunks = set(get_chunks(predicted_id[:len_], LABEL.vocab.stoi))

                # Updating the count variables
                correct_preds += len(lab_chunks & lab_pred_chunks)
                total_preds += len(lab_pred_chunks)
                total_correct += len(lab_chunks)

        # Calculating the F1-Score
        p = correct_preds / total_preds
        r = correct_preds / total_correct
        micro_F1 = 2 * p * r / (p + r)

    if epoch is not None:
        tqdm.write(
            "Epoch: %d, %s Entity Micro F1: %.3f, Loss %.3f" % (epoch + 1, name,micro_F1 , total_loss))
    else:
        tqdm.write(
            "%s Acc: %.3f, Loss %.3f" % (name, micro_F1, total_loss))
    return micro_F1

def predict(data_iter):
    model.eval()
    with torch.no_grad():
        orig_texts = []
        gold_seqs = []
        predicted_seqs = []
        for i, batch in enumerate(data_iter):
            orig_text = [' '.join(e.word) for e in data_iter.dataset.examples[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]]
            words, lens = batch.word
            predicted_seq, _ = model(words, batch.char, lens)  # predicted_seq : (batch_size, seq_len)
            gold_seqs.extend(batch.label.tolist())
            orig_texts.extend(orig_text)
            predicted_seqs.extend(predicted_seq.tolist())
        write_predicted_labels("predictions.txt", orig_texts, LABEL.vocab.itos, gold_seqs, predicted_seqs)


def write_predicted_labels(output_file, orig_text, id2label, gold_seq, predicted_seq):
    pad_idx = WORD.vocab.stoi[WORD.pad_token]
    with codecs.open(output_file, 'w', encoding='utf-8') as writer:
        for text, predict, gold in zip(orig_text, predicted_seq, gold_seq):
            for token, p_id, g_id in zip(text.split(), predict, gold):
                if g_id == pad_idx: break
                output_line = ' '.join([token, id2label[g_id], id2label[p_id]])
                writer.write(output_line + '\n')
            writer.write('\n')


if __name__ == "__main__":
    train_iter, dev_iter, test_iter, WORD, CHAR, LABEL = load_iters(BATCH_SIZE, device, data_path, vectors, LOWER_CASE)

    model = NER_Model(len(WORD.vocab), WORD_EMBEDDING_SIZE,
                      len(CHAR.vocab), CHAR_EMBEDDING_SIZE,
                      len(LABEL.vocab.stoi), HIDDEN_SIZE, DROPOUT_RATE, LSTM_LAYER_NUM,
                      KERNEL_STEP, N_FILTERS, USE_CHAR, WORD.vocab.vectors, FREEZE).to(device)

    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)
    train(train_iter, dev_iter, optimizer, EPOCHS, CLIP, PATIENCE)
    eval(test_iter, "Test", use_cache=True)
    predict(test_iter)
