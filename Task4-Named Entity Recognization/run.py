# -*- coding:utf8 -*-
import torch
import torch.optim as optim
from tqdm import tqdm
from torchtext.vocab import Vectors
from models import NER_Model
import codecs
from util import load_iters, get_chunks

torch.manual_seed(1)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

DATA_PATH = "data"
PREDICT_OUT_FILE = "res2.txt"
BEST_MODEL = "best_model2.ckpt"
BATCH_SIZE = 10
LOWER_CASE = False
EPOCHS = 200

# embedding
WORD_VECTORS = None
# WORD_VECTORS = Vectors('glove.6B.100d.txt', '../../embeddings/glove.6B')
WORD_EMBEDDING_SIZE = 100
CHAR_VECTORS = None
CHAR_EMBEDDING_SIZE = 30  # the input char embedding to CNN
FREEZE_EMBEDDING = False

# SGD parameters
LEARNING_RATE = 0.015
DECAY_RATE = 0.05
MOMENTUM = 0.9
CLIP = 5
PATIENCE = 5

# network parameters
HIDDEN_SIZE = 400  # every LSTM's(forward and backward) hidden size is half of HIDDEN_SIZE
LSTM_LAYER_NUM = 1
DROPOUT_RATE = (0.5, 0.5, (0.5, 0.5))  # after embed layer, other case, (input to rnn, between rnn layers)
USE_CHAR = True  # use char level information
N_FILTERS = 30  # the output char embedding from CNN
KERNEL_STEP = 3  # n-gram size of CNN
USE_CRF = True


def train(train_iter, dev_iter, optimizer):
    best_dev_f1 = -1
    patience_counter = 0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0
        train_iter.init_epoch()
        for i, batch in enumerate(tqdm(train_iter)):
            words, lens = batch.word
            labels = batch.label
            if i < 2:
                tqdm.write(' '.join([WORD.vocab.itos[i] for i in words[0]]))
                tqdm.write(' '.join([LABEL.vocab.itos[i] for i in labels[0]]))
            model.zero_grad()
            loss = model(words, batch.char, lens, labels)
            total_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
            optimizer.step()
        tqdm.write("Epoch: %d, Train Loss: %d" % (epoch, total_loss))

        lr = LEARNING_RATE / (1 + DECAY_RATE * epoch)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        dev_f1 = eval(dev_iter, "Dev", epoch)
        if dev_f1 < best_dev_f1:
            patience_counter += 1
            tqdm.write("No improvement, patience: %d/%d" % (patience_counter, PATIENCE))
        else:
            best_dev_f1 = dev_f1
            patience_counter = 0
            torch.save(model.state_dict(), BEST_MODEL)
            tqdm.write("New best model, saved to best_model.ckpt, patience: 0/%d" % PATIENCE)
        if patience_counter >= PATIENCE:
            tqdm.write("Early stopping: patience limit reached, stopping...")
            break


def eval(data_iter, name, epoch=None, best_model=None):
    if best_model:
        model.load_state_dict(torch.load(best_model))
    model.eval()
    with torch.no_grad():
        total_loss = 0
        res = {'ootv': [0, 0, 0], 'ooev': [0, 0, 0], 'oobv': [0, 0, 0], 'iv': [0, 0, 0],
               'total': [0, 0, 0]}  # e.g. 'iv':[correct_preds, total_preds, total_correct]
        for i, batch in enumerate(data_iter):
            words, lens = batch.word
            labels = batch.label
            predicted_seq, _ = model(words, batch.char, lens)  # predicted_seq : (batch_size, seq_len)
            loss = model(words, batch.char, lens, labels)
            total_loss += loss.item()

            orig_text = [e.word for e in data_iter.dataset.examples[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]]
            for text, ground_truth_id, predicted_id, len_ in zip(orig_text, labels.cpu().numpy(),
                                                                 predicted_seq.cpu().numpy(),
                                                                 lens.cpu().numpy()):
                lab_chunks = set(get_chunks(ground_truth_id[:len_], LABEL.vocab.stoi))
                lab_pred_chunks = set(get_chunks(predicted_id[:len_], LABEL.vocab.stoi))

                for chunk in list(lab_chunks):
                    entity_word = ' '.join([text[ix] for ix in range(chunk[1], chunk[2])])
                    # type, 1:ootv, 2:ooev, 3:oobv, 4:iv
                    entity_type = WORD.vocab.dev_entity2type[entity_word] if name == "Dev" else \
                    WORD.vocab.test_entity2type[entity_word]
                    if entity_type == 1:
                        if chunk in lab_pred_chunks:
                            res['ootv'][0] += 1
                        res['ootv'][2] += 1
                    elif entity_type == 2:
                        if chunk in lab_pred_chunks:
                            res['ooev'][0] += 1
                        res['ooev'][2] += 1
                    elif entity_type == 3:
                        if chunk in lab_pred_chunks:
                            res['oobv'][0] += 1
                        res['oobv'][2] += 1
                    else:
                        if chunk in lab_pred_chunks:
                            res['iv'][0] += 1
                        res['iv'][2] += 1
                    if chunk in lab_pred_chunks:
                        res['total'][0] += 1
                    res['total'][2] += 1
                for chunk in list(lab_pred_chunks):
                    entity_word = ' '.join([text[ix] for ix in range(chunk[1], chunk[2])])
                    # type, 1:ootv, 2:ooev, 3:oobv, 4:iv
                    entity_type = WORD.vocab.dev_entity2type.get(entity_word, None) if name == "Dev" else \
                        WORD.vocab.test_entity2type.get(entity_word, None)
                    if entity_type == 1:
                        res['ootv'][1] += 1
                    elif entity_type == 2:
                        res['ooev'][1] += 1
                    elif entity_type == 3:
                        res['oobv'][1] += 1
                    elif entity_type == 4:
                        res['iv'][1] += 1
                    res['total'][1] += 1

        # Calculating the F1-Score
        for k, v in res.items():
            p = v[0] / v[1] if v[1] != 0 else 0
            r = v[0] / v[2] if v[2] != 0 else 0
            micro_F1 = 2 * p * r / (p + r) if (p + r) != 0 else 0
            if epoch is not None:
                tqdm.write(
                    "Epoch: %d, %s, %s Entity Micro F1: %.3f, Loss %.3f" % (epoch, name, k, micro_F1, total_loss))
            else:
                tqdm.write(
                    "%s, %s Entity Micro F1: %.3f, Loss %.3f" % (name, k, micro_F1, total_loss))
    return micro_F1


def predict(data_iter, out_file):
    model.eval()
    with torch.no_grad():
        gold_seqs = []
        predicted_seqs = []
        word_seqs = []
        for i, batch in enumerate(data_iter):
            words, lens = batch.word
            predicted_seq, _ = model(words, batch.char, lens)  # predicted_seq : (batch_size, seq_len)
            gold_seqs.extend(batch.label.tolist())
            predicted_seqs.extend(predicted_seq.tolist())
            word_seqs.extend(words.tolist())
        write_predicted_labels(out_file, data_iter.dataset.examples, word_seqs, LABEL.vocab.itos, gold_seqs,
                               predicted_seqs)


def write_predicted_labels(output_file, orig_text, word_ids, id2label, gold_seq, predicted_seq):
    with codecs.open(output_file, 'w', encoding='utf-8') as writer:
        for text, wids, predict, gold in zip(orig_text, word_ids, predicted_seq, gold_seq):
            ix = 0
            for w_id, p_id, g_id in zip(wids, predict, gold):
                if w_id == pad_idx: break
                output_line = ' '.join([text.word[ix], id2label[g_id], id2label[p_id]])
                writer.write(output_line + '\n')
                ix += 1
            writer.write('\n')


if __name__ == "__main__":
    train_iter, dev_iter, test_iter, WORD, CHAR, LABEL = load_iters(WORD_EMBEDDING_SIZE, WORD_VECTORS,
                                                                    CHAR_EMBEDDING_SIZE, CHAR_VECTORS,
                                                                    BATCH_SIZE, DEVICE, DATA_PATH, LOWER_CASE)

    model = NER_Model(WORD.vocab.vectors, CHAR.vocab.vectors, len(LABEL.vocab.stoi), HIDDEN_SIZE, DROPOUT_RATE,
                      LSTM_LAYER_NUM,
                      KERNEL_STEP, N_FILTERS, USE_CHAR, FREEZE_EMBEDDING, USE_CRF).to(DEVICE)
    print(model)
    pad_idx = WORD.vocab.stoi[WORD.pad_token]

    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)
    # train(train_iter, dev_iter, optimizer)
    eval(test_iter, "Test", best_model=BEST_MODEL)
    predict(test_iter, PREDICT_OUT_FILE)
