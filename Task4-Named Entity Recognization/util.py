from torchtext import data
from torchtext.data import Iterator, BucketIterator
import os
import re
import math
import torch
import numpy as np


def read_data(input_file):
    """Reads a BIO data."""
    with open(input_file) as f:
        lines = []
        words = []
        labels = []
        for line in f:
            contends = line.strip()
            # if contends.startswith("-DOCSTART-"):
            #     continue
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


def get_char_detail(train, other, embed_vocab=None):
    char2type = {}  # type, 1:ootv, 2:ooev, 3:oobv, 4:iv
    ootv = 0
    ootv_set = set()
    ooev = 0
    oobv = 0
    iv = 0
    fuzzy_iv = 0
    for sent in other:
        for w in sent:
            for c in w:
                if c not in char2type:
                    if c not in train.stoi:
                        if embed_vocab and (c in embed_vocab.stoi or c.lower() in embed_vocab.stoi):
                            ootv += 1
                            ootv_set.add(c)
                            char2type[c] = 1
                        else:
                            oobv += 1
                            char2type[c] = 3
                    else:
                        if embed_vocab and (c in embed_vocab.stoi or c.lower() in embed_vocab.stoi):
                            fuzzy_iv += 1 if c.lower() in embed_vocab.stoi else 0
                            iv += 1
                            char2type[c] = 4
                        else:
                            ooev += 1
                            char2type[c] = 2
    print("IV {}(fuzzy {})\nOOTV {}\nOOEV {}\nOOBV {}\n".format(iv, fuzzy_iv, ootv, ooev, oobv))
    return char2type, ootv_set


def get_word_detail(train, other, embed_vocab=None):
    '''
    OOTV words are the ones do not appear in training set but in embedding vocabulary
    OOEV words are the ones do not appear in embedding vocabulary but in training set
    OOBV words are the ones do not appears in both the training and embedding vocabulary
    IV words the ones appears in both the training and embedding vocabulary
    '''
    word2type = {}  # type, 1:ootv, 2:ooev, 3:oobv, 4:iv
    ootv = 0
    ootv_set = set()
    ooev = 0
    oobv = 0
    iv = 0
    fuzzy_iv = 0
    for sent in other:
        for w in sent:
            if w not in word2type:
                if w not in train.stoi:
                    if embed_vocab and (w in embed_vocab.stoi or w.lower() in embed_vocab.stoi):
                        ootv += 1
                        ootv_set.add(w)
                        word2type[w] = 1
                    else:
                        oobv += 1
                        word2type[w] = 3
                else:
                    if embed_vocab and (w in embed_vocab.stoi or w.lower() in embed_vocab.stoi):
                        fuzzy_iv += 1 if w not in embed_vocab.stoi else 0
                        iv += 1
                        word2type[w] = 4
                    else:
                        ooev += 1
                        word2type[w] = 2
    print("IV {}(fuzzy {})\nOOTV {}\nOOEV {}\nOOBV {}\n".format(iv, fuzzy_iv, ootv, ooev, oobv))
    return word2type, ootv_set


def get_entity_detail(vocab, data, tag2id, embed_vocab=None):
    entity2type = {}  # type, 1:ootv, 2:ooev, 3:oobv, 4:iv
    ootv = 0  # every word of the entity have embedding, but at least one word not in training set
    oobv = 0  # an entity is considered as OOBV if there exists at least one word not in training set and at least one word not in embedding vocabulary
    ooev = 0  # every word of the entity is in training set, but at least one word not have embedding
    iv = 0
    for ex in data.examples:
        ens = get_chunks(ex.label, tag2id, id_format=False)
        for e in ens:
            if e not in entity2type:
                entity_words = [ex.word[ix] for ix in range(e[1], e[2])]
                entity_word = ' '.join(entity_words)
                not_in_vocab = len(list(filter(lambda w: w not in vocab.stoi, entity_words)))
                if embed_vocab:
                    not_in_embed = len(list(
                        filter(lambda w: w not in embed_vocab.stoi and w.lower() not in embed_vocab.stoi,
                               entity_words)))
                if not_in_vocab > 0:
                    if embed_vocab and not_in_embed == 0:
                        ootv += 1
                        entity2type[entity_word] = 1
                    else:
                        oobv += 1
                        entity2type[entity_word] = 3
                else:
                    if embed_vocab and not_in_embed == 0:
                        iv += 1
                        entity2type[entity_word] = 4
                    else:
                        ooev += 1
                        entity2type[entity_word] = 2

    print("IV {}\nOOTV {}\nOOEV {}\nOOBV {}\n".format(iv, ootv, ooev, oobv))
    return entity2type


def extend(vocab, v, sort=False):
    words = sorted(v) if sort else v
    for w in words:
        if w not in vocab.stoi:
            vocab.itos.append(w)
            vocab.stoi[w] = len(vocab.itos) - 1


def get_entities(vocab, data, tag2id):
    entities = {}
    unk = 0
    conflict = 0
    for ex in data.examples:
        ens = get_chunks(ex.label, tag2id, id_format=False)
        for e in ens:
            entity_words = [ex.word[ix] if ex.word[ix] in vocab.stoi else vocab.UNK for ix in range(e[1], e[2])]
            entities.setdefault(' '.join(entity_words), set())
            entities[' '.join(entity_words)].add(e[0])
            if vocab.UNK in entity_words:
                unk += 1
            if len(entities[' '.join(entity_words)]) == 2:
                conflict += 1
    print("entities contains `UNK` %d\nconflict entities %d\nall entities: %d\n" % (unk, conflict, len(entities)))
    return entities


def load_iters(word_embed_size, word_vectors, char_embedding_size, char_vectors, batch_size=32, device="cpu",
               data_path='data', word2lower=True):
    zero_char_in_word = lambda ex: [re.sub('\d', '0', w) for w in ex]
    zero_char = lambda w: [re.sub('\d', '0', c) for c in w]

    WORD_TEXT = data.Field(lower=word2lower, batch_first=True, include_lengths=True,
                           preprocessing=zero_char_in_word)
    CHAR_NESTING = data.Field(tokenize=list, preprocessing=zero_char)  # process a word in char list
    CHAR_TEXT = data.NestedField(CHAR_NESTING)
    LABEL = data.Field(unk_token=None, pad_token="O", batch_first=True)

    train_data = ConllDataset(WORD_TEXT, CHAR_TEXT, LABEL, os.path.join(data_path, "train.txt"))
    dev_data = ConllDataset(WORD_TEXT, CHAR_TEXT, LABEL, os.path.join(data_path, "dev.txt"))
    test_data = ConllDataset(WORD_TEXT, CHAR_TEXT, LABEL, os.path.join(data_path, "test.txt"))

    print("train sentence num / total word num: %d/%d" % (
        len(train_data.examples), np.array([len(_.word) for _ in train_data.examples]).sum()))
    print("dev sentence num / total word num: %d/%d" % (
        len(dev_data.examples), np.array([len(_.word) for _ in dev_data.examples]).sum()))
    print("test sentence num / total word num: %d/%d" % (
        len(test_data.examples), np.array([len(_.word) for _ in test_data.examples]).sum()))

    LABEL.build_vocab(train_data.label)
    WORD_TEXT.build_vocab(train_data.word, max_size=50000, min_freq=1)
    CHAR_TEXT.build_vocab(train_data.char, max_size=50000, min_freq=1)

    # ------------------- word oov analysis-----------------------
    print('*' * 50 + ' unique words details of dev set ' + '*' * 50)
    dev_word2type, dev_ootv_set = get_word_detail(WORD_TEXT.vocab, dev_data.word, word_vectors)
    print('#' * 110)
    print('*' * 50 + ' unique words details of test set ' + '*' * 50)
    test_word2type, test_ootv_set = get_word_detail(WORD_TEXT.vocab, test_data.word, word_vectors)
    print('#' * 110)
    WORD_TEXT.vocab.dev_word2type = dev_word2type
    WORD_TEXT.vocab.test_word2type = test_word2type

    # ------------------- entity oov analysis-----------------------
    print('*' * 50 + ' get train entities ' + '*' * 50)
    train_entities = get_entities(WORD_TEXT.vocab, train_data, LABEL.vocab.stoi)
    print('#' * 110)
    print('*' * 50 + ' get dev entities ' + '*' * 50)
    dev_entity2type = get_entity_detail(WORD_TEXT.vocab, dev_data, LABEL.vocab.stoi, word_vectors)
    print('#' * 110)
    print('*' * 50 + ' get test entities ' + '*' * 50)
    test_entity2type = get_entity_detail(WORD_TEXT.vocab, test_data, LABEL.vocab.stoi, word_vectors)
    print('#' * 110)
    WORD_TEXT.vocab.dev_entity2type = dev_entity2type
    WORD_TEXT.vocab.test_entity2type = test_entity2type

    # ------------------- extend word vocab with ootv words -----------------------
    print('*' * 50 + 'extending ootv words to vocab' + '*' * 50)
    ootv = list(dev_ootv_set.union(test_ootv_set))
    extend(WORD_TEXT.vocab, ootv)
    print('extended %d words' % len(ootv))
    print('#' * 110)

    # ------------------- generate word embedding -----------------------
    vectors_to_use = unk_init(torch.zeros((len(WORD_TEXT.vocab), word_embed_size)))
    if word_vectors is not None:
        vectors_to_use = get_vectors(vectors_to_use, WORD_TEXT.vocab, word_vectors)
    WORD_TEXT.vocab.vectors = vectors_to_use

    # ------------------- char oov analysis-----------------------
    print('*' * 50 + ' unique chars details of dev set ' + '*' * 50)
    dev_char2type, dev_ootv_set = get_char_detail(CHAR_TEXT.vocab, dev_data.char, char_vectors)
    print('#' * 110)
    print('*' * 50 + ' unique chars details of test set ' + '*' * 50)
    test_char2type, test_ootv_set = get_char_detail(CHAR_TEXT.vocab, test_data.char, char_vectors)
    print('#' * 110)
    CHAR_TEXT.vocab.dev_char2type = dev_char2type
    CHAR_TEXT.vocab.test_char2type = test_char2type

    # ------------------- extend char vocab with ootv chars -----------------------
    print('*' * 50 + 'extending ootv chars to vocab' + '*' * 50)
    ootv = list(dev_ootv_set.union(test_ootv_set))
    extend(CHAR_TEXT.vocab, ootv)
    print('extended %d chars' % len(ootv))
    print('#' * 110)

    # ------------------- generate char embedding -----------------------
    vectors_to_use = unk_init(torch.zeros((len(CHAR_TEXT.vocab), char_embedding_size)))
    if char_vectors is not None:
        vectors_to_use = get_vectors(vectors_to_use, CHAR_TEXT.vocab, char_vectors)
    CHAR_TEXT.vocab.vectors = vectors_to_use

    print("word vocab size: ", len(WORD_TEXT.vocab))
    print("char vocab size: ", len(CHAR_TEXT.vocab))
    print("label vocab size: ", len(LABEL.vocab))

    train_iter = BucketIterator(train_data, batch_size=batch_size, device=device, sort_key=lambda x: len(x.word),
                                sort_within_batch=True, repeat=False, shuffle=True)
    dev_iter = Iterator(dev_data, batch_size=batch_size, device=device, sort=False, sort_within_batch=False,
                        repeat=False, shuffle=False)
    test_iter = Iterator(test_data, batch_size=batch_size, device=device, sort=False, sort_within_batch=False,
                         repeat=False, shuffle=False)
    return train_iter, dev_iter, test_iter, WORD_TEXT, CHAR_TEXT, LABEL


def get_chunk_type(tok, idx_to_tag):
    """
    The function takes in a chunk ("B-PER") and then splits it into the tag (PER) and its class (B)
    as defined in BIOES

    Args:
        tok: id of token, ex 4
        idx_to_tag: dictionary {4: "B-PER", ...}

    Returns:
        tuple: "B", "PER"

    """

    tag_name = idx_to_tag[tok]
    tag_class = tag_name.split('-')[0]
    tag_type = tag_name.split('-')[-1]
    return tag_class, tag_type


def get_chunks(seq, tags, bioes=True, id_format=True):
    """
    Given a sequence of tags, group entities and their position
    """
    if not id_format:
        seq = [tags[_] for _ in seq]

    # We assume by default the tags lie outside a named entity
    default = tags["O"]

    idx_to_tag = {idx: tag for tag, idx in tags.items()}

    chunks = []

    chunk_class, chunk_type, chunk_start = None, None, None
    for i, tok in enumerate(seq):
        if tok == default and (chunk_class in (["E", "S"] if bioes else ["B", "I"])):
            # Add a chunk.
            chunk = (chunk_type, chunk_start, i)
            chunks.append(chunk)
            chunk_class, chunk_type, chunk_start = "O", None, None

        if tok != default:
            tok_chunk_class, tok_chunk_type = get_chunk_type(tok, idx_to_tag)
            if chunk_type is None:
                # Initialize chunk for each entity
                chunk_class, chunk_type, chunk_start = tok_chunk_class, tok_chunk_type, i
            else:
                if bioes:
                    if chunk_class in ["E", "S"]:
                        chunk = (chunk_type, chunk_start, i)
                        chunks.append(chunk)
                        if tok_chunk_class in ["B", "S"]:
                            chunk_class, chunk_type, chunk_start = tok_chunk_class, tok_chunk_type, i
                        else:
                            chunk_class, chunk_type, chunk_start = None, None, None
                    elif tok_chunk_type == chunk_type and chunk_class in ["B", "I"]:
                        chunk_class = tok_chunk_class
                    else:
                        chunk_class, chunk_type = None, None
                else:  # BIO schema
                    if tok_chunk_class == "B":
                        chunk = (chunk_type, chunk_start, i)
                        chunks.append(chunk)
                        chunk_class, chunk_type, chunk_start = tok_chunk_class, tok_chunk_type, i
                    else:
                        chunk_class, chunk_type = None, None

    if chunk_type is not None:
        chunk = (chunk_type, chunk_start, len(seq))
        chunks.append(chunk)

    return chunks


def get_vectors(embed, vocab, pretrain_embed_vocab):
    oov = 0
    for i, word in enumerate(vocab.itos):
        index = pretrain_embed_vocab.stoi.get(word, None)  # digit or None
        if index is None:
            if word.lower() in pretrain_embed_vocab.stoi:
                index = pretrain_embed_vocab.stoi[word.lower()]
        if index:
            embed[i] = pretrain_embed_vocab.vectors[index]
        else:
            oov += 1
    print('train vocab oov %d \ntrain vocab + dev ootv + test ootv: %d' % (oov, len(vocab.stoi)))
    return embed


def test_get_chunks():
    print(get_chunks([4, 2, 1, 2, 3, 3],
                     {'O': 0, "B-PER": 1, "I-PER": 2, "E-PER": 3, "S-PER": 4}))
    print(get_chunks(["S-PER", "I-PER", "B-PER", "I-PER", "E-PER", "E-PER"],
                     {'O': 0, "B-PER": 1, "I-PER": 2, "E-PER": 3, "S-PER": 4}, id_format=False))
