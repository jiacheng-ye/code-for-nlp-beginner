import pandas as pd
import os
from collections import Counter
from scipy.sparse import csr_matrix
import numpy as np
from models import LogisticRegression, SoftmaxRegression

train_epochs = 10000
learning_rate = 0.00005
batch_size = 1024
class_num = 5
data_path = "data"
regularization = "l1"
C = 0.8


class Ngram():
    def __init__(self, n_grams, max_tf=0.8):
        ''' n_grams: tuple, n_gram range'''
        self.n_grams = n_grams
        self.tok2id = {}
        self.tok2tf = Counter()
        self.max_tf = max_tf

    @staticmethod
    def tokenize(text):
        ''' In this task, we simply the following tokenizer.'''
        return text.lower().split(" ")

    def get_n_grams(self, toks):
        ngrams_toks = []
        for ngrams in range(self.n_grams[0], self.n_grams[1] + 1):
            for i in range(0, len(toks) - ngrams + 1):
                ngrams_toks.append(' '.join(toks[i:i + ngrams]))
        return ngrams_toks

    def fit(self, datas, fix_vocab=False):
        ''' Transform the data into n-gram vectors. Using csr_matrix to store this sparse matrix.'''
        if not fix_vocab:
            for data in datas:
                toks = self.tokenize(data)
                ngrams_toks = self.get_n_grams(toks)
                self.tok2tf.update(Counter(ngrams_toks))
            self.tok2tf = dict(filter(lambda x: x[1] < self.max_tf * len(datas), self.tok2tf.items()))
            self.tok2id = dict([(k, i) for i, k in enumerate(self.tok2tf.keys())])
        # the column indices for row i are stored in indices[indptr[i]:indptr[i+1]]
        # and their corresponding values are stored in nums[indptr[i]:indptr[i+1]]
        indices = []
        indptr = [0]
        nums = []
        for data in datas:
            toks = self.tokenize(data)
            ngrams_counter = Counter(self.get_n_grams(toks))
            for k, v in ngrams_counter.items():
                if k in self.tok2id:
                    indices.append(self.tok2id[k])
                    nums.append(v)
            indptr.append(len(indices))
        return csr_matrix((nums, indices, indptr), dtype=int, shape=(len(datas), len(self.tok2id)))


def train_test_split(X, Y, shuffle=True):
    '''
    Split data into train set, dev set and test set.
    '''
    assert X.shape[0] == Y.shape[0], "The length of X and Y must be equal."
    len_ = X.shape[0]
    index = np.arange(0, len_)
    if shuffle:
        np.random.shuffle(index)
    train_num = int(0.8 * len_)
    dev_num = int(0.1 * len_)
    test_num = len_ - train_num - dev_num
    return X[index[:train_num]], X[index[train_num:train_num + dev_num]], X[index[-test_num:]], \
           Y[index[:train_num]], Y[index[train_num:train_num + dev_num]], Y[index[-test_num:]]


def minibatch(data, minibatch_idx):
    return data[minibatch_idx] if type(data) in [np.ndarray, csr_matrix] else [data[i] for i in minibatch_idx]


def get_minibatches(data, minibatch_size, shuffle=True):
    """
    Iterates through the provided data one minibatch at at time. You can use this function to
    iterate through data in minibatches as follows:

        for inputs_minibatch in get_minibatches(inputs, minibatch_size):
            ...

    Or with multiple data sources:

        for inputs_minibatch, labels_minibatch in get_minibatches([inputs, labels], minibatch_size):
            ...
    Args:
        data: there are two possible values:
            - a list or numpy array
            - a list where each element is either a list or numpy array
        minibatch_size: the maximum number of items in a minibatch
        shuffle: whether to randomize the order of returned data
    Returns:
        minibatches: the return value depends on data:
            - If data is a list/array it yields the next minibatch of data.
            - If data a list of lists/arrays it returns the next minibatch of each element in the
              list. This can be used to iterate through multiple data sources
              (e.g., features and labels) at the same time.
    """
    list_data = type(data) is list and (type(data[0]) is list or type(data[0]) in [np.ndarray, csr_matrix])
    if list_data:
        data_size = data[0].shape[0] if type(data[0]) in [np.ndarray, csr_matrix] else len(data[0])
    else:
        data_size = data[0].shape[0] if type(data) in [np.ndarray, csr_matrix] else len(data)
    indices = np.arange(data_size)
    if shuffle:
        np.random.shuffle(indices)
    for minibatch_start in np.arange(0, data_size, minibatch_size):
        minibatch_indices = indices[minibatch_start:minibatch_start + minibatch_size]
        yield [minibatch(d, minibatch_indices) for d in data] if list_data \
            else minibatch(data, minibatch_indices)


if __name__ == "__main__":

    train = pd.read_csv(os.path.join(data_path, 'train.tsv'), sep='\t')
    test = pd.read_csv(os.path.join(data_path, 'test.tsv'), sep='\t')

    ngram = Ngram((1, 1))
    X = ngram.fit(train['Phrase'])
    Y = train['Sentiment'].values

    # convert to 2 classes to test our LogisticRegression.
    # Y = train['Sentiment'].apply(lambda x:1 if x>2 else 0).values
    # lr = LogisticRegression(X.shape[1], learning_rate, "l2")

    lr = SoftmaxRegression(X.shape[1], class_num, learning_rate, regularization, C)

    train_X, dev_X, test_X, train_Y, dev_Y, test_Y = train_test_split(X, Y)

    # # Method1: (batch) gradient descent
    # for epoch in range(train_epochs):
    #     train_mle = lr.mle(train_X, train_Y)
    #     print("Epoch %s, Train MLE %.3f" % (epoch, train_mle))
    #     lr.gd(train_X, train_Y)
    #     predict_dev_Y = lr.predict(dev_X)
    #     print("Epoch %s, Dev Acc %.3f" % (epoch, (predict_dev_Y == dev_Y).sum() / len(dev_Y)))

    # # Method2: stochastic gradient descent
    # for epoch in range(train_epochs):
    #     for batch_X, batch_Y in get_minibatches([train_X, train_Y], 1, True):
    #         lr.gd(batch_X, batch_Y)
    #     predict_dev_Y = lr.predict(dev_X)
    #     print("Epoch %s, Dev Acc %.3f" % (epoch, (predict_dev_Y == dev_Y).sum() / len(dev_Y)))

    # Method3: mini-batch gradient descent
    for epoch in range(train_epochs):
        for batch_X, batch_Y in get_minibatches([train_X, train_Y], batch_size, True):
            lr.gd(batch_X, batch_Y)
        predict_dev_Y = lr.predict(dev_X)
        print("Epoch %s, Dev Acc %.3f" % (epoch, (predict_dev_Y == dev_Y).sum() / len(dev_Y)))

    # testing
    predict_test_Y = lr.predict(test_X)
    print("Test Acc %.3f" % ((predict_test_Y == test_Y).sum() / len(test_Y)))

    # predicting
    to_predict_X = ngram.fit(test['Phrase'], fix_vocab=True)
    test['Sentiment'] = lr.predict(to_predict_X)
    test[['Sentiment', 'PhraseId']].set_index('PhraseId').to_csv('numpy_based_lr.csv')
