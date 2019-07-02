import numpy as np


class LogisticRegression():
    ''' Only for two classes classification.'''

    def __init__(self, num_features, learning_rate=0.01, regularization=None, C=1):
        self.w = np.random.uniform(size=num_features)
        self.learning_rate = learning_rate
        self.num_features = num_features
        self.regularization = regularization
        self.C = C

    def _exp_dot(self, x):
        return np.exp(x.dot(self.w))

    def predict(self, x):
        '''
        Return the predicted classes.
        :param x: (batch_size, num_features)
        :return: (batch_size)
        '''
        probs = sigmoid(self._exp_dot(x))
        return (probs > 0.5).astype(np.int)

    def gd(self, x, y):
        '''
        Perform one gradient descent.
        :param x:(batch_size, num_features)
        :param y:(batch_size)
        :return: None
        '''
        probs = sigmoid(self._exp_dot(x))
        gradients = (x.multiply((y - probs).reshape(-1, 1))).sum(0)
        gradients = np.array(gradients.tolist()).reshape(self.num_features)
        if self.regularization == "l2":
            self.w += self.learning_rate * (gradients * self.C - self.w)
        elif self.regularization == "l1":
            self.w += self.learning_rate * (gradients * self.C - np.sign(self.w))
        else:
            self.w += self.learning_rate * gradients

    def mle(self, x, y):
        ''' Return the MLE estimates, log[p(y|x)]'''
        return (y * x.dot(self.w) - np.log(1 + self._exp_dot(x))).sum()


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class SoftmaxRegression():
    ''' Multi-classes classification.'''

    def __init__(self, num_features, num_classes, learning_rate=0.01, regularization=None, C=1):
        self.w = np.random.uniform(size=(num_features, num_classes))
        self.learning_rate = learning_rate
        self.num_features = num_features
        self.num_classes = num_classes
        self.regularization = regularization
        self.C = C

    def predict(self, x):
        '''
        Return the predicted classes.
        :param x: (batch_size, num_features)
        :return: (batch_size)
        '''
        probs = softmax(x.dot(self.w))
        return probs.argmax(-1)

    def gd(self, x, y):
        '''
        Perform one gradient descent.
        :param x:(batch_size, num_features)
        :param y:(batch_size)
        :return: None
        '''
        probs = softmax(x.dot(self.w))
        gradients = x.transpose().dot(to_onehot(y, self.num_classes) - probs)
        if self.regularization == "l2":
            self.w += self.learning_rate * (gradients * self.C - self.w)
        elif self.regularization == "l1":
            self.w += self.learning_rate * (gradients * self.C - np.sign(self.w))
        else:
            self.w += self.learning_rate * gradients

    def mle(self, x, y):
        '''
        Perform the MLE estimation.
        :param x: (batch_size, num_features)
        :param y: (batch_size)
        :return: scalar
        '''
        probs = softmax(x.dot(self.w))
        return (to_onehot(y, self.num_classes) * np.log(probs)).sum()


def softmax(x):
    return np.exp(x) / np.exp(x).sum(-1, keepdims=True)


def to_onehot(x, class_num):
    return np.eye(class_num)[x]
