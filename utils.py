import numpy as np

def char_to_string(int_to_vocab, indexes):
    return ''.join([int_to_vocab[i] for i in indexes])

def softmax(x):
    xt = np.exp(x - np.max(x))
    return xt / np.sum(xt)

def one_hot_encoding(index, shape):
    x = np.zeros(shape)
    x[index] = 1
    return x