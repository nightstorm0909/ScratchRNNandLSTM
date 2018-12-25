import numpy as np
from utils import softmax, char_to_string

class RNNModel():
    def __init__(self, char_dim, hidden_dim = 100, bptt_truncate = 4):
        self.char_dim = char_dim
        self.hidden_dim = hidden_dim
        self.bptt_truncate = bptt_truncate
        self.s0 = np.zeros((self.hidden_dim))
        self.U = np.random.uniform(-np.sqrt(1. / char_dim), np.sqrt(1. / char_dim), size = (hidden_dim, char_dim))
        self.V = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), size = (char_dim, hidden_dim))
        self.W = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), size = (hidden_dim, hidden_dim))
    
    def forward_propagation(self, x, flag = True):
        if flag:
            T = len(x) # total no of time steps
        else:
            T = 1
        s = np.zeros((T + 1, self.hidden_dim))
        s[-1] = self.s0
        o = np.zeros((T, self.char_dim))    # Output
        
        if flag:
            for t in range(T):
                # indexing U by x[t]. it is the same as multiplying U with a one-hot vector
                s[t] = np.tanh(self.U[:, x[t]] + self.W.dot(s[t-1]))
                o[t] = softmax(self.V.dot(s[t]))
        else:
            for t in range(T):
                s[t] = np.tanh(self.U[:, x] + self.W.dot(s[t-1]))
                o[t] = softmax(self.V.dot(s[t]))
        
        self.s0 = s[T - 1, :]
        #print(self.s0.shape)
        return [o, s]
    
    def bptt(self, x, y):
        # y: targets with indexes of the vocabulary
        T = len(y)
        # perform forward propagation
        o, s = self.forward_propagation(x)
        # Accumulate the gradients in these variables
        dLdU = np.zeros(self.U.shape)
        dLdV = np.zeros(self.V.shape)
        dLdW = np.zeros(self.W.shape)
        delta_o = o        
        #print(T)
        delta_o[np.arange(len(y)), y] -= 1      # y_hat - y
        # for each output backwards
        for t in range(T):
            dLdV += np.outer(delta_o[t], s[t])
            # initial delta calculation
            delta_t = self.V.T.dot(delta_o[t]) * (1 - (s[t] ** 2))  # shape: hidden_dim x 1
            # for given time step t go back from time step t to t-1, t-2 ...
            for bptt_step in np.arange(max(0, t - self.bptt_truncate), t + 1)[::-1]:
                #print("Backpropagation step t = {} bptt step = {}".format(t, bptt_step))
                dLdW += np.outer(delta_t, s[bptt_step - 1])
                dLdU[:, x[bptt_step]] += delta_t
                # update delta for next step
                delta_t *= self.W.T.dot(1 - (s[bptt_step - 1] ** 2))
        return [dLdU, dLdV, dLdW]
    
    def sgd_step(self, x, y, learning_rate):
        dLdU, dLdV, dLdW = self.bptt(x, y)
        self.U -= learning_rate * dLdU
        self.V -= learning_rate * dLdV
        self.W -= learning_rate * dLdW
    
    def predict(self, x):
        o, s = self.forward_propagation(x)
        return np.argmax(o, axis = 1)
    
    def total_loss(self, x, y):
        L = 0       # Loss
        error = 0
        # for each sentence
        o, s = self.forward_propagation(x)
        # we only care about our prediction of the correct chars
        correct_word_predictions =  o[np.arange(len(y)), y]
        # add to the loss based on how off we were
        L += -1 * np.sum(np.log(correct_word_predictions + 0.0001))
        
        out = np.argmax(o, axis = 1)
        for i in range(len(y)):
            if out[i] != y[i]:
                error += 1
        #if np.argmax()
        
        return (L / len(y)), error / len(y)
    
    def generate_chars(self, x, num_chars = 20):
        o, s = self.forward_propagation(x[:-1])
        reply = []
        for i in range(num_chars):    
            o, _ = self.forward_propagation(x[-1], flag = False)
            #reply.append(np.argmax(o, axis = 1)[0])
            reply.append(np.random.choice(range(self.char_dim), p = o.ravel()))
        return reply