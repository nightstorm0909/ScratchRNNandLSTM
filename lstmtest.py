import numpy as np
from lstmModel import LstmParam, LstmNetwork
import io
from utils import char_to_string, one_hot_encoding
import matplotlib.pyplot as plt
import pickle

class LossLayer:
    @classmethod
    def loss(self, pred, label):
        predictions = pred[label]
        #print(-np.log(predictions), label, predictions)
        return -np.log(predictions + 0.001)
    
    @classmethod
    def bottom_diff(self, pred, label):
        #diff = np.zeros_like(pred)
        #diff[0] = 2 * (pred[0] - label)
        diff = pred
        diff[label] = pred[label] - 1
        return diff

data_URL = "shakespeare_train.txt"
validation_URL = "shakespeare_valid.txt"

with io.open(data_URL, 'r', encoding = 'utf8') as f:
    text = f.read()

with io.open(validation_URL, 'r', encoding = 'utf8') as f:
    validation_text = f.read()

# Character collection
vocab = set(text)
vocabulary_size = len(vocab)

# Num of epochs
n_epochs = 20

# Construct character dictionary
vocab_to_int = {c: i for i, c in enumerate(vocab)}
int_to_vocab = dict(enumerate(vocab))

# Encode data, shape = [number of characters]
train_data = np.array([vocab_to_int[c] for c in text], dtype = np.int32)
validation_data = np.array([vocab_to_int[c] for c in validation_text], np.int32)

# parameters for input data dimension and lstm cell count
mem_cell_ct = 40
x_dim = vocabulary_size

lstm_param = LstmParam(mem_cell_ct, x_dim)
lstm_net = LstmNetwork(lstm_param)

X_train, y_train = train_data[:10000], train_data[1:10001]
X_valid, y_valid = validation_data[:10000], validation_data[1:10001]
learning_rate = 0.001
losses = []
errors = []
validation_losses = []
validation_errors = []

test_text = "JULIET"
test_data = np.array([vocab_to_int[c] for c in test_text], dtype = np.int32)
gen_len = 20
batch_size = 10000
n_batches = 50

for epoch in range(n_epochs):
    loss = 0
    error = 0
    #print()
    for i in range(n_batches):
        idx = i * batch_size
        #print(idx)
        X_train, y_train = train_data[idx: idx+batch_size], train_data[idx+1: (idx+1) + batch_size]
        for ind in range(len(y_train)):
            # forward propagation
            lstm_net.x_list_add(one_hot_encoding(X_train[ind], vocabulary_size))
        
        total_loss, terror = lstm_net.y_list_is(y_train, LossLayer)
        lstm_param.apply_diff(lr = learning_rate)
        loss += total_loss
        error += terror
        print("[INFO] Epoch: ", epoch + 1, "Batch no.: ", i + 1)
        lstm_net.x_list_clear()
    losses.append(loss / (n_batches * batch_size))
    errors.append((error / (n_batches * batch_size)) * 100)
    
    valid_loss, valid_error = lstm_net.total_loss(X_valid, y_valid, LossLayer)
    validation_losses.append(valid_loss / len(y_valid))
    validation_errors.append((valid_error / len(y_valid)) * 100)
    
    print("[INFO] Epoch: ", epoch,"loss: {:.5f}".format(losses[-1]), 
          "validation loss: {:.5f}".format(validation_losses[-1]))
    
    
    if epoch % 10 == 0:
        with open("lstm.txt", "a+") as f:
            f.write("[INFO] epoch: {}, generated:".format(epoch))
            f.write(char_to_string(int_to_vocab, lstm_net.generate(test_data, gen_len)))
            f.write('\n')
    
    if (len(losses) > 1) and losses[-1] > losses[-2]:
        learning_rate *= 0.5
        print("New learning rate: ", learning_rate)
    
    # Backpropagation
    lstm_param.apply_diff(lr = learning_rate)
    lstm_net.x_list_clear()

PICKLE_FILE = 'lstm_{}h_{}epoch.pickle'.format(mem_cell_ct, n_epochs)
with open(PICKLE_FILE, 'wb') as file:
    temp = [losses, validation_losses, errors, validation_errors]
    pickle.dump(temp, file)

with open("lstm.txt", "a+") as f:
    f.write("Final input: {}".format(test_text))
    f.write(char_to_string(int_to_vocab, lstm_net.generate(test_data, 100)))
    f.write('\n')   

fig = plt.figure()
plt.title("Loss")
plt.plot(losses, label = "Training loss")
plt.plot(validation_losses, label = "Validation loss")
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.legend()

fig = plt.figure()
plt.title("Error Rate")
plt.plot(errors, label = "Training error rate")
plt.plot(validation_errors, label = "Validation error rate")
plt.xlabel("Iterations")
plt.ylabel("Error Rate")
plt.legend()

fig = plt.figure()
plt.hist(X_train, vocabulary_size)
plt.xticks(np.arange(len(int_to_vocab)), [int_to_vocab[i] for i in range(len(int_to_vocab))])
plt.show()