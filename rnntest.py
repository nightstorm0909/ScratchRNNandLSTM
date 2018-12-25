import numpy as np
import io
from utils import char_to_string
import matplotlib.pyplot as plt
import pickle
from rnnmodel import RNNModel

data_URL = "shakespeare_train.txt"
validation_URL = "shakespeare_valid.txt"

with io.open(data_URL, 'r', encoding = 'utf8') as f:
    text = f.read()

with io.open(validation_URL, 'r', encoding = 'utf8') as f:
    validation_text = f.read()

# Character collection
vocab = set(text)

# Construct character dictionary
vocab_to_int = {c: i for i, c in enumerate(vocab)}
int_to_vocab = dict(enumerate(vocab))

# Encode data, shape = [number of characters]
train_data = np.array([vocab_to_int[c] for c in text], dtype = np.int32)
validation_data = np.array([vocab_to_int[c] for c in validation_text], np.int32)

vocabulary_size = len(vocab)
gen_len = 30

#o, s = model.forward_propagation(train_data[:1000])
#print("Expected Loss for random prediction: ", np.log(vocabulary_size))
#print("Actual Loss: ", model.total_loss(train_data[:1000], train_data[1:1001]))

def train(model, X_train, y_train, learning_rate = 0.005, nepoch = 100, batch_size = 16):
    losses = []
    errors = []
    validation_losses = []
    validation_errors = []
    
    for epoch in range(nepoch):        
        ############ Training
        for i in range(0, len(y_train), batch_size):
            #print('i: ',i)
            model.sgd_step(X_train[i : i + batch_size], y_train[i : i + batch_size], learning_rate)
            print("[INFO] Epoch: {}, batch no.: {}".format(epoch + 1, i / batch_size + 1))
        loss, error = model.total_loss(X_train, y_train)
        losses.append(loss)
        errors.append(error)
        validation_loss, valid_error = model.total_loss(validation_data[:100000], validation_data[1:100001])
        validation_losses.append(validation_loss)
        validation_errors.append(valid_error)
        # adjust the learning rate if loss increases
        if (len(losses) > 1) and losses[-1] > losses[-2]:
            learning_rate *= 0.5
            print('[INFO] Setting learning rate to ', learning_rate)
        
        print("[INFO] Epoch = {}, Loss: {:.5f}, Validation loss: {:.5f}".format(epoch + 1,
                                                                                loss, validation_loss))
        if (epoch % 10) == 0:
            test_text = "JULIET"
            test_data = np.array([vocab_to_int[c] for c in test_text], dtype = np.int32)
            #print(test_data.shape)
            reply = model.generate_chars(test_data, gen_len)
            #print(char_to_string(int_to_vocab, reply))
            with open("rnn.txt", "a+") as f:
                f.write("[INFO] epoch: {}, generated:".format(epoch))
                f.write(char_to_string(int_to_vocab, reply))
                f.write('\n')
        
    return losses, validation_losses, errors, validation_errors

hidden_dim = 40
n_epochs = 100
batch_size = 10

model = RNNModel(vocabulary_size, hidden_dim = hidden_dim, bptt_truncate = batch_size)

losses, validation_losses, errors, validation_errors = train(model, train_data[:1000000], 
                                                             train_data[1:1000001], 
                                                             nepoch = n_epochs, batch_size = batch_size)
PICKLE_FILE = 'rnn_{}h_{}epoch_{}seq.pickle'.format(hidden_dim, n_epochs, batch_size)
with open(PICKLE_FILE, 'wb') as file:
    temp = [losses, validation_losses, errors, validation_errors]
    pickle.dump(temp, file)
    
test_text = "JULIET"
test_data = np.array([vocab_to_int[c] for c in test_text], dtype = np.int32)
reply = model.generate_chars(test_data, 100)
with open("rnn.txt", "a+") as f:
    f.write("Final Input:{}".format(test_text))
    f.write(char_to_string(int_to_vocab, reply))
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
plt.show()