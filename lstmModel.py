import numpy as np
from utils import softmax, one_hot_encoding

def sigmoid(x):
    return 1. / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def tanh_derivative(x):
    return 1. - x ** 2

# random values in the range [a, b] with shape *args
def rand_arr(a, b, *args):
    #np.random.seed(0)
    return np.random.rand(*args) * (b - a) + a

class LstmParam:
    def __init__(self, mem_cell_ct, x_dim):
        self.mem_cell_ct = mem_cell_ct
        self.x_dim = x_dim
        concat_len = x_dim + mem_cell_ct
        
        # weight matrices
        self.wg = rand_arr(-0.1, 0.1, mem_cell_ct, concat_len)
        self.wi = rand_arr(-0.1, 0.1, mem_cell_ct, concat_len) 
        self.wf = rand_arr(-0.1, 0.1, mem_cell_ct, concat_len)
        self.wo = rand_arr(-0.1, 0.1, mem_cell_ct, concat_len)
        self.wout = rand_arr(-0.1, 0.1, x_dim, mem_cell_ct)
        
        # bias terms
        self.bg = rand_arr(-0.1, 0.1, mem_cell_ct) 
        self.bi = rand_arr(-0.1, 0.1, mem_cell_ct) 
        self.bf = rand_arr(-0.1, 0.1, mem_cell_ct) 
        self.bo = rand_arr(-0.1, 0.1, mem_cell_ct)
        self.bout = rand_arr(-0.1, 0.1, x_dim)
        
        # diffs (derivative of loss function w.r.t. all parameters)
        self.wg_diff = np.zeros((mem_cell_ct, concat_len)) 
        self.wi_diff = np.zeros((mem_cell_ct, concat_len)) 
        self.wf_diff = np.zeros((mem_cell_ct, concat_len)) 
        self.wo_diff = np.zeros((mem_cell_ct, concat_len))
        self.wout_diff = np.zeros((x_dim, mem_cell_ct))
        self.bg_diff = np.zeros(mem_cell_ct) 
        self.bi_diff = np.zeros(mem_cell_ct) 
        self.bf_diff = np.zeros(mem_cell_ct) 
        self.bo_diff = np.zeros(mem_cell_ct)
        self.bout_diff = np.zeros(x_dim)
        
    def apply_diff(self, lr = 0.01):
        self.wg -= lr * self.wg_diff
        self.wi -= lr * self.wi_diff
        self.wf -= lr * self.wf_diff
        self.wo -= lr * self.wo_diff
        self.wout -= lr * self.wout_diff
        self.bg -= lr * self.bg_diff
        self.bi -= lr * self.bi_diff
        self.bf -= lr * self.bf_diff
        self.bo -= lr * self.bo_diff
        self.bout -= lr * self.bout_diff
        
        # reset diffs to zero
        self.wg_diff = np.zeros_like(self.wg)
        self.wi_diff = np.zeros_like(self.wi) 
        self.wf_diff = np.zeros_like(self.wf) 
        self.wo_diff = np.zeros_like(self.wo)
        self.wout_diff = np.zeros_like(self.wout)
        self.bg_diff = np.zeros_like(self.bg)
        self.bi_diff = np.zeros_like(self.bi) 
        self.bf_diff = np.zeros_like(self.bf) 
        self.bo_diff = np.zeros_like(self.bo)
        self.bout_diff = np.zeros_like(self.bout)
        
class LstmState:
    def __init__(self, mem_cell_ct, x_dim):
        self.g = np.zeros(mem_cell_ct)
        self.i = np.zeros(mem_cell_ct)
        self.f = np.zeros(mem_cell_ct)
        self.o = np.zeros(mem_cell_ct)
        self.s = np.zeros(mem_cell_ct)
        self.h = np.zeros(mem_cell_ct)
        self.out = np.zeros((x_dim))
        self.bottom_diff_h = np.zeros_like(self.h)
        self.bottom_diff_s = np.zeros_like(self.s)

class LstmNode:
    def __init__(self, lstm_param, lstm_state):
        self.state = lstm_state
        self.param = lstm_param
        self.xc = None          # input concatanated with recurrent input
        
    def forward(self, x, s_prev = None, h_prev = None):
        if s_prev is None:
            s_prev = np.zeros_like(self.state.s)
        if h_prev is None:
            h_prev = np.zeros_like(self.state.h)
            
        # save data for use in backpropagation
        self.s_prev = s_prev
        self.h_prev = h_prev
        
        # concatanate x(t) and h(t-1)
        xc = np.hstack((x, self.h_prev))
        #print("xc: ", xc.shape, self.param.wg.shape, np.dot(self.param.wg, xc).shape)
        
        self.state.g = np.tanh(np.dot(self.param.wg, xc) + self.param.bg)
        self.state.i = sigmoid(np.dot(self.param.wi, xc) + self.param.bi)
        self.state.f = sigmoid(np.dot(self.param.wf, xc) + self.param.bf)
        self.state.o = sigmoid(np.dot(self.param.wo, xc) + self.param.bo)
        self.state.s = self.state.g * self.state.i + (self.state.f * self.s_prev)
        self.state.h = np.tanh(self.state.s) * self.state.o
        self.state.out = softmax(np.dot(self.param.wout, self.state.h) + self.param.bout)
        
        self.xc = xc
        
    def backprop(self, top_diff_out, top_diff_s, prev_h_diff = None):
        #print(top_diff_out.shape, self.param.wout_diff.shape)
        self.param.wout_diff += np.outer(top_diff_out, self.state.h)
        self.param.bout_diff += top_diff_out
        
        top_diff_h = np.dot(self.param.wout.T ,top_diff_out)
        if prev_h_diff is not None:
            top_diff_h += prev_h_diff
        ds = (self.state.o * top_diff_h * tanh_derivative(np.tanh(self.state.s)))+ top_diff_s
        do = self.state.s * top_diff_h
        di = self.state.g * ds
        dg = self.state.i * ds
        df = self.s_prev * ds
        
        # diffs w.r.t. vector inside sigma / tanh function
        di_input = sigmoid_derivative(self.state.i) * di
        df_input = sigmoid_derivative(self.state.f) * df
        do_input = sigmoid_derivative(self.state.o) * do
        dg_input = tanh_derivative(self.state.g) * dg
        
        # diffs w.r.t. inputs
        self.param.wi_diff += np.outer(di_input, self.xc)
        self.param.wf_diff += np.outer(df_input, self.xc)
        self.param.wo_diff += np.outer(do_input, self.xc)
        self.param.wg_diff += np.outer(dg_input, self.xc)
        self.param.bi_diff += di_input
        self.param.bf_diff += df_input       
        self.param.bo_diff += do_input
        self.param.bg_diff += dg_input
        
        # compute bottom diff
        dxc = np.zeros_like(self.xc)
        dxc += np.dot(self.param.wi.T, di_input)
        dxc += np.dot(self.param.wf.T, df_input)
        dxc += np.dot(self.param.wo.T, do_input)
        dxc += np.dot(self.param.wg.T, dg_input)
        
        # save bottom diffs
        self.state.bottom_diff_s = ds * self.state.f
        self.state.bottom_diff_h = dxc[self.param.x_dim:]
        
class LstmNetwork:
    def __init__(self, lstm_param):
        self.lstm_param = lstm_param
        self.lstm_node_list = []
        self.x_list = []        # input sequence
        
    def y_list_is(self, y_list, loss_layer):
        assert len(y_list) == len(self.x_list)
        idx = len(self.x_list) - 1
        error = 0
        #loss = loss_layer.loss(self.lstm_node_list[idx].state.h, y_list[idx])
        loss = loss_layer.loss(self.lstm_node_list[idx].state.out, y_list[idx])
        #print(np.argmax(self.lstm_node_list[idx].state.out), y_list[idx])
        if np.argmax(self.lstm_node_list[idx].state.out) != y_list[idx]:
            error += 1
        
        ##################### BACKPROPAGATION
        #diff_h = loss_layer.bottom_diff(self.lstm_node_list[idx].state.h, y_list[idx])
        diff_out = loss_layer.bottom_diff(self.lstm_node_list[idx].state.out, y_list[idx])
        diff_s = np.zeros(self.lstm_param.mem_cell_ct)
        self.lstm_node_list[idx].backprop(diff_out, diff_s)
        idx -= 1
        
        while idx >= 0:
            #loss += loss_layer.loss(self.lstm_node_list[idx].state.h, y_list[idx])
            loss += loss_layer.loss(self.lstm_node_list[idx].state.out, y_list[idx])
            if np.argmax(self.lstm_node_list[idx].state.out) != y_list[idx]:
                error += 1
            #diff_h = loss_layer.bottom_diff(self.lstm_node_list[idx].state.h, y_list[idx])
            diff_out = loss_layer.bottom_diff(self.lstm_node_list[idx].state.out, y_list[idx])
            #diff_h += self.lstm_node_list[idx + 1].state.bottom_diff_h
            diff_s = self.lstm_node_list[idx + 1].state.bottom_diff_s
            self.lstm_node_list[idx].backprop(diff_out, diff_s,
                                              self.lstm_node_list[idx + 1].state.bottom_diff_h)
            idx -= 1
        return loss, error
    
    def x_list_clear(self):
        self.x_list = []
    
    def x_list_add(self, x):
        self.x_list.append(x)
        if len(self.x_list) > len(self.lstm_node_list):
            # need to add new lstm node, create new state mem
            lstm_state = LstmState(self.lstm_param.mem_cell_ct, self.lstm_param.x_dim)
            self.lstm_node_list.append(LstmNode(self.lstm_param, lstm_state))
        
        ##################### FORWARD PROPAGATION
        # get index of most recent x input
        idx = len(self.x_list) - 1
        if idx == 0:
            # no recurrent inputs yet
            self.lstm_node_list[idx].forward(x)
        else:
            s_prev = self.lstm_node_list[idx - 1].state.s
            h_prev = self.lstm_node_list[idx - 1].state.h
            self.lstm_node_list[idx].forward(x, s_prev, h_prev)
    
    def total_loss(self, x, labels, loss_layer):
        lstm_node = []
        s_prev = None
        h_prev = None
        loss = 0
        error = 0
        for i in range(len(x)):
            lstm_state = LstmState(self.lstm_param.mem_cell_ct, self.lstm_param.x_dim)
            lstm_node = LstmNode(self.lstm_param, lstm_state)
            lstm_node.forward(one_hot_encoding(x[i], self.lstm_param.x_dim), s_prev, h_prev)
            s_prev = lstm_node.state.s
            h_prev = lstm_node.state.h
            loss += loss_layer.loss(lstm_node.state.out, labels[i])
            if np.argmax(lstm_node.state.out) != labels[i]:
                error += 1
        return loss, error
    
    def generate(self, input_text, gen_len):
        s_prev = None
        h_prev = None
        gen = []
        for i in range(len(input_text)):
            lstm_state = LstmState(self.lstm_param.mem_cell_ct, self.lstm_param.x_dim)
            lstm_node = LstmNode(self.lstm_param, lstm_state)
            lstm_node.forward(one_hot_encoding(input_text[i], self.lstm_param.x_dim), s_prev, h_prev)
            s_prev = lstm_node.state.s
            h_prev = lstm_node.state.h
        for i in range(gen_len):
            out = np.argmax(lstm_node.state.out)
            #print(out, lstm_node.state.out[out])
            lstm_state = LstmState(self.lstm_param.mem_cell_ct, self.lstm_param.x_dim)
            lstm_node = LstmNode(self.lstm_param, lstm_state)
            lstm_node.forward(one_hot_encoding(out, self.lstm_param.x_dim), s_prev, h_prev)
            #gen.append(np.argmax(lstm_node.state.out))
            gen.append(np.random.choice(range(self.lstm_param.x_dim), p = lstm_node.state.out.ravel()))
            s_prev = lstm_node.state.s
            h_prev = lstm_node.state.h
        return gen