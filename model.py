from torch import nn, Tensor
from numpy.random import normal
import tokenutil

class LSTM(nn.Module):
    def __init__(self):
        self.evalmode = False
        self.start_sign = '_'#Tensor(normal(size=(tokenutil.maxlen,1,300)))
        self.maxlen = tokenutil.maxlen
        self.num_layers = 6
        self.hidden_size = 300
        self.ori_state = self._init_ori_state()
        super(LSTM, self).__init__()
        self.drop = nn.Dropout()
        self.word2vec = tokenutil.word2vec
        self.encoder = nn.LSTM(300,self.hidden_size,self.num_layers,dropout=0.5) # out_ch越大, feature越多, linear要处理的越多, 隐状态越大 
        self.decoder = nn.LSTM(300,self.hidden_size,self.num_layers,dropout=0.5)
        self.linear = nn.Linear(self.hidden_size,300)

    def _init_ori_state(self):
        """return normal tensors to initalize hidden state and cell state (h0, c0)"""
        return (Tensor(normal(size=(self.num_layers,1,self.hidden_size))),Tensor(normal(size=(self.num_layers,1,self.hidden_size)))) 
    
    def tokenizer(self, x:str):
        return tokenutil.tokenizer(x)

    def forward(self, x):
        """y has the same shape with tokenized string"""
        if self.evalmode:
            x = tokenutil.tokenizer(x)
        y, (h1, c1) = self.encoder(x, self.ori_state)
        y = tokenutil.vec2str(y)
        y = self.tokenizer(y) # y = self.embedding[y]# y = self.embedding(y)
        y, (_, _) = self.decoder(y, (h1, c1))
        y = self.drop(self.linear(y))
        if self.evalmode:
            y = tokenutil.vec2str(y)
        return y

    def vec2str(self, y):
        return tokenutil.vec2str(y)

       # LSTM(in_ch, out_ch, num_layers)(max_len, batch_size, in_ch, (hidden & cell_state)) 
#   -> output(max_len, batch_size, out_ch), hidden & cell_state(num_layers, batch, out_ch)

if __name__ == '__main__':
    rnn = LSTM()
    input = '我是你爹'#Tensor(randn(100,1,300)) # input中maxlen, batchsize可变, 无影响. input_size是词向量长度, 跟定义走
    #h0 = Tensor(randn(8,1,256))      # D*num_layers, D为1(双向取2), num_layers跟定义走
    #c0 = Tensor(randn(8,1,256))      # batch_size跟input走. hiddensize跟定义走
    output = rnn.forward(input) # maxlen, batch_size跟input定, out_ch经过linear转为词表
    outstr = rnn.vec2str(output)
    instr = rnn.vec2str(rnn.tokenizer('我是你爹'))
    print('outstr='+outstr)
    print('instr='+instr)
    assert rnn.forward('都是发扣税的').shape == rnn.tokenizer('二娃分为').shape
    