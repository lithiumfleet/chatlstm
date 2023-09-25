from gensim.models import KeyedVectors
import numpy as np
from torch import Tensor

word2vec = KeyedVectors.load_word2vec_format('./token_vec_300.bin')# self.embedding = nn.Embedding(20029,300)
maxlen = 30

def tokenizer(x:str):
    """
    adjust input string then convert to vector(maxlen, 1, word_vec_size).
    value of word_vec_size depend on dictionary.
    tokenizer force this model unable to train on batch, so batch loss should be considered in trainning script. 
    """
    x = _to_maxlen(x)
    x = x.replace(' ', '_')
    vec_x = []
    for c in x:
        try:
            vec_x.append(np.array(word2vec[c]).reshape(1,-1))
        except: 
            vec_x.append(np.array(word2vec['_']).reshape(1,-1))
    vec_x = np.stack(vec_x)
    return Tensor(vec_x)


def _to_maxlen(x:str):
    """pad(use space) or truncate input string length to self.maxlen"""
    len_x = len(x)
    if len_x < maxlen:
        x += ' '*(maxlen-len_x)
    else:
        x = x[:maxlen]
    return x

def vec2str(y):
    """
    describe:  reverse process of forward. for sim_c = similar[0][0], temperature=0
    arguments: y: tensor a vector(maxlen,1,word_vec_size)
    return:    a string decoded by dictionary, length = 100
    """
    output = ''
    for c in y:
        arr_c = c[0].detach().numpy()
        sim_c = word2vec.most_similar(arr_c)[0][0]
        if sim_c == '_': sim_c = ' '
        output += sim_c[0][0]
    return output
