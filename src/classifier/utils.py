import torch
import Constants
import numpy as np
from torch.autograd import Variable

import sys
class Logger(object):
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def close(self):
        self.log.close()

    def flush(self):
        pass

#from https://github.com/danqi/rc-cnn-dailymail
def gen_embeddings(vocab, ntokens, dim, in_file=None):
    """
        Generate an initial embedding matrix for `word_dict`.
        If an embedding file is not given or a word is not in the embedding file,
        a randomly initialized vector will be used.
    """

    embeddings = np.zeros((ntokens, dim))
    
    if in_file is not None:
        print ('Loading embedding file: %s' % in_file)
        pre_trained = 0
        avg_sigma = 0
        avg_mu = 0
        initialized = {}
        words_set = []
        words_set.append(Constants.UNK_WORD)
        embed_set = []
        embed_set.append(None)
        for line in open(in_file).readlines():
            sp = line.split()
            assert len(sp) == dim + 1
            words_set.append(sp[0])
            embed_set.append([float(x) for x in sp[1:]])
        
        words_set = vocab.convertToIdx(words_set, Constants.UNK_WORD)
        for i, (word) in enumerate(words_set):
            if (word == words_set[0]):
                continue
                
            initialized[word] = True
            pre_trained += 1
            embeddings[word] = embed_set[i]
            mu = embeddings[word].mean()
            sigma = np.std(embeddings[word])
            avg_mu += mu
            avg_sigma += sigma
            
        avg_sigma /= 1. * pre_trained
        avg_mu /= 1. * pre_trained
        
        for i in range(ntokens):
            if i not in initialized:
                embeddings[i] = np.random.normal(avg_mu, avg_sigma, (dim,))
        print ('Pre-trained: %d (%.2f%%)' %
                     (pre_trained, pre_trained * 100.0 / ntokens))
        
    print('finish loading')
    return torch.from_numpy(embeddings).float().cuda()