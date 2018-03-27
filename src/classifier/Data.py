import torch
from torch.autograd import Variable

import Constants
import math
import copy
from collections import defaultdict
import json

class BucketIterator(object):
    def __init__(self, data, batchSize, cuda, args, shuffle=True, infor_weighting=0):
        self.data = data
        self.cuda = cuda
        self.numData = len(self.data)
        self.batchSize = batchSize
        self.cacheSize = self.batchSize * 20
        self.numBatches = (self.numData - 1) // batchSize + 1
        self.shuffle = shuffle
        self.other_word_cost = args.other_word_cost
        self.infor_weighting = infor_weighting
        self._reset()

    def _batchify(self, data, align_right=False):
        max_length = max(x.size(0) for x in data)
        out = data[0].new(len(data), max_length).fill_(Constants.PAD)
        for i in range(len(data)):
            data_length = data[i].size(0)
            offset = max_length - data_length if align_right else 0
            out[i].narrow(0, offset, data_length).copy_(data[i])
        out = out.t().contiguous()
        if self.cuda:
            out = out.cuda()
        v = Variable(out)
        return max_length, v

    def _reset(self):
        self.currIdx = 0

        if self.shuffle:
            self.dataOrders = torch.randperm(self.numData)
        else:
            self.dataOrders = torch.LongTensor(range(self.numData))

    def __iter__(self):
        self._reset()
        while True:
            caches = []
            for i in range(self.cacheSize):
                if self.currIdx == self.numData:
                    break
                dataIdx = self.dataOrders[self.currIdx]
                if self.infor_weighting:
                    caches.append((self.data[dataIdx]["article"], self.data[dataIdx]["options"],
                                   self.data[dataIdx]["answers"], self.data[dataIdx]["place_holder_pos"], self.data[dataIdx]["predict_blank"]))
                else:
                    caches.append((self.data[dataIdx]["article"], self.data[dataIdx]["options"],
                                   self.data[dataIdx]["answers"], self.data[dataIdx]["place_holder_pos"], None))
                self.currIdx += 1
            if self.shuffle:
                caches = sorted(caches, key=lambda s: len(s[0]), reverse=True)
            else:
                pass
            batches = []
            for i in range(0, len(caches), self.batchSize):
                batches.append(caches[i:i+self.batchSize])
                
            for batch in batches:
                articles, options, answers, place_holder_pos, predict_blank = zip(*batch)
                max_length, articles = self._batchify(articles)
                if self.infor_weighting:
                    max_length_predict_blank, predict_blank = self._batchify(predict_blank)
                    assert max_length == max_length_predict_blank
                else:
                    predict_blank = None

                new_options = []
                for i, o in enumerate(options):
                    for j, op in enumerate(o):
                        new_options += op
                options = new_options
                _, options = self._batchify(options)

                article_idx = []
                offset = 0
                cnt = 0
                new_place_holder = []
                for i in range(len(place_holder_pos)):
                    for j in range(len(place_holder_pos[i])):
                        new_place_holder.append(place_holder_pos[i][j] + offset)
                        article_idx.append(i)
                        cnt += 1
                    offset += max_length
                place_holder_pos = torch.LongTensor(new_place_holder)
                article_idx = torch.LongTensor(article_idx)
                new_answers = []
                for ans in answers:
                    new_answers += ans
                answers = torch.LongTensor(new_answers)
                
                if self.cuda:
                    answers = answers.cuda()
                    place_holder_pos = place_holder_pos.cuda()
                    article_idx = article_idx.cuda()
                    
                place_holder_pos = Variable(place_holder_pos)
                article_idx = Variable(article_idx)
                answers = Variable(answers)
                
                assert answers.size(0) == place_holder_pos.size(0)
                assert answers.size(0) * 4 == options.size(1)
                
                yield articles, options, answers, place_holder_pos, article_idx, predict_blank

            if self.currIdx == self.numData:
                raise StopIteration

    def __len__(self):
        return self.numBatches
