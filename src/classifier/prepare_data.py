import json
import os

import torch
from nltk import sent_tokenize, word_tokenize
import copy
import Constants
from Dict import Dict
import string

total_data = 0

def tokenize(st, sentence_split=None, option=False):
    #TODO: The tokenizer's performance is suboptimal
    if option and (st[-1] in string.punctuation or st[0] == "$"): #to deal with options
        st = st[:-1]
    if len(st) > 0:
        if option and (st[0] in string.punctuation or st[0] == "$"):  # to deal with options
            st = st[0:]
    st = st.replace("<IMG>", "")
    st = st.replace("[KS5UKS5U]", "")
    st = st.replace("[:Z|xx|k.Com]", "")
    st = st.replace("(;)", "")
    ans = []
    for sent in sent_tokenize(st):
        if sentence_split is not None and len(ans) > 0:
            ans += [sentence_split]
        for w in word_tokenize(sent):
            if len(ans) > 0 and (w == "'re" or w == "n't" or w == "'s" or w == "'m" or w == "'" and len(ans[-1]) > 0) and ans[-1] != "_":
                ans[-1] += w
            else:
                ans += [w]
    ans = " ".join(ans).lower()
    if option and ans.find(" ") != -1:
        print(ans)
    return ans

def tokenize_data(dir, data_set, difficulty_set):
    data = []
    for d in difficulty_set:
        new_path = os.path.join(dir, data_set, d)
        for inf in os.listdir(new_path):
            inf_path = os.path.join(new_path, inf)
            obj = json.load(open(inf_path, "r"))
            obj["article"] = tokenize(obj["article"])
            for k in range(len(obj['options'])):
                for i in range(4):
                    obj['options'][k][i] = tokenize(obj['options'][k][i], option=True)
            data += [obj]
    return data

def get_data(dir, data_set, difficulty_set, vocab=None):
    global total_data
    data = []
    total_data = 0
    print(data_set, difficulty_set)
    for d in difficulty_set:
        new_path = os.path.join(dir, data_set, d)
        for inf in os.listdir(new_path):
            total_data += 1
            inf_path = os.path.join(new_path, inf)
            obj = json.load(open(inf_path, "r"))
            obj["article"] = obj["article"].replace(".", " . ")
            obj["article"] = tokenize(obj["article"])
            article_sentence_split = tokenize(obj["article"], "////").split("////")
            place_holder_sentences = []
            for st in article_sentence_split:
                for k in range(st.count("_")):
                    place_holder_sentences += [st.strip()]
            words = obj["article"].split()
            if vocab:
                obj["article"] = vocab.convertToIdx(words, Constants.UNK_WORD, Constants.EOS_WORD, Constants.BOS_WORD)
            words = vocab.convertToLabels(obj["article"], )
            options = copy.deepcopy(obj['options'])
            for k in range(len(obj['answers'])):
                obj["answers"][k] = ord(str(obj["answers"][k])) - ord('A')
            for k in range(len(obj['options'])):
                for i in range(4):
                    options[k][i] = tokenize(options[k][i], option=True)
                    if options[k][i].find(" ") != -1:
                        options[k][i] = options[k][i].replace(" ", "")
                    if vocab:
                        if options[k][i] == "": #convert empty option to unk
                            options[k][i] = "qwer"
                        obj["options"][k][i] = vocab.convertToIdx(options[k][i].split(), Constants.UNK_WORD)
            obj["place_holder_pos"] = []
            for i in range(len(words)):
                if words[i] == "_":
                    obj["place_holder_pos"].append(i)
            assert len(obj["place_holder_pos"]) == len(obj['answers']) == len(obj['options'])
            obj["place_holder_pos"] = torch.LongTensor(obj["place_holder_pos"])
            obj["types"] = []
            for i in range(len(obj["answers"])):
                obj["types"] += [""]
            data += [obj]
    return data

def makeVocabulary(sentences, size=1000000, min_freq=None):
    vocab = Dict([Constants.PAD_WORD, Constants.UNK_WORD, Constants.EOS_WORD, Constants.BOS_WORD])
    for sentence in sentences:
        for word in sentence.split():
            vocab.add(word)
    originalSize = vocab.size()
    if min_freq is not None:
        vocab = vocab.prune_by_freq(min_freq)
    else:
        vocab = vocab.prune(size)
    print('Created dictionary of size %d (pruned from %d)' %
          (vocab.size(), originalSize))
    return vocab

def get_sentences(data):
    sentences = []
    for d in data:
        sentences += [d["article"]]
        for k in range(len(d["options"])):
            for i in range(4):
                sentences += [d["options"][k][i]]
    return sentences

if __name__ == "__main__":
    difficulty_set = ["middle", "high"]
    data_dir = "../../data/CLOTH/"
    output_dir = "../../data/"
    min_freq = 2
    #set vocabulary
    vocab = makeVocabulary(get_sentences(tokenize_data(data_dir, "train", difficulty_set)), min_freq=min_freq)
    vocab.writeFile(os.path.join(output_dir, "dict.txt"))
    dataset = torch.load('../../data/train.pt') #the tokenizer uses random functions, we fix the vocab here
    vocab = dataset["vocab"]
    unigram_dis = dataset["unigram_dis"]
    for i in string.punctuation:
        idx = vocab.lookup(i)
        if idx is not None:
            unigram_dis[idx] = 0
    unigram_dis[vocab.lookup("_")] = 0
    normalized_unigram = unigram_dis.float() * 1. / unigram_dis.sum()
    data = {}
    data_sets = ["train", "valid", "test"]

    for data_set in data_sets:
        if data_set != "test":
            data[data_set] = get_data(data_dir, data_set, difficulty_set, vocab)
        else:
            data["test"] = {}
            for d in difficulty_set:
                data["test"][d] = get_data(data_dir, data_set, [d], vocab)
            data["test"]["whole"] = get_data(data_dir, data_set, difficulty_set, vocab)
    save_data = {'vocab': vocab,
                 'data': data,
                 'unigram_dis': unigram_dis}
    save_name = "train_debug.pt"
    torch.save(save_data, os.path.join(output_dir, save_name))
