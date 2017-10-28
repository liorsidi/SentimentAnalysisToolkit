import csv
import numpy as np
from nltk import RegexpTokenizer, LancasterStemmer

from Evaluation import find_best_words


def get_raw_data(path, categories):
    x_raw = []
    y_raw = []
    with open(path, 'r') as csvfile:
        raw = csv.reader(csvfile, delimiter=',', quotechar='|')
        i=0
        for row in raw:
            if (i > 0):
                x_raw.append(str(row[:-1]))
                y_raw.append(str(row[len(row)-1]))
            i+=1
    return np.array(x_raw), np.array(np.unique(np.array(y_raw), return_inverse=True)[1])


# the following tokenizer forms tokens out of alphabetic sequences, money expressions, and any other non-whitespace sequences
class Tokenizer(object):
    def __init__(self):
        self.tok = RegexpTokenizer('\w+|\$[\d\.]+|\S+')
        self.stemmer = LancasterStemmer()

    def __call__(self, doc):
        return [self.stemmer.stem(token)
                for token in self.tok.tokenize(doc)]


# uses Stemming
class Tokenizer_stemmer(object):
    def __init__(self):
        self.tok = RegexpTokenizer('\w+|\$[\d\.]+|\S+')
        self.stemmer = LancasterStemmer()

    def __call__(self, doc):
        return [self.stemmer.stem(token)
                for token in self.tok.tokenize(doc)]

# evaluate vocabulary selection
def set_vec_vocabulary(old_vec, vocab_list, replace):
    new_vec = deepcopy(old_vec)
    if replace:
        top_words = {}
        i = 0

        for feature in vocab_list:
            top_words[feature] = i
            i += 1
    else:
        i = 0
        top_words = old_vec.vocabulary_

        n = len(top_words)
        for feature in vocab_list:
            if feature not in vocab_list:
                top_words[feature] = n + i
                i += 1

    print(len(top_words))
    new_vec.vocabulary = top_words
    return new_vec


def enrich_words(senti_best_words, w2v_model, topn, senti_word, opp_word, threshold, words_for_sim, all_words):
    enrich_words = {}
    for best_word, sim in senti_best_words.iteritems():
        if sim > threshold:
            most_sim = w2v_model.most_similar(senti_word, topn=topn)
            enrich_words.update(find_best_words(w2v_model, most_sim, senti_word, threshold, words_for_sim, all_words))

            most_sim = w2v_model.most_similar(positive=[best_word, senti_word], negative=[opp_word])
            enrich_words.update(find_best_words(w2v_model, most_sim, senti_word, threshold, words_for_sim, all_words))
    return enrich_words


def union_param(new,prev):
    for k, v in prev.iteritems():
        new[k] = (prev[k],)
    return new



def extract_calc_words_from_file(path,w2v_model,words):
    dic = {}
    with open(path) as f:
        for line in f:
            if line.startswith(';') or line == '\n' :
                continue
            line = line.split()[0]
            if line in w2v_model:
                dic[line] = {}
                for word in words:
                     if word in w2v_model:
                        dic[line][word + '_sim'] = w2v_model.similarity(word,line)
    return dic