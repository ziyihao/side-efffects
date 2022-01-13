#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  5 22:07:37 2021

@author: ziyihao
"""

pip install -U gensim
import os
import collections
from collections import Counter
#genism is the module that specialize in NLP, expecially in topic modeling and word embedding (word2vector)
from gensim.models import Word2Vec #to install gensim, open powershell prompt, and type in: conda install -c anaconda gensim
from gensim.test.utils import get_tmpfile
import csv     #we need this to write results into a csv file.



class MySentences(object):      #this function helps us read all of the txt files in the designated folder
    def __init__(self, dirname):
        self.dirname = dirname

    def __iter__(self):
        for fname in os.listdir(self.dirname):
            for line in open(os.path.join(self.dirname, fname)):
                yield line.split()

def w2v(text):      #this function trains the word2vector model (output) using text documents (input)
    model_w2v = Word2Vec(text)
    return model_w2v

def Convert(lst):     #this function converts a list of list (input) into a dictionary (output)
    it = iter(lst)
    res_dct = dict(it)
    return res_dct

def find_similar(word,model_w2v):   #this function returns the expanded dictionary (most similar words to the seed words)
    word_dict = model_w2v.wv.most_similar(positive=[word],topn=20)   
    d = Convert(word_dict)
    return d

def get_seedword(keyword,dict):
    return dict.get(keyword)


def main():
    curDir = os.getcwd()  
    os.chdir('/Users/ziyihao/Desktop/7039/final project/Typeofeffects') 
    sentences = MySentences('./type')
    model_w2v = w2v(sentences)
    model_w2v.save("word2vec.model")

    #seed words:
    category = {'pain': ['muscle','pain'],
                'fever': ['fever', ],
                'chill': ['chills'],
                'tired': ['tired'],
                'headache': ['headaches','headache'],
                'sore': ['sore','throat','arm','soreness'],
                'swelling': ['arm'],
                'neasea': ['sick']}
    
    keywords = []
    for item in category.keys():
        keywords.append(item)
    keyword_dict = {}
    for keyword in keywords:
        seedword_dict = {}
        seedwords = get_seedword(keyword,category)
        for seedword in seedwords:
            sim_dict = find_similar(seedword,model_w2v)
            seedword_dict.update({seedword: sim_dict})
        keyword_dict.update({keyword: seedword_dict})

    #Remove all the repeated words that do not have the highest probability
    update_dict = dict(keyword_dict) #copy dictionary from above
    for keyword in keyword_dict.keys():
        lst = [] #create list of words for the whole dictionary
        for seedword in keyword_dict[keyword]:
            for item in keyword_dict[keyword][seedword]:
                lst.append(item)
        cnt = Counter(lst) #count the words frequencies
        repeat_item = [x for x, y in cnt.items() if y > 1] #find repeated words
        print(repeat_item)
        for word in repeat_item:
            repeat_dict = {} #create probability dictionary for each word in the form {probability1:word,probability2:word...}
            for seedword in keyword_dict[keyword]:
                if word in list(keyword_dict[keyword][seedword].keys()):
                    repeat_dict.update({keyword_dict[keyword][seedword][word]: word})
            prob = [key for key in repeat_dict.keys()] #get probabilities of all repeated words
            max_prob = max(prob)
            for seedword in keyword_dict[keyword]:
                if word in list(keyword_dict[keyword][seedword].keys()):
                    if max_prob and keyword_dict[keyword][seedword][word] != max_prob:
                        del update_dict[keyword][seedword][word] #delete the words that do not have maximum probability

    with open('updated_dict.csv', 'w') as csv_file:
        csvwriter = csv.writer(csv_file, delimiter='\t')
        for keyword in update_dict:
            for seedword in update_dict[keyword]:
                csvwriter.writerow([keyword, ',', seedword, ',', list(update_dict[keyword][seedword].keys())])


if __name__ == "__main__":
    main()