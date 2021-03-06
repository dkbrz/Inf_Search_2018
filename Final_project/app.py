from flask import Flask
from flask import request
from flask import render_template
#from funcs import *


app = Flask(__name__)

import os
import string
import re
from itertools import zip_longest

from gensim.models import Word2Vec, KeyedVectors
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.feature_extraction.text import TfidfVectorizer

from nltk.tokenize import word_tokenize, wordpunct_tokenize
punkt = string.punctuation+'»«–…'

from nltk.corpus import stopwords
stop_words = set(stopwords.words('russian'))

import pymorphy2
morph = pymorphy2.MorphAnalyzer()

from judicial_splitter import splitter

from tqdm import tqdm_notebook as tqdm

import json
import pickle
from scipy.sparse import csr_matrix, load_npz

from heapdict import heapdict
from sklearn.metrics.pairwise import cosine_similarity

import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings("ignore")

from collections import Counter
import math
from collections import defaultdict
from itertools import islice
import datetime

def preprocessing(text, stop=False):
    global wordpunct_tokenize, stop_words, punkt
    #text = re.sub(r"([a-zа-я0-9])(.)([A-ZА-Я0-9])", r"\1\2 \3", text)
    text = word_tokenize(text)
    new_text= []
    for word in text:
        word = word.strip(punkt)
        if word:
            if word in punkt: continue
            elif word.isdigit(): continue
            elif stop and word in stop_words: continue
            else: new_text.append(morph.parse(word)[0].normal_form)
    return new_text
def sigmoid(x):
    return 1/(1+np.exp(-x))

def get_w2v_vectors(text, k=300, prep=False):
    """Получает вектор документа"""
    
    global w2v_model, stop_words, word_tokenize, tfidf
    
    if prep:
        arr_text = preprocessing(text, stop=True)
    else:
        arr_text = text.split()
    n = 0
    vector = np.array([0]*300)
    address = {key:value for key, value in enumerate(tfidf.transform([' '.join(arr_text)]).toarray()[0]) if value != 0}
    for word in set(arr_text):
        if word not in stop_words:
            try:
                weight = address[tfidf.vocabulary_[word]]
                vec = np.array(w2v_model.wv[word]*weight)
                vector = vector + vec
                n += weight
            except:
                continue
    if n > 0: vector = vector / n
    
    return vector


def score_BM25(qf,n) -> float:
    """
    Compute similarity score between search query and documents from collection
    :return: score
    """
    global index, dictionary, term_doc_matrix, doc_length, avgdl, N, k1, b, blength
    score = math.log(1 + (N-n+0.5)/(n+0.5)) * (k1+1)*qf/(qf+blength)
    return score

def get_okapi(query) -> float:
    """
    Compute sim score between search query and all documents in collection
    Collect as pair (doc_id, score)
    :param query: input text
    :return: list of lists with (doc_id, score)
    """
    global index, dictionary, term_doc_matrix, doc_length, avgdl, N, k1, b
    query = query.split()
    result = csr_matrix((1,9947), dtype=int)
    for word in query:
        if word in dictionary:
            idx = dictionary[word]
            n = (term_doc_matrix[idx]>0).sum()
            result += score_BM25(term_doc_matrix[idx].todense(),n)
    if type(result) == csr_matrix:
        return np.zeros((1,9947))
    return result


def _search(query, mode='blend', n=10, p=0.5):
    query = ' '.join(preprocessing(query))
    
    if mode == 'w2v':
        q_w2v = np.array(get_w2v_vectors(query, k=300, prep=False))
        w_d = cosine_similarity([q_w2v], w2v_data)
        result = np.array([max(w_d[0][titles_data[title]]) if title in titles_data else 0 for title in data['name'].values])
    
    elif mode == 'd2v':
        d2v_model.random.seed(23)
        q_d2v = np.array(d2v_model.infer_vector(query.strip().split()))
        w_d = cosine_similarity([q_d2v], d2v_data)
        result = np.array([max(w_d[0][titles_data[title]]) if title in titles_data else 0 for title in data['name'].values])
        
    elif mode == 'okapi':
        result = np.array(get_okapi(query))
    
    elif mode == 'blend':
        q_w2v = np.array(get_w2v_vectors(query, k=300, prep=False))
        d2v_model.random.seed(23)
        q_d2v = np.array(d2v_model.infer_vector(query.strip().split()))

        okapi = np.array(get_okapi(query))

        w_d = p*cosine_similarity([q_w2v], w2v_data)+(1-p)*cosine_similarity([q_d2v], d2v_data)
        w_d = np.array([max(w_d[0][titles_data[title]]) if title in titles_data else 0 for title in data['name'].values])
        w_d = sigmoid(w_d * (1+ np.log(1+okapi)))

        result = 0.5*w_d + 0.3*cosine_similarity([q_w2v], w2v_titles) + 0.2*cosine_similarity([q_w2v], area)
    
    return result

def search(query, n=10, mode='blend',p=0.5):
    result = _search(query, n=10, mode=mode, p=p)[0]
    data2['Поиск'] = result
    return data2.sort_values(['Поиск'], ascending=False)[:n]


@app.route('/',  methods=['GET'])
def index():
    if request.args:
        query = request.args['query']
        if query.strip() == '':
            return render_template('index.html',links='Пустой запрос!')
        p = int(request.args['w2v'])
        try:
            n = int(request.args['n'])
        except:
            n = 10
        mode = request.args['mode']
        if mode == '': mode = 'blend'
        
        t1 = datetime.datetime.now()
        result = search(query, mode=mode, p=p/100, n=n).to_html(index_names=False)
        t1 = datetime.datetime.now() - t1
        result = result.replace("\\n","<br>").replace("\\t"," ").replace('style="text-align: right;"', 'style="text-align: center;"')
        result = result.replace('border="1"', 'style="border: 1px solid black" rules="rows"')
        result = re.sub('<th>[0-9]{0,}</th>','', result)
        return render_template('index.html', links=result, time = str(t1.total_seconds()))
    return render_template('index.html',links='')

if __name__ == '__main__':
    pd.set_option('display.max_colwidth', -1)
    data = pd.read_csv('./data/data.csv', sep='\t', keep_default_na=False)[['name']]
    data2 = pd.read_csv('./data/data2.csv', sep='\t', keep_default_na=False)
    w2v_model = KeyedVectors.load('./data/w2v_model')
    d2v_model = Doc2Vec.load('./data/d2v_1000')
    term_doc_matrix = load_npz('./data/term_doc_matrix.npz')
    
    with open ('./data/tfidf', 'rb') as output:
        tfidf = pickle.load(output)
    
    with open('./data/index.json', 'r') as outfile:
        index = json.load(outfile)

    with open('./data/dictionary.json', 'r') as outfile:
        dictionary = json.load(outfile)

    with open('./data/doc_length.json', 'r') as outfile:
        doc_length = json.load(outfile)

    k1 = 2.0
    b = 0.75
    avgdl = 48.64381220468483
    N = 9947

    doc_length = [doc_length[i]/avgdl for i in sorted(doc_length)]
    blength = k1*(1-b+b*np.array(doc_length))

    with open('./data/w2v_titles.json','r') as f:
        w2v_titles = np.array(json.load(f))

    with open('./data/titles_data.json','r') as f:
        titles_data = json.load(f)
        
    with open('./data/d2v_data.json','r') as f:
        d2v_data = np.array(json.load(f))
        
    with open('./data/area.json','r') as f:
        area = np.array(json.load(f))
        
    with open('./data/w2v_data.json','r') as f:
        w2v_data = np.array(json.load(f))
    app.run(host='localhost', port=5005, debug=True)
    
    

