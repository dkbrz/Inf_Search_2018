from flask import Flask
from flask import request
from flask import render_template
#from funcs import *

app = Flask(__name__)

import string
import re

punkt = string.punctuation+'»«–…'

stop_words = ['и',
 'в',
 'во',
 'не',
 'что',
 'он',
 'на',
 'я',
 'с',
 'со',
 'как',
 'а',
 'то',
 'все',
 'она',
 'так',
 'его',
 'но',
 'да',
 'ты',
 'к',
 'у',
 'же',
 'вы',
 'за',
 'бы',
 'по',
 'только',
 'ее',
 'мне',
 'было',
 'вот',
 'от',
 'меня',
 'еще',
 'нет',
 'о',
 'из',
 'ему',
 'теперь',
 'когда',
 'даже',
 'ну',
 'вдруг',
 'ли',
 'если',
 'уже',
 'или',
 'ни',
 'быть',
 'был',
 'него',
 'до',
 'вас',
 'нибудь',
 'опять',
 'уж',
 'вам',
 'ведь',
 'там',
 'потом',
 'себя',
 'ничего',
 'ей',
 'может',
 'они',
 'тут',
 'где',
 'есть',
 'надо',
 'ней',
 'для',
 'мы',
 'тебя',
 'их',
 'чем',
 'была',
 'сам',
 'чтоб',
 'без',
 'будто',
 'чего',
 'раз',
 'тоже',
 'себе',
 'под',
 'будет',
 'ж',
 'тогда',
 'кто',
 'этот',
 'того',
 'потому',
 'этого',
 'какой',
 'совсем',
 'ним',
 'здесь',
 'этом',
 'один',
 'почти',
 'мой',
 'тем',
 'чтобы',
 'нее',
 'сейчас',
 'были',
 'куда',
 'зачем',
 'всех',
 'никогда',
 'можно',
 'при',
 'наконец',
 'два',
 'об',
 'другой',
 'хоть',
 'после',
 'над',
 'больше',
 'тот',
 'через',
 'эти',
 'нас',
 'про',
 'всего',
 'них',
 'какая',
 'много',
 'разве',
 'три',
 'эту',
 'моя',
 'впрочем',
 'хорошо',
 'свою',
 'этой',
 'перед',
 'иногда',
 'лучше',
 'чуть',
 'том',
 'нельзя',
 'такой',
 'им',
 'более',
 'всегда',
 'конечно',
 'всю',
 'между']

import pymorphy2
morph = pymorphy2.MorphAnalyzer()

import json
from scipy.sparse import csr_matrix, load_npz


import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings("ignore")

import math
import datetime

pd.set_option('display.max_colwidth', -1)
data = pd.read_csv('./data/data.csv', sep='\t', keep_default_na=False)[['name']]
data2 = pd.read_csv('./data/data2.csv', sep='\t', keep_default_na=False)
term_doc_matrix = load_npz('./data/term_doc_matrix.npz')

k1 = 2.0
b = 0.75
avgdl = 48.64381220468483
N = 9947

with open('./data/dictionary.json', 'r') as outfile:
    dictionary = json.load(outfile)

with open('./data/doc_length.json', 'r') as outfile:
    doc_length = json.load(outfile)

doc_length = [doc_length[i]/avgdl for i in sorted(doc_length)]
blength = k1*(1-b+b*np.array(doc_length))

def preprocessing(text, stop=False):
    global stop_words, punkt
    #text = re.sub(r"([a-zа-я0-9])(.)([A-ZА-Я0-9])", r"\1\2 \3", text)
    text = text.split()
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
    global dictionary, term_doc_matrix, doc_length, avgdl, N, k1, b
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
    global index, dictionary, term_doc_matrix, doc_length, avgdl, N, k1, b
    query = ' '.join(preprocessing(query))

    if mode == 'okapi':
        result = np.array(get_okapi(query))


    return result

def search(query, n=10, mode='blend',p=0.5):
    result = _search(query, n=10, mode=mode, p=p)[0]
    data2['Поиск'] = result
    return data2.sort_values(['Поиск'], ascending=False)[:n]


@app.route('/',  methods=['GET','POST'])
def index():
    if request.args:
        query = request.args['query']
        if query.strip() == '':
            return render_template('index.html',links='Пустой запрос!')
        try:
            n = int(request.args['n'])
        except:
            n = 10
        mode = 'okapi'

        t1 = datetime.datetime.now()
        result = search(query, mode=mode, p=0, n=n).to_html(index_names=False)
        t1 = datetime.datetime.now() - t1
        result = result.replace("\\n","<br>").replace("\\t"," ").replace('style="text-align: right;"', 'style="text-align: center;"')
        result = result.replace('border="1"', 'style="border: 1px solid black" rules="rows"')
        result = re.sub('<th>[0-9]{0,}</th>','', result)
        return render_template('index.html', links=result, time = str(t1.total_seconds()))
    return render_template('index.html',links='')

if __name__ == '__main__':
    app.run()



