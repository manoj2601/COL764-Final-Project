from bs4 import BeautifulSoup
import csv
import os
import json
import nltk
from stemmer import PorterStemmer
import re
from collections import Counter
import math
import numpy as np

stopwords = ['a', 'an', 'the', 'them', 'is', 'are', 'am', 'i', 'he', 'she', 'it', 'they']
def getRelevantDocuments(queryRel, i):
	ret = []
	for j in range(0, len(queryRel[i])):
		ret.append(queryRel[i][j][2])
	return ret

def remove_punctuation(data):
    symbols = "!\"#$%&()*+-./:;<=>?@[\]^_`{|}~\n"
    ret = ""
    for i in range(0, len(data)):
    	if data[i] in symbols:
    		ret+=' '
    	else:
    		ret += data[i] 
    return ret

def stemming(line, stopwords):
	p = PorterStemmer()
	output = []
	for c in line:
		if c == '' or c in stopwords:
			continue
		word = c.strip()
		if word.isalpha():
			word = word.lower()
		stemmed = p.stem(word, 0, len(word)-1)
		stemmed = stemmed.strip()
		if stemmed in stopwords or len(stemmed) < 3:
			continue
		if stemmed != '':
			output.append(stemmed)
		word = ''
	return output

def getQueries(filename):
	file = open(filename, 'r')
	queries = []
	while(True):
		str = file.readline()
		if(str == ""):
			break
		prev = 'A'
		curr = 'A'
		i = 0
		while(True):
			prev = curr
			curr = str[i]
			if(prev == '|' and curr == '|'):
				break
			i+=1
		#pre-processing
		text = remove_punctuation(str[i+1:])
		words = tokenization(text)
		stemmed = stemming(words, stopwords)
		queries.append(stemmed)
	file.close()
	return queries

def tokenization(line):
	return re.split(r'[,.:;"`\'\(\)\{\}\[\] ]+|\n', line)
    
def getWords(doc):
	file = open('./AILA_2019_dataset/Object_casedocs/'+doc, 'r')
	text = file.read()
	text = remove_punctuation(text)
	words = tokenization(text)
	return stemming(words, stopwords)

def getTfIdf(allwords, text, lenDocs):
	cnt = Counter(text)
	tf = {}
	for word in cnt:
		tf[word] = cnt[word]/len(content)
	idf = {}
	for word in cnt:
		idf[word] = math.log(lenDocs/allwords[word])

	tfIdf = {}
	for word in tf:
		tfIdf[word] = tf[word]*idf[word]
	return tfIdf

def getCosine(m1, m2):
	ret = 0
	sqrt1 = 0
	sqrt2 = 0
	for word in m1:
		if(word in m2):
			ret += m1[word]*m2[word]
	for word in m1:
		sqrt1 += m1[word]*m1[word]
	for word in m2:
		sqrt2 += m2[word]*m2[word]
	if(sqrt1 == 0):
		return 0
	if(sqrt2 == 0):
		return 0
	sqrt1 = math.sqrt(sqrt1)
	sqrt2 = math.sqrt(sqrt2)
	return ret/(sqrt1*sqrt2)
