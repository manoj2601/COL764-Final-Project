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

def getDictionary(text):
	dic = {}
	for word in text:
		if word in dic:
			dic[word] += 1
		else:
			dic[word] = 1
	return dic

def getTfIdf(allwords, text, lenDocs):
	cnt = getDictionary(text)
	tf = {}
	for word in cnt:
		tf[word] = cnt[word]/len(text)
	idf = {}
	for word in cnt:
		if word in allwords:
			idf[word] = math.log(lenDocs/allwords[word])
		else:
			idf[word] = 0
	tfIdf = {}
	for word in tf:
		tfIdf[word] = tf[word]*idf[word]
	return tfIdf

def getBM25(cntDoc, query, avgDlen, lenDoc, totalDocs, allwords):
	cnt = Counter(query)
	fqD = {}
	for word in cnt:
		fqD[word] = cntDoc[word]/lenDoc
	modD = lenDoc
	
	tf = {}
	for word in cnt:
		tf[word] = (fqD[word]*(k+1))/(fqD[word]+k1*(1-b+b*(lenDoc/avgDlen)))

	idf = {}
	for word in cnt:
		idf[word] = math.log(((totalDocs-allwords[word]+0.5)/(allwords[word]+0.5))+1)

	sum = 0
	for word in tf:
		sum += tf[word]*idf[word]
	return sum



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
