from codecs import getwriter
from sys import is_finalizing
from bs4 import BeautifulSoup
from collections import Counter
import csv
import os
import math
import json
import nltk
from stemmer import PorterStemmer
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import re
from helper import tokenization
from helper import getWords
from helper import stemming
from helper import getQueries
from helper import getTfIdf
from helper import getCosine
from helper import getBM25
#reading queries from Query_doc and pre-processing 
queryList = getQueries('./AILA_2019_dataset/Query_doc.txt')
totalQueries = len(queryList)
print("step 1 complete")


#reading relevant and non-relevant documents of the queries
file = open('./AILA_2019_dataset/relevance_judgments_priorcases.txt', 'r')
queryRel = []
queryNonRel = []
for i in range(0, totalQueries):
	queryRel.append([])
	queryNonRel.append([])

#filling queryRel and queryNonRel
while(True):
	str1 = file.readline()
	if(str1 == ''):
		break
	str1 = str1.split()
	if(int(str1[3]) == 0):
		queryNonRel[int(str1[0][6:])-1].append(str1[2])
	else:
		queryRel[int(str1[0][6:])-1].append(str1[2])
file.close()
print("step 2 complete")


# reading all casedocs and creating dictionary of words
# docs: ['C1.txt', 'C2.txt', .......]
# allwords: dictionary with key : word and value : no. of docs in which the word appears
# total: dictionary with key : doc_name and value : total no. of words in that document
# tfDocuments : dictionary with key : doc_name and value : a dictionary with (key : word, value : frequency of the word in that document)

docs = os.listdir("./AILA_2019_dataset/Object_casedocs")
allwords = {}
avgDlen = 0
tfDocuments = {}
total = {}
for doc in docs:
	content = getWords(doc)
	avgDlen += len(content)
	cnt = Counter(content)
	total[doc] = len(content)
	tfDocuments[doc] = cnt
	for word in cnt:
		if word not in allwords:
			allwords[word]=1
		else:
			allwords[word]+=1
print("step 3 complete")

avgDlen = avgDlen/len(docs)

# idf : dictionary with key : word and value : idf value of that word
idf = {}
for word in allwords:
	idf[word] = math.log(len(docs)/allwords[word])


# tfIdfDocs : dictionary with key: doc_name, value : tfIdf value dictionary (word, float) of that document
tfIdfDocs = {}
for doc in docs:
	cnt = tfDocuments[doc]
	tf = {}
	for word in cnt:
		tf[word] = cnt[word]/total[doc]
	tfIdf = {}
	for word in tf:
		tfIdf[word] = tf[word]*idf[word]
	tfIdfDocs[doc] = tfIdf

print("step 4 complete")
#got tfIdf of each document in vector form
#find tfIdf of queries
tfIdfQueries = {}
for i in range(0, len(queryList)):
	tfIdfQueries[i] = getTfIdf(allwords, queryList[i], len(docs))

print("step 5 complete")
file = open('./output1.txt', 'w')
for i in range(0, len(queryList)):
	
	getResult = {}
	for doc in docs:
		cosineSimilarity = getCosine(tfIdfQueries[i], tfIdfDocs[doc])
		bm25 = getBM25(tfDocuments[doc], queryList[i], avgDlen, len(docs), len(docs), allwords)
		f = cosineSimilarity*bm25
		getResult[doc] = f
	
	l = []
	for doc in getResult:
		l.append((getResult[doc], doc))
	l.sort(key = lambda x: x[0])
	for j in range(0, len(l)):
		string = str(i+1)+" Q0 "+str(l[len(l)-1-j][0])+" "+str(j+1)+" "+l[len(l)-1-j][1]+" runid1"
		file.write(string+"\n")
file.close()
print("final finished")
