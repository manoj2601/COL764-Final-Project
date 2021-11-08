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
	str = file.readline()
	if(str == ''):
		break
	str = str.split()
	if(int(str[3]) == 0):
		queryNonRel[int(str[0][6:])-1].append(str[2])
	else:
		queryRel[int(str[0][6:])-1].append(str[2])
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


# rochhio method
# alpha = 1
# beta = 0.7
# gamma = 0.1
# # Rocchio Method of Relevance Feedback
# newTfIdf = {}
# for i in range(0, len(queryList)):
# 	Dr = queryRel[i]
# 	Dn = queryNonRel[i]
# 	avgDr = {}
# 	size = len(Dr)
# 	for word in allwords:
# 		sum = 0
# 		for d in Dr:
# 			if(word in tfIdfDocs[d]):
# 				sum += tfIdfDocs[d][word]
# 		avgDr[word] = (sum/size)*beta
	
# 	size = len(Dn)
# 	avgDn = {}
# 	for word in allwords:
# 		sum = 0
# 		for doc in Dn:
# 			if word in tfIdfDocs[doc]:
# 				sum += tfIdfDocs[doc][word]
# 		avgDn[word] = (sum/size)*gamma
	
# 	newTfIdf1 = {}
# 	for word in allwords:
# 		newTfIdf1[word] = 0
# 		if word in tfIdfQueries[i]:
# 			newTfIdf1[word] += alpha*tfIdfQueries[i][word]
# 		if word in avgDr:	
# 			newTfIdf1[word] +=avgDr[word]
# 		if word in avgDn:
# 			newTfIdf1[word] -= avgDn[word]
# 	newTfIdf[i] = newTfIdf1

print("step 5 complete")

#new modified tf Idf of queries created 


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
		file.write(str(i+1)+" Q0 "+str(l[len(l)-1-j][0])+" "+str(j+1)+" "+str(l[len(l)-1-j][1])+" runid1\n")
file.close()
print("final finished")
