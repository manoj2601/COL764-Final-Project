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
from helper import getRelevantDocuments
from helper import getCosine

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
tfDocuments = {}
total = {}
for doc in docs:
	content = getWords(doc)
	cnt = Counter(content)
	total[doc] = len(content)
	tfDocuments[doc] = cnt
	for word in cnt:
		if word not in allwords:
			allwords[word]=1
		else:
			allwords[word]+=1
print("step 3 complete")


# idf : dictionary with key : word and value : idf value of that word
idf = {}
for word in allwords:
	idf[word] = math.log(len(docs)/allwords[word])


# tfIdfDocs : dictionary with key: doc_name, value : tfIdf value of that document
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
exit(1)
#got tfIdf of each document in vector form
#find tfIdf of queries
tfIdfQueries = {}
for i in range(0, len(queryList)):
	tfIdfQueries[queryList[i]] = getTfIdf(allwords, queryList[i], len(docs))


# # rochhio method
# alpha = 1
# beta = 0.75
# gamma = 0.15
# #Rocchio Method of Relevance Feedback
# newTfIdf = {}
# for i in range(0, len(queryList)):
# 	Dr = getRelevantDocuments(queryRel, i)
# 	avgDr = {}
# 	size = len(Dr)
# 	for word in allwords:
# 		sum = 0
# 		for d in Dr:
# 			if(word in tfIdfDocs[d]):
# 				sum += tfIdfDocs[d][word]
# 		avgDr[word] = (sum/size)*beta
	
# 	size = len(docs)-size
# 	avgDn = {}
# 	for word in allwords:
# 		sum = 0
# 		for doc in docs:
# 			if doc in Dr:
# 				continue
# 			if word in tfIdfDocs[doc]:
# 				sum += tfIdfDocs[doc][word]
# 		avgDn[word] = (sum/size)*gamma
# 	newTfIdf1 = {}
# 	for word in allwords:
# 		newTfIdf1[word] = 0
# 		if word in tfIdfQueries[queryList[i]]:
# 			newTfIdf1[word] += alpha*tfIdfQueries[queryList[i]][word]
# 		if word in avgDr:	
# 			newTfIdf1[word] +=avgDr[word]
# 		if word in avgDn:
# 			newTfIdf1[word] -= avgDn[word]
# 	newTfIdf[queryList[i]] = newTfIdf1

# print("step 5 complete")

# def compare(item1, item2):
# 	return item1[0] < item2[0]

# file.close()
# #new modified tf Idf of queries created 
# file = open('./output1.txt', 'w')
# for i in range(0, len(queryList)):
# 	newCosines = {}
# 	Dr = getRelevantDocuments(queryRel, i)
# 	for doc in Dr:
# 		newCosines[doc] = getCosine(newTfIdf[queryList[i]], tfIdfDocs[doc])
# 	l = []
# 	for doc in newCosines:
# 		l.append((newCosines[doc], doc))
# 	l.sort(key = lambda x: x[0])
# 	for j in range(0, len(l)):
# 		file.write(str(i+1)+" Q0 "+str(l[len(l)-1-j][0])+" "+str(j+1)+" "+str(l[len(l)-1-j][1])+" runid1\n")
# file.close()
# print("final finished")