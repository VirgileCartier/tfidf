import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

documentA = 'the man went out for a walk'
documentB = 'the children sat around the fire'

bagA = documentA.split(' ')
bagB = documentB.split(' ')

uniqueWords = set(bagA).union(set(bagB))

numOfWordsA = dict.fromkeys(uniqueWords, 0)

for words in bagA:
    numOfWordsA[words] += 1

numOfWordsB = dict.fromkeys(uniqueWords, 0)

for words in bagB:
    numOfWordsB[words] += 1

print("mots A : " + '\n')
print(numOfWordsA)
print('\n')
print("mots B : " + '\n')
print(numOfWordsB)

#tf term frequency -> le nombre de foix qu'un mot apparaît dans un document divisé par le nombre total de mots dans ledit documents.

def computeTF(wordDict, bag):
    tfDict = {}
    bagOfWordsCount = len(bag)
    for word, count in wordDict.items():
        tfDict[word] = count / float(bagOfWordsCount)
    return tfDict

tfA = computeTF(numOfWordsA, bagA)
tfB = computeTF(numOfWordsB, bagB)

print("TF A : " + '\n')
print(tfA)
print('\n')
print("TF B : " + '\n')
print(tfB)

#IDF -> inverse data frequency -> logarithme du nombre de documents divisé par le nombre de documents qui contiennent le mot m.
#IDF détermine le poids des mots rare à l'intérieur de chaque document du corpus :

def computeIDF(documents):
    import math

    N = len(documents)

    idfDict = dict.fromkeys(documents[0].keys(),0)

    for document in documents:
        for word, val in document.items():
            if val > 0:
                idfDict[word] += 1

    for word, val in idfDict.items():
        idfDict[word] = math.log(N /float(val))

    return idfDict

idfs = computeIDF([numOfWordsA, numOfWordsB])

print("IDF : " + '\n')
print(idfs)

# TF-IDF, TF x IDF

def computeTFIDF(tfBagOfWords, idfs):
    tfidf = {}
    for word, val in tfBagOfWords.items():
        tfidf[word] = val * idfs[word]
    return tfidf

tfidfA = computeTFIDF(tfA, idfs)
tfidfB = computeTFIDF(tfB, idfs)

df = pd.DataFrame([tfidfA, tfidfB])

print(" TF - IDF " + '\n')
print(df)

# la class proposée par SKLEARN :
vectorizer = TfidfVectorizer()

vectors = vectorizer.fit_transform([documentA,documentB])

features_names = vectorizer.get_feature_names_out()

dense = vectors.todense()

denseliste = dense.tolist()

pddf = pd.DataFrame(denseliste, columns = features_names)

print(" TF - IDF - sklearn " + '\n')
print(pddf)