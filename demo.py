# -*- coding: utf-8 -*-
from keras.models import load_model
import numpy as np
import pickle

def tokenize(text):
    tokens = []
    for i in range(len(text)):
        tokens.append(text[i:i + 1])
    return tokens

with open('catdict.pickle', 'rb') as handle:
    catdic = pickle.load(handle)
with open('chardict.pickle', 'rb') as handle:
    chardic = pickle.load(handle)
'''dataArray = np.load('outtest0.npy')

for x in np.nditer(dataArray):
    print(x)'''

print("Loading model...")
model = load_model('autocat.h5')
print('\n')
print("Model Loaded Successfully")
print('\n')
print('**************************************************')
print('* NicoNicoDouga Comment Prediction Demonstration *')
print('**************************************************')
print('\n')
while True:
    comment = input("Enter a comment to predict (Enter D when done):")
    if comment == 'D':
        break
    charList = []
    f = open('test.txt', encoding='utf-8')
    for line in f:
            line = line.replace(":\"", "").split("\"")
            for i in range(len(line)):
                comment = line[i]
                for t in tokenize(comment):
                    if t not in chardic:
                        charList.append(chardic[t])
    inArray = np.array([[0]*(5000 - len(charList)) + charList])
    predictions = model.predict(inArray)
    '''predictionList = np.ndarray.tolist(predictions)
    print(predictionList.index(max(predictionList)))'''
    maxPrediction = np.argmax(predictions)
    invCatDic = {v: k for k, v in catdic.items()}
    print(invCatDic[maxPrediction])



        