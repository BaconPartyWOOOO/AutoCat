from keras.layers import Embedding, LSTM, Dense, Conv1D, MaxPooling1D, Dropout, Activation
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras import optimizers
from keras import models
import numpy as np
import random
import pickle
import os

def formout(output):
    data = np.zeros((len(output), 31))
    for i in range(len(output)):
        data[i, output[i - 1]] = 1
    return data

def fix(outtest, dict):
    return np.vectorize(dict.get)(outtest)

def train():
    with open('catdict.pickle', 'rb') as handle:
        catdic = pickle.load(handle)
    if os.path.exists('autocat.h5'):
        model = models.load_model('autocat.h5')
    else:
        model = create()
    training = 0
    while True:
        testing = random.randint(0, 35)
        if training > 83:
            training = 0
        splits = 10
        for i in range(splits):
            data = (np.load('indata' + str(training) + '.npy'), formout(np.load('outdata' + str(training) + '.npy')))
            test = (np.load('intest' + str(testing) + '.npy'), formout(fix(np.load('outtest' + str(testing) + '.npy'), catdic)))
            data = (data[0][int(i * len(data[0]) / splits):int((i + 1) * len(data[0]) / splits),], data[1][int(i * len(data[1]) / splits):int((i + 1) * len(data[1]) / splits),])
            test = (test[0][int(i * len(test[0]) / splits):int((i + 1) * len(test[0]) / splits),], test[1][int(i * len(test[1]) / splits):int((i + 1) * len(test[1]) / splits),])
            model.fit(x=data[0], y=data[1], epochs=1, batch_size=10, validation_data=test)
            print("Saving...", end='\r')
            model.save('autocat.h5')
            print("Saved")
        training += 1

def create():
    model = Sequential()
    model.add(Embedding(25562, 128, input_length=5000, embeddings_regularizer=l2(0.0001), mask_zero=True))
    model.add(LSTM(256, kernel_regularizer = l2(0.0001), return_sequences=True, dropout=0.2, recurrent_dropout=0.2))
    model.add(LSTM(256, kernel_regularizer = l2(0.0001), return_sequences=True, dropout=0.2, recurrent_dropout=0.2))
    model.add(LSTM(128, kernel_regularizer = l2(0.0001), dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(128, kernel_regularizer=l2(0.0001)))
    model.add(BatchNormalization())
    model.add(Activation('tanh'))
    model.add(Dropout(0.2))
    model.add(Dense(31, kernel_regularizer=l2(0.0001)))
    model.add(BatchNormalization())
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizers.Nadam(),
                  metrics=['accuracy'])
    return model

if __name__ == "__main__":
    train()