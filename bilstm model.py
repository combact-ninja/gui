import numpy as np
from keras import Sequential
from keras.layers import *
from termcolor import colored
from tensorflow import keras

def train_test_split(feat, lab, percent):
    xtrain, xtest, ytrain, ytest = [], [], [], []
    all = np.unique(lab)
    for i in all:
        index = np.where(i == lab)[0]
        div = int(index.shape[0] * percent)
        x_train = feat[index[:div], :]
        x_test = feat[index[div:], :]
        y_train = lab[index[:div]]
        y_test = lab[index[div:]]
        xtrain.append(x_train), xtest.append(x_test), ytrain.append(y_train), ytest.append(y_test)

    xtrain = np.vstack(xtrain)
    xtest = np.vstack(xtest)
    ytrain = np.hstack(ytrain)
    ytest = np.hstack(ytest)
    return xtrain, xtest, ytrain, ytest



def bilstm(xtrain, xtest, ytrain, ytest):
    xtrain =xtrain.reshape(xtrain.shape[0], xtrain.shape[1], 1)
    print(colored("Bidirectional Long short term Memory >> ", color='blue', on_color='on_grey'))
    m = Sequential()
    m.add(Bidirectional(LSTM(64, return_sequences=True, input_shape=(xtrain.shape[1], xtrain.shape[2]))))
    m.add(Bidirectional(LSTM(32, return_sequences=False, input_shape=(xtrain.shape[1], xtrain.shape[2]))))
    m.add(Dropout(0.5))
    m.add(Dense(16, 'relu'))
    m.add(Dense(8, 'relu'))
    m.add(Dense(ytrain.shape[1], activation='softmax'))
    # compile the model
    m.compile('Adam', 'categorical_crossentropy', 'accuracy')
    # fitting the model to train data
    m.fit(xtrain, ytrain, epochs=2, batch_size=8)
    # preds = m.predict(xtest)
    # pred = np.argmax(preds, axis=1)
    # y_true = np.argmax(ytest, axis=1)
    m.save('proposed_model.h5')


feat = np.load('all_features.npy')
lab = np.load('all_labels.npy')
xtrain, xtest, ytrain, ytest = train_test_split(feat, lab, 0.8)
xtrain = xtrain.astype(np.float32) / xtrain.max()
# Data Categorical
ytrain = keras.utils.to_categorical(ytrain)
bilstm(xtrain, xtest, ytrain, ytest)