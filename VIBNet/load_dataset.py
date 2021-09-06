import pickle
import matplotlib.pyplot as plt
import numpy as np
import argparse

def to_onehot(yy):
        yy1 = np.zeros([len(yy), max(yy) + 1])
        yy1[np.arange(len(yy)), yy] = 1
        return yy1

def load_data(PATH, SNR_Filter=list(range(21)), fraction=1):
    with open(PATH, 'rb') as xd1:  
        Xd = pickle.load(xd1, encoding='latin1')  # , encoding='latin1'
        snrs, mods = map(lambda j: sorted(
            list(set(map(lambda x: x[j], Xd.keys())))), [1, 0])
    X = []
    lbl = []
    SNR = []
    k = 0
    for mod in mods:
        k += 1
        for snr in snrs:
            SNR.append(snr)
            X.append(Xd[(mod, snr)])
            for i in range(Xd[(mod, snr)].shape[0]):
                lbl.append((mod, snr))
    X = np.vstack(X)
    np.random.seed(2016)  
    n_examples = X.shape[0]
    n_train = n_examples * 0.5  # ĺŻšĺ
    train_idx = np.random.choice(
        range(0, n_examples), size=int(n_train), replace=False)
    test_idx = list(set(range(0, n_examples)) - set(train_idx))  # label
    test_idx2 = []
    X_train = X[train_idx]

    for i in test_idx:
        if SNR[i//1000] in SNR_Filter:
            test_idx2.append(i)
    
    X_test = X[test_idx2]
    trainy = list(map(lambda x: mods.index(lbl[x][0]), train_idx))
    Y_train = to_onehot(trainy)
    Y_test = to_onehot(list(map(lambda x: mods.index(lbl[x][0]), test_idx2)))
    return X_train[:fraction*X_train.shape[0]], X_test[:fraction*X_train.shape[0]], Y_train[:fraction*X_train.shape[0]], Y_test[:fraction*X_train.shape[0]], mods
