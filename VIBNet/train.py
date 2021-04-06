from sklearn.metrics import cohen_kappa_score, accuracy_score
from sklearn.metrics import confusion_matrix
import sys
import pickle
import matplotlib.pyplot as plt
import numpy as np
import argparse
import tensorflow as tf
from tensorflow_probability import distributions as ds
from VIBNET import VIBNet
def to_onehot(yy):
        yy1 = np.zeros([len(yy), max(yy) + 1])
        yy1[np.arange(len(yy)), yy] = 1
        return yy1

def load_data(PATH):
    with open("RML2016.10a_dict.dat", 'rb') as xd1:  
        Xd = pickle.load(xd1, encoding='latin1')  # , encoding='latin1'
        snrs, mods = map(lambda j: sorted(
            list(set(map(lambda x: x[j], Xd.keys())))), [1, 0])
    X = []
    lbl = []
    SNR = []
    for mod in mods:
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
        if SNR[i//1000] >= 0:
            test_idx2.append(i)
    
    X_test = X[test_idx2]
    
    trainy = list(map(lambda x: mods.index(lbl[x][0]), train_idx))
    Y_train = to_onehot(trainy)
    Y_test = to_onehot(list(map(lambda x: mods.index(lbl[x][0]), test_idx2)))
    return X_train, X_test, Y_train, Y_test,mods


parser = argparse.ArgumentParser()
parser.add_argument('--Path', type=str , help='Path to Dataset')
parser.add_argument('--Beta',type=float, default = 1e-3, help='IB Ratio' )
parser.add_argument('--Prior_Mean',type=float, default = 0, help='Mean of Prior')
parser.add_argument('--Prior_Sigma',type=float, default = 1.0, help='Sigma of Prior')
parser.add_argument("--dropout", type=float, default=0.5, help='DropoutRatio')
args = parser.parse_args()

PATH = args.Path

X_train, X_test, Y_train, Y_test,mods = load_data(PATH)
in_shp = list(X_train.shape[1:])
print(X_train.shape, X_test.shape)
print(Y_train.shape, Y_test.shape)
classes = mods
print(in_shp)
BETA = args.Beta
prior = ds.Normal(args.Prior_Mean, args.Prior_Sigma)
dr = args.dropout
VIB = VIBNet(dr,BETA,classes,prior)
VIB.compile(optimizer=tf.keras.optimizers.Adam(lr=5e-4), metrics=[tf.keras.metrics.CategoricalAccuracy(name='categorical_accuracy')])
history = VIB.fit(X_train,Y_train,validation_data=(X_test, Y_test), epochs=10000, batch_size=1024)
VIB.predict(X_test)
VIB.save("checkpoint")