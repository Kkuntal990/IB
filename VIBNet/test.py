from sklearn.metrics import cohen_kappa_score, accuracy_score
from sklearn.metrics import confusion_matrix
from cleverhans.tf2.attacks.projected_gradient_descent import projected_gradient_descent
from cleverhans.tf2.attacks.fast_gradient_method import fast_gradient_method
import pickle
import matplotlib.pyplot as plt
import numpy as np
import argparse
import tensorflow as tf
from tensorflow_probability import distributions as ds


def to_onehot(yy):
        yy1 = np.zeros([len(yy), max(yy) + 1])
        yy1[np.arange(len(yy)), yy] = 1
        return yy1

def load_data(PATH):
    with open(PATH, 'rb') as xd1:  
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
        if args.SNR:

            if SNR[i//1000] == args.SNR:
                test_idx2.append(i)
        else:
            if SNR[i//1000] >= 0:
                test_idx2.append(i)

    
    X_test = X[test_idx2]
    
    trainy = list(map(lambda x: mods.index(lbl[x][0]), train_idx))
    Y_train = to_onehot(trainy)
    Y_test = to_onehot(list(map(lambda x: mods.index(lbl[x][0]), test_idx2)))
    return X_train, X_test, Y_train, Y_test,mods


parser = argparse.ArgumentParser()
parser.add_argument('--Path', type=str , help='Path to Dataset')
parser.add_argument('--SNR',type=float, default = None, help='SNR Filter' )
parser.add_argument('--LoadPath',type=str, default = None, help='Load Model Path' )


args = parser.parse_args()
SNR_filter = args.SNR
PATH = args.Path

X_train, X_test, Y_train, Y_test,mods = load_data(PATH)
in_shp = list(X_train.shape[1:])
print(X_train.shape, X_test.shape)
print(Y_train.shape, Y_test.shape)
classes = mods
print(in_shp)

VIB = tf.keras.models.load_model(args.LoadPath)
eps = np.linspace(0,3e-3,10)
OP = []
for __ in eps:
  X_Adv = fast_gradient_method(VIB, X_test, __, np.inf)
  Y_pred = np.argmax(VIB(X_test),axis=1)
  Y_test2 = np.argmax(Y_test,axis=1)
  co = 0
  for i in range(len(X_test)):
    if(Y_test2[i]==Y_pred[i]):
      co+=1
  print(co)
  OP.append(co/len(X_test))
plt.plot(eps,OP)