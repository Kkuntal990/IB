from cleverhans.tf2.attacks.projected_gradient_descent import projected_gradient_descent
from cleverhans.tf2.attacks.fast_gradient_method import fast_gradient_method
import matplotlib.pyplot as plt
import numpy as np
import argparse
import tensorflow as tf
from tensorflow_probability import distributions as ds
from load_dataset import load_data


def AdversarialCompare(PATH, model1, model2, SNR_Filter=list(range(19)), max_eps=1e-3, times=10, ratio=20):

    VIB = tf.keras.models.load_model(model1)
    CNN = tf.keras.models.load_model(model2)
    X_train, X_test, Y_train, Y_test, mods = load_data(PATH, SNR_Filter)
    in_shp = list(X_train.shape[1:])

    print(X_train.shape, X_test.shape)
    print(Y_train.shape, Y_test.shape)

    #printing acc of VIB
    Y_pred = np.argmax(VIB(X_test), axis=1)
    Y_test2 = np.argmax(Y_test, axis=1)
    co = 0
    for i in range(len(X_test)):
        if(Y_test2[i] == Y_pred[i]):
            co += 1
    print(co/len(X_test))

    #getting accuracy of CNN
    Y_pred = np.argmax(CNN(X_test), axis=1)
    Y_test2 = np.argmax(Y_test, axis=1)
    co = 0
    for i in range(len(X_test)):
        if(Y_test2[i] == Y_pred[i]):
            co += 1
    print(co/len(X_test))

    classes = mods
    print(in_shp)
    eps = [1e-4, 5*1e-4, 1e-3, 5*1e-3, 8*1e-3,
            1e-2, 2*1e-2, 0.03, 0.04, 5*1e-2, 0.65, 8*1e-2, 1e-1, 0.5]
    OP = []

    for __ in eps:
        tmp = []
        for i in range(times):
            X_Adv = projected_gradient_descent(
                VIB, X_test, __, __/ratio, 40, np.inf)
            Y_pred = np.argmax(VIB(X_Adv), axis=1)
            Y_test2 = np.argmax(Y_test, axis=1)
            co = 0
            for i in range(len(X_test)):
                if(Y_test2[i] == Y_pred[i]):
                    co += 1
            print(co/len(X_test))
            tmp.append(co/len(X_test))
        OP.append(np.min(tmp))

    OP2 = []
    for __ in eps:
        tmp = []
        for i in range(times):
            X_Adv = projected_gradient_descent(
                CNN, X_test, __, __/ratio, 40, np.inf)
            Y_pred = np.argmax(CNN(X_Adv), axis=1)
            Y_test2 = np.argmax(Y_test, axis=1)
            co = 0
            for i in range(len(X_test)):
                if(Y_test2[i] == Y_pred[i]):
                    co += 1
            print(co/len(X_test))
            tmp.append(co/len(X_test))
        OP2.append(np.min(tmp))

    return eps, OP, OP2
