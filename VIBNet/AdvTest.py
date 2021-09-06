from cleverhans.tf2.attacks.projected_gradient_descent import projected_gradient_descent
from cleverhans.tf2.attacks.fast_gradient_method import fast_gradient_method
import easydict
import matplotlib.pyplot as plt
import numpy as np
import argparse
import tensorflow as tf
from tensorflow_probability import distributions as ds
from load_dataset import load_data
from easydict import EasyDict


def test_adv(models = [], metrics = [],eps = 0.02, ratio=20, data: EasyDict = EasyDict(train=[], test=[])):
    
    progress_bar_test = tf.keras.utils.Progbar(data.test.shape[0])
    for x,y in data.test:
        y_A = 1
        y_B = 1
        
        for i in range(models):
            x_a = projected_gradient_descent(
                models[i], x, eps, eps/ratio, 50, np.inf, rand_init=np.random.normal(size=1))
            y_a = models[i](x_a)
            metrics[i](np.argmax(y), y_a)
        progress_bar_test.add(x.shape[0])
    result = []

    for m in metrics:
        result.append(m.result()*100)
        m.reset_state()

    return result
        

def AdversarialCompare(PATH, model1, model2, SNR_Filter=list(range(19)), max_eps=1e-3, times=10, ratio=20):

    VIB = tf.keras.models.load_model(model1)
    CNN = tf.keras.models.load_model(model2)
    X_train, X_test, Y_train, Y_test, mods = load_data(PATH, SNR_Filter)
    train_data = (X_train, Y_train)
    test_data = (X_test, Y_test)
    in_shp = list(X_train.shape[1:])

    # print(X_train.shape, X_test.shape)
    # print(Y_train.shape, Y_test.shape)

    test_acc_VIB = tf.metrics.SparseCategoricalAccuracy()
    test_acc_CNN = tf.metrics.SparseCategoricalAccuracy()
    #printing acc of VIB
    test_acc_VIB(np.argmax(Y_test, axis=1), VIB(X_test))
    test_acc_CNN(np.argmax(Y_test, axis=1), CNN(X_test))
    print("VIB accuracy %f" %(test_acc_VIB.result()*100))
    print("CNN accuracy %f" % (test_acc_CNN.result()*100))

    test_acc_CNN.reset_state()
    test_acc_VIB.reset_state()


    #eps = [1e-4, 5*1e-4, 1e-3, 5*1e-3, 8*1e-3,
    #        1e-2, 2*1e-2, 0.03, 0.04, 5*1e-2, 0.65, 8*1e-2, 1e-1, 0.5]
    eps = [0.05]
    OP = []
    OP3 =[]
    OP4 = []

    for __ in eps:
        tmp = []
        for i in range(times):
            X_Adv = projected_gradient_descent(
                VIB, X_test, __, __/ratio, 40, np.inf, rand_init=1.0)
            Y_pred = np.argmax(VIB(X_Adv), axis=1)
            Y_test2 = np.argmax(Y_test, axis=1)
            co = 0
            for i in range(len(X_test)):
                if(Y_test2[i] == Y_pred[i]):
                    co += 1
            print(co/len(X_test))
            tmp.append(co/len(X_test))
        OP.append(np.min(tmp))
        OP3.append(np.max(tmp))

    OP2 = []
    for __ in eps:
        tmp = []
        for i in range(times):
            X_Adv = projected_gradient_descent(
                CNN, X_test, __, __/ratio, 40, np.inf, rand_init=1.0)
            Y_pred = np.argmax(CNN(X_Adv), axis=1)
            Y_test2 = np.argmax(Y_test, axis=1)
            co = 0
            for i in range(len(X_test)):
                if(Y_test2[i] == Y_pred[i]):
                    co += 1
            print(co/len(X_test))
            tmp.append(co/len(X_test))
        OP2.append(np.min(tmp))
        OP4.append(np.max(tmp))

    return eps, OP, OP2, OP3, OP4
