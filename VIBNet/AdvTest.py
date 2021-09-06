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


def test_adv(models = [], metrics = [],ep = 0.02, ratio=20, data = (), times = 10):
    
    progress_bar_test = tf.keras.utils.Progbar(data[0].shape[0])
    print(data[0].shape)
    for x,y in data:
        for i in range(models):
            x_a = projected_gradient_descent(
                models[i], x, ep, ep/ratio, 50, np.inf, rand_init=np.random.normal(size=1))
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
    OP2 =[]


    for __ in eps:
        res = test_adv(models=[VIB, CNN], metrics=[test_acc_VIB, test_acc_CNN], ep =__, ratio=20, data = (X_test, Y_test), times=10)
        print('At \u03B5 = %f \n VIB : %f \n CNN : %f' % (__, res[0], res[1]))
        OP.append(res[0])
        OP2.append(res[1])



    return eps, OP, OP2
