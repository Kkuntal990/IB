from cleverhans.tf2.attacks.projected_gradient_descent import projected_gradient_descent
from cleverhans.tf2.attacks.fast_gradient_method import fast_gradient_method
import matplotlib.pyplot as plt
import numpy as np
import argparse
import tensorflow as tf
from tensorflow_probability import distributions as ds
from load_dataset import load_data

def AdversarialCompare(PATH, model1,model2,SNR_Filter=list(range(19)),max_eps=1e-3):
    
        VIB = tf.keras.models.load_model(model1)
        CNN = tf.keras.models.load_model(model2)
        X_train, X_test, Y_train, Y_test, mods = load_data(PATH,SNR_Filter)
        in_shp = list(X_train.shape[1:])
        print(X_train.shape, X_test.shape)
        print(Y_train.shape, Y_test.shape)
        classes = mods
        print(in_shp)
        eps = np.linspace(0,max_eps,10)
        OP = []
      
        for __ in eps:
            X_Adv = fast_gradient_method(VIB, X_test, __, np.inf)
            Y_pred = np.argmax(VIB(X_Adv),axis=1)
            Y_test2 = np.argmax(Y_test,axis=1)
            co = 0
            for i in range(len(X_test)):
                if(Y_test2[i]==Y_pred[i]):
                    co+=1
            print(co)
            OP.append(co/len(X_test))
        OP2 = []
        for __ in eps:
            X_Adv = fast_gradient_method(CNN, X_test, __, np.inf)
            Y_pred = np.argmax(CNN(X_Adv),axis=1)
            Y_test2 = np.argmax(Y_test,axis=1)
            co = 0
            for i in range(len(X_test)):
                if(Y_test2[i]==Y_pred[i]):
                    co+=1
            print(co)
            OP2.append(co/len(X_test))

        return eps,OP,OP2
        
