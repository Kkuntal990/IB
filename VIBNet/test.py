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
from load_dataset import load_data


parser = argparse.ArgumentParser()
parser.add_argument('--Path', type=str , help='Path to Dataset')
parser.add_argument('--SNR',type=float, default = None, help='SNR Filter' )

args = parser.parse_args()
SNR_filter = args.SNR
PATH = args.Path

X_train, X_test, Y_train, Y_test,mod_train, mod_test, mods = load_data(PATH)
in_shp = list(X_train.shape[1:])
print(X_train.shape, X_test.shape)
print(Y_train.shape, Y_test.shape)
classes = mods
print(in_shp)

VIB = tf.keras.models.load_model('checkpoint')
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
