import numpy as np
import argparse
import tensorflow as tf
from tensorflow_probability import distributions as ds
from load_dataset import load_data


parser = argparse.ArgumentParser()
parser.add_argument('Path', type=str , help='Path to Dataset')
parser.add_argument('--SNR',type=list, default = list(range(19)), help='SNR Filter' )
parser.add_argument('LoadPath',type=str, default = None, help='Load Model Path' )
args = parser.parse_args()

SNR_filter = args.SNR
PATH = args.Path

X_train, X_test, Y_train, Y_test, mods = load_data(PATH,SNR_filter)
in_shp = list(X_train.shape[1:])
print(X_train.shape, X_test.shape)
print(Y_train.shape, Y_test.shape)
classes = mods
print(in_shp)

def TestModel(Model,X_test,Y_test):
  OP = []
  Y_pred = np.argmax(model(X_test),axis=1)
  Y_test2 = np.argmax(Y_test,axis=1)
  co = 0
  for i in range(len(X_test)):
    if(Y_test2[i]==Y_pred[i]):
      co+=1
  return co
  

model = tf.keras.models.load_model(args.LoadPath)
print(TestModel(model,X_test,Y_test))
