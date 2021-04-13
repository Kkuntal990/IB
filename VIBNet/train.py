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
from load_dataset import load_data


parser = argparse.ArgumentParser()
parser.add_argument('--Path', type=str , help='Path to Dataset')
parser.add_argument('--epochs', type=int,default=1000, help='Epochs')
parser.add_argument('--lr',type=float, default = 5e-4, help='Learning Rate' )
parser.add_argument('--BatchSize', type=int,default=1024, help='Batch Size')
parser.add_argument('--Beta',type=float, default = 1e-3, help='IB Ratio' )
parser.add_argument('--Prior_Mean',type=float, default = 0, help='Mean of Prior')
parser.add_argument('--Prior_Sigma',type=float, default = 1.0, help='Sigma of Prior')
parser.add_argument("--dropout", type=float, default=0.5, help='DropoutRatio')
parser.add_argument("--loadpath", type=str, default=None, help='Load Model Path')
parser.add_argument("--savepath", type=str, default="checkpoint", help='Save Model Path')
args = parser.parse_args()

PATH = args.Path

X_train, X_test, Y_train, Y_test, mods = load_data(PATH)
in_shp = list(X_train.shape[1:])
print(X_train.shape, X_test.shape)
print(Y_train.shape, Y_test.shape)
classes = mods
print(in_shp)
BETA = args.Beta
prior = ds.Normal(args.Prior_Mean, args.Prior_Sigma)
dr = args.dropout

if(args.loadpath):
    VIB = tf.keras.models.load_model(args.loadpath)
else:
    VIB = VIBNet(dr,BETA,classes,prior)
    VIB.compile(optimizer=tf.keras.optimizers.Adam(lr=args.lr), metrics=[tf.keras.metrics.CategoricalAccuracy(name='categorical_accuracy')])
history = VIB.fit(X_train,Y_train,validation_data=(X_test, Y_test), epochs=args.epochs, batch_size=args.BatchSize)
VIB.predict(X_test)
VIB.save(args.savepath)