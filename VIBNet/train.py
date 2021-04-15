import argparse
import tensorflow as tf
from tensorflow_probability import distributions as ds
from VIBNET import VIBNet, LSTM_VIB
from load_dataset import load_data
from CNN import CNN

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
parser.add_argument("--model_type", type=str, default="VIB", help='Model Type \n1.VIB - VIBNet \n2.CNN - CNN')
args = parser.parse_args()

PATH = args.Path

X_train, X_test, Y_train, Y_test, mods = load_data(PATH)
print(X_train.shape, X_test.shape)
print(Y_train.shape, Y_test.shape)
classes = mods
BETA = args.Beta
prior = ds.Normal(args.Prior_Mean, args.Prior_Sigma)
dr = args.dropout

if(args.loadpath):
    model = tf.keras.models.load_model(args.loadpath)
else:
    if(args.model_type=="VIB"):
        model = VIBNet(dr,BETA,classes,prior)
        model.compile(optimizer=tf.keras.optimizers.Adam(lr=args.lr), metrics=[tf.keras.metrics.CategoricalAccuracy(name='categorical_accuracy')])
    elif (args.model_type=="LSTM_VIB"):
        model = LSTM_VIB(dr, BETA, classes, prior)
        model.compile(optimizer=tf.keras.optimizers.Adam(lr=args.lr), metrics=[tf.keras.metrics.CategoricalAccuracy(name='categorical_accuracy')])
    else:
        model = CNN(dr,classes)
        model.compile(loss='categorical_crossentropy', optimizer='adam',metrics = [tf.keras.metrics.CategoricalAccuracy(name='categorical_accuracy')])

history = model.fit(X_train,Y_train,validation_data=(X_test, Y_test), epochs=args.epochs, batch_size=args.BatchSize)
model.predict(X_test)
model.save(args.savepath)
