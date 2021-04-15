
import tensorflow
import numpy as np
tf = tensorflow

def CNN(dr,classes):

    model = tf.keras.models.Sequential()  
    model.add(tf.keras.Input(shape=(2,128,)))
    model.add(tf.keras.layers.Reshape(target_shape=(2,128,1,)))
    model.add(tf.keras.layers.ZeroPadding2D((0, 2)))
    model.add(tf.keras.layers.Conv2D(256, (1, 3), padding='valid', activation="relu", name="conv1", kernel_initializer='glorot_uniform'))
    model.add(tf.keras.layers.Dropout(dr))
    model.add(tf.keras.layers.ZeroPadding2D((0, 2), data_format="channels_first"))
    model.add(tf.keras.layers.Conv2D(80, (2, 3), padding="valid", activation="relu", name="conv2", kernel_initializer='glorot_uniform'))
    model.add(tf.keras.layers.Dropout(dr))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(256, activation='relu', kernel_initializer='he_normal', name="dense1"))
    model.add(tf.keras.layers.Dropout(dr))
    model.add(tf.keras.layers.Dense(len(classes), kernel_initializer='he_normal', name="dense2"))
    model.add(tf.keras.layers.Activation('softmax'))
    model.add(tf.keras.layers.Reshape([len(classes)]))
    model.compile(loss='categorical_crossentropy', optimizer='adam',metrics = [tf.keras.metrics.CategoricalAccuracy(name='categorical_accuracy')])
    return model

def LSTM(dr, classes):
    input_x = tf.keras.layers.Input(shape=(2, 128,))
    input_x = tf.keras.layers.Reshape(target_shape=(1, 2, 128,))(input_x)
    # input_x_padding = tf.keras.layers.ZeroPadding2D(
    #     (0, 2), data_format="channels_first")(input_x)

    layer11 = tf.keras.layers.Conv2D(50, (1, 8), padding='same', activation="relu", name="conv11", kernel_initializer='glorot_uniform',
                    data_format="channels_first")(input_x)
    layer11 = tf.keras.layers.Dropout(dr)(layer11)

    # layer11_padding = tf.keras.layers.ZeroPadding2D(
    #     (0, 2), data_format="channels_first")(layer11)
    layer12 = tf.keras.layers.Conv2D(50, (1, 8), padding="same", activation="relu", name="conv12", kernel_initializer='glorot_uniform',
                    data_format="channels_first")(layer11)
    layer12 = tf.keras.layers.Dropout(dr)(layer12)

    # layer12 = tf.keras.layers.ZeroPadding2D(
    #     (0, 2), data_format="channels_first")(layer12)
    layer13 = tf.keras.layers.Conv2D(50, (1, 8), padding='same', activation="relu", name="conv13", kernel_initializer='glorot_uniform',
                    data_format="channels_first")(layer12)
    layer13 = tf.keras.layers.Dropout(dr)(layer13)

    # <type 'tuple'>: (None, 50, 2, 242),
    concat = tf.keras.layers.concatenate([layer11, layer13])
    concat_size = list(np.shape(concat))
    input_dim = int(concat_size[-1] * concat_size[-2])
    timesteps = int(concat_size[-3])
    concat = tf.keras.layers.Reshape((timesteps, input_dim))(concat)
    lstm_out = tf.keras.layers.LSTM(
        50, input_dim=input_dim, input_length=timesteps)(concat)

    layer_dense1 = tf.keras.layers.Dense(256, activation='relu',
                        kernel_initializer='he_normal', name="dense1")(lstm_out)
    layer_dropout = tf.keras.layers.Dropout(dr)(layer_dense1)
    layer_dense2 = tf.keras.layers.Dense(len(classes), kernel_initializer='he_normal',
                        name="dense2")(layer_dropout)
    layer_softmax = tf.keras.layers.Activation('softmax')(layer_dense2)
    output = tf.keras.layers.Reshape([len(classes)])(layer_softmax)
    model = tf.keras.Model(input, output)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=[
                  tf.keras.metrics.CategoricalAccuracy(name='categorical_accuracy')])
    return model


