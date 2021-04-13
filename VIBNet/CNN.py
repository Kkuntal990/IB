
import tensorflow
tf = tensorflow

def CNN(dr,classes):

    model = tf.keras.models.Sequential()  
    model.add(tf.keras.Input(shape=(2,128,)))
    model.add(tf.keras.Reshape(target_shape=(2,128,1,)))
    model.add(tf.keras.ZeroPadding2D((0, 2)))
    model.add(tf.keras.Conv2D(256, (1, 3), padding='valid', activation="relu", name="conv1", kernel_initializer='glorot_uniform'))
    model.add(tf.keras.Dropout(dr))
    model.add(tf.keras.ZeroPadding2D((0, 2), data_format="channels_first"))
    model.add(tf.keras.Conv2D(80, (2, 3), padding="valid", activation="relu", name="conv2", kernel_initializer='glorot_uniform'))
    model.add(tf.keras.Dropout(dr))
    model.add(tf.keras.Flatten())
    model.add(tf.keras.Dense(256, activation='relu', kernel_initializer='he_normal', name="dense1"))
    model.add(tf.keras.Dropout(dr))
    model.add(tf.keras.Dense(len(classes), kernel_initializer='he_normal', name="dense2"))
    model.add(tf.keras.Activation('softmax'))
    model.add(tf.keras.Reshape([len(classes)]))
    model.compile(loss='categorical_crossentropy', optimizer='adam',metrics = [tf.keras.metrics.CategoricalAccuracy(name='categorical_accuracy')])
    return model