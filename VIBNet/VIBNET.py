import tensorflow
tf = tensorflow
from tensorflow_probability import distributions as ds
import math
import numpy as np
class VIBNet(tf.keras.Model):
    def __init__(self,dr,BETA,classes,prior, **kwargs):
        super(VIBNet, self).__init__(**kwargs)

        self.BETA = BETA
        self.prior = prior
        self.classes = classes
        inp = tf.keras.Input(shape=(2,128,))
        x = tf.keras.layers.Reshape(target_shape=(2,128,1,))(inp)
        x = tf.keras.layers.ZeroPadding2D((0,2), data_format="channels_first")(x)
        x = tf.keras.layers.Conv2D(64, (1, 3), padding='valid', activation="relu", name="conv1",kernel_initializer='glorot_uniform')(x)
        x = tf.keras.layers.Dropout(dr)(x)
        x = tf.keras.layers.ZeroPadding2D((0, 2), data_format="channels_first")(x)
        x = tf.keras.layers.Conv2D(256, (2, 3), padding="valid", activation="relu", name="conv2",kernel_initializer='glorot_uniform')(x)
        x = tf.keras.layers.Dropout(dr)(x)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(512, activation='relu', name="mu",kernel_initializer='he_normal')(x)
        '''
        mu = Dense(128, activation='relu', name="mu")(x)
        mu = Dropout(dr)(mu)

        sigma = Dense(128, activation='relu', name="sigma")(x)
        sigma = Dropout(dr)(sigma)
        '''
        mu, sigma = x[:, :256], x[:, 256:]
        #z = keras.layers.Lambda(sample_z, output_shape=(256, ), name='z')([mu, sigma])
        encoder = tf.keras.Model(inp, [mu,sigma], name="encoder")
        self.encoder = encoder
        z = tf.keras.Input(shape=(256,))
        y = tf.keras.layers.Dense(len(classes), name="dense3", activation='relu',kernel_initializer='he_normal')(z)
        decoder = tf.keras.Model(z,y, name="decoder")
        self.decoder = decoder
        self.encoder.summary()
        self.decoder.summary()
        self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
        self.val_loss_tracker = tf.keras.metrics.Mean(name="val_loss")
        self.class_loss_tracker = tf.keras.metrics.Mean(
             name="class_loss"
         )
        self.info_loss_tracker = tf.keras.metrics.Mean(name="info_loss")
        self.IZY_bound_tracker = tf.keras.metrics.Mean(name="IZY_bound")
        self.IZX_bound_tracker = tf.keras.metrics.Mean(name="IZX_bound")
        self.train_acc_tracker = tf.keras.metrics.Mean(name="Training accuracy")
        self.val_acc_tracker = tf.keras.metrics.Mean(name="validation accuracy")

    @property
    def metrics(self):
        return [
            self.class_loss_tracker,
            self.info_loss_tracker,
            self.total_loss_tracker,
            self.IZY_bound_tracker,
            self.IZX_bound_tracker
            ,self.train_acc_tracker
        ]

    def call(self, data):
      mu,sigma = self.encoder(data)
      sigma = tf.math.softplus(sigma-5.0)
      z_dist = ds.Normal(mu, sigma)
      z = z_dist.sample()
      y = self.decoder(z)
      return tf.nn.softmax(y)

    @tf.function
    def train_step(self, inp):
        data, y_true = inp
        with tf.GradientTape() as tape:
            mu,sigma = self.encoder(data)
            sigma = tf.math.softplus(sigma-5.0)
            z_dist = ds.Normal(mu, sigma)
            z = z_dist.sample()
            y = self.decoder(z)
            
          
            info_loss = tf.reduce_sum(tf.reduce_mean(ds.kl_divergence(z_dist, self.prior), 0)) /math.log(2)
            class_loss = tf.compat.v1.losses.softmax_cross_entropy(logits=y, onehot_labels=y_true)
            total_loss =  class_loss + self.BETA*info_loss
        training_acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y, 1), tf.argmax(y_true, axis=1)), tf.float32))
        IZY_bound =  math.log(10, 2) - class_loss
        IZX_bound = info_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.class_loss_tracker.update_state(class_loss)
        self.info_loss_tracker.update_state(info_loss)
        self.total_loss_tracker.update_state(total_loss)
        self.IZY_bound_tracker.update_state(IZY_bound)
        self.IZX_bound_tracker.update_state(IZX_bound)
        self.train_acc_tracker.update_state(training_acc)
        
        return {
            "loss": self.total_loss_tracker.result(),
            "info_loss": self.info_loss_tracker.result(),
            "class_loss": self.class_loss_tracker.result(),
            "IZY_bound" : self.IZY_bound_tracker.result(),
            "IZX_bound" : self.IZX_bound_tracker.result()
            ,"Train acc" : self.train_acc_tracker.result()
        }
    @tf.function
    def test_step(self, val_inp):
        x_val, y_val = val_inp
        mu,sigma = self.encoder(x_val)
        sigma = tf.math.softplus(sigma-5)
        z_dist = ds.Normal(mu, sigma)
        z = z_dist.sample()
        y = self.decoder(z)
        info_loss = tf.reduce_sum(tf.reduce_mean(ds.kl_divergence(z_dist, self.prior), 0)) /math.log(2)
        class_loss = tf.compat.v1.losses.softmax_cross_entropy(logits=y, onehot_labels=y_val)
        val_loss =  class_loss + self.BETA*info_loss
        val_acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(tf.nn.softmax(y), 1), tf.argmax(y_val, axis=1)), tf.float32))
        self.val_loss_tracker.update_state(val_loss)
        self.val_acc_tracker.update_state(val_acc)
        return {
            "acc" : self.val_acc_tracker.result(),
            "loss" : self.val_loss_tracker.result()
        }


class LSTM_VIB(tf.keras.Model):
    def __init__(self, dr, BETA, classes, prior, **kwargs):
        super(LSTM_VIB, self).__init__(**kwargs)

        self.BETA = BETA
        self.prior = prior
        self.classes = classes
        inp = tf.keras.Input(shape=(2, 128,))
        x = tf.keras.layers.Reshape(target_shape=(2, 128, 1,))(inp)
        # x = tf.keras.layers.ZeroPadding2D(
        #     (0, 2))(x)
        print(x.shape)
        x = tf.keras.layers.Conv2D(50, (1, 8), padding='same', activation="relu",
                                   name="conv1", kernel_initializer='glorot_uniform')(x)
        print(x.shape)
        x1 = tf.keras.layers.Dropout(dr)(x)
        # x2 = tf.keras.layers.ZeroPadding2D(
        #     (0, 2))(x1)
        x2 = tf.keras.layers.Conv2D(50, (1, 8), padding="same", activation="relu",
                                    name="conv2", kernel_initializer='glorot_uniform')(x1)
        x2 = tf.keras.layers.Dropout(dr)(x2)
        # x2 = tf.keras.layers.ZeroPadding2D(
        #     (0, 2))(x2)
        x3 = tf.keras.layers.Conv2D(50, (1, 8), padding="same", activation="relu",
                                    name="conv3", kernel_initializer='glorot_uniform')(x2)
        x3 = tf.keras.layers.Dropout(dr)(x3)
        concat = tf.keras.layers.concatenate([x1, x3])
        concat_size = list(np.shape(concat))
        input_dim = int(concat_size[-1] * concat_size[-2])
        timesteps = int(concat_size[-3])
        concat = tf.keras.layers.Reshape((timesteps, input_dim))(concat)
        lstm_out = tf.keras.layers.LSTM(50, input_dim=input_dim,
                                        input_length=timesteps)(concat)
        dense1 = tf.keras.layers.Dense(256, activation='relu',
                                       kernel_initializer='he_normal', name="dense1")(lstm_out)

        '''
        mu = Dense(128, activation='relu', name="mu")(x)
        mu = Dropout(dr)(mu)

        sigma = Dense(128, activation='relu', name="sigma")(x)
        sigma = Dropout(dr)(sigma)
        '''
        mu, sigma = dense1[:, :128], dense1[:, 128:]
        #z = keras.layers.Lambda(sample_z, output_shape=(256, ), name='z')([mu, sigma])
        encoder = tf.keras.Model(inp, [mu, sigma], name="encoder")
        self.encoder = encoder
        z = tf.keras.Input(shape=(128,))
        y = tf.keras.layers.Dense(
            len(classes), name="dense3", activation='relu', kernel_initializer='he_normal')(z)
        decoder = tf.keras.Model(z, y, name="decoder")
        self.decoder = decoder
        self.encoder.summary()
        self.decoder.summary()
        self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
        self.val_loss_tracker = tf.keras.metrics.Mean(name="val_loss")
        self.class_loss_tracker = tf.keras.metrics.Mean(
            name="class_loss"
        )
        self.info_loss_tracker = tf.keras.metrics.Mean(name="info_loss")
        self.IZY_bound_tracker = tf.keras.metrics.Mean(name="IZY_bound")
        self.IZX_bound_tracker = tf.keras.metrics.Mean(name="IZX_bound")
        self.train_acc_tracker = tf.keras.metrics.Mean(
            name="Training accuracy")
        self.val_acc_tracker = tf.keras.metrics.Mean(
            name="validation accuracy")

    @property
    def metrics(self):
        return [
            self.class_loss_tracker,
            self.info_loss_tracker,
            self.total_loss_tracker,
            self.IZY_bound_tracker,
            self.IZX_bound_tracker, self.train_acc_tracker
        ]

    def call(self, data):
      mu, sigma = self.encoder(data)
      sigma = tf.math.softplus(sigma-5.0)
      z_dist = ds.Normal(mu, sigma)
      z = z_dist.sample()
      y = self.decoder(z)
      return tf.nn.softmax(y)

    @tf.function
    def train_step(self, inp):
        data, y_true = inp
        with tf.GradientTape() as tape:
            mu, sigma = self.encoder(data)
            sigma = tf.math.softplus(sigma-5.0)
            z_dist = ds.Normal(mu, sigma)
            z = z_dist.sample()
            y = self.decoder(z)

            info_loss = tf.reduce_sum(tf.reduce_mean(
                ds.kl_divergence(z_dist, self.prior), 0)) / math.log(2)
            class_loss = tf.compat.v1.losses.softmax_cross_entropy(
                logits=y, onehot_labels=y_true)
            total_loss = class_loss + self.BETA*info_loss
        training_acc = tf.reduce_mean(
            tf.cast(tf.equal(tf.argmax(y, 1), tf.argmax(y_true, axis=1)), tf.float32))
        IZY_bound = math.log(10, 2) - class_loss
        IZX_bound = info_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.class_loss_tracker.update_state(class_loss)
        self.info_loss_tracker.update_state(info_loss)
        self.total_loss_tracker.update_state(total_loss)
        self.IZY_bound_tracker.update_state(IZY_bound)
        self.IZX_bound_tracker.update_state(IZX_bound)
        self.train_acc_tracker.update_state(training_acc)

        return {
            "loss": self.total_loss_tracker.result(),
            "info_loss": self.info_loss_tracker.result(),
            "class_loss": self.class_loss_tracker.result(),
            "IZY_bound": self.IZY_bound_tracker.result(),
            "IZX_bound": self.IZX_bound_tracker.result(), "Train acc": self.train_acc_tracker.result()
        }

    @tf.function
    def test_step(self, val_inp):
        x_val, y_val = val_inp
        mu, sigma = self.encoder(x_val)
        sigma = tf.math.softplus(sigma-5)
        z_dist = ds.Normal(mu, sigma)
        z = z_dist.sample()
        y = self.decoder(z)
        info_loss = tf.reduce_sum(tf.reduce_mean(
            ds.kl_divergence(z_dist, self.prior), 0)) / math.log(2)
        class_loss = tf.compat.v1.losses.softmax_cross_entropy(
            logits=y, onehot_labels=y_val)
        val_loss = class_loss + self.BETA*info_loss
        val_acc = tf.reduce_mean(tf.cast(tf.equal(
            tf.argmax(tf.nn.softmax(y), 1), tf.argmax(y_val, axis=1)), tf.float32))
        self.val_loss_tracker.update_state(val_loss)
        self.val_acc_tracker.update_state(val_acc)
        return {
            "acc": self.val_acc_tracker.result(),
            "loss": self.val_loss_tracker.result()
        }
