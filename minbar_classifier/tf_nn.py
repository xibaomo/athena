import tensorflow as tf
import numpy as np

class TFClassifier(object):
    def __init__(self,x_dim,y_dim):
        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(64,activation='relu',input_shape=x_dim),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(y_dim)
        ])
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True)
        self.model.compile(optimizer='adam',loss=loss_fn,metrics=['accuracy'])

        print(self.model.summary())

    def fit(self,x_train,y_train):
        self.model.fit(x_train,y_train,epochs = 100)
        self.model.evaluate(x_train, y_train, verbose = 2)

    def predict(self,x):
        prob_model = tf.keras.Sequential([
            self.model,
            tf.keras.layers.Softmax()
        ])
        raw_y = prob_model(x).numpy()
        y = np.zeros(x.shape[0])
        for i in range(x.shape[0]):
            y[i] = np.argmax(raw_y[i])

        return y