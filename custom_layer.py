import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras
from keras import layers, models
from keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train  = x_train.reshape(-1, 28*28).astype("float32")/255.0
x_test = x_test.reshape(-1, 28*28).astype("float32")/255.0


class Dense(layers.Layer):
    def __init__(self, units):
        super().__init__()
        self.units = units

    def  build(self, input_shape):
        self.w = self.add_weight(
            name='w',
            shape=(input_shape[-1], self.units),
            initializer='random_normal',
            trainable=True
        )
        self.b = self.add_weight(
            name='b',
            shape=(self.units,),
            initializer='zeros',
            trainable=True
        )

    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b


class MyRelu(layers.Layer):
    def __init__(self):
        super().__init__()

    def call(self, x):
        return tf.math.maximum(x,0)
class MyModel(models.Model):
    def __init__(self, num_classes):
        super().__init__()
        self.dense1 = Dense(64)
        self.dense2 = Dense(num_classes)
        # self.dense1 = layers.Dense(64)
        # self.dense2 = layers.Dense(num_classes)

    def call(self, inputs, training=False):
        x = self.dense1(inputs)
        x = tf.nn.relu(x)
        x = self.dense2(x)
        return tf.nn.softmax(x)


model = MyModel(10)
model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(),
    optimizer=keras.optimizers.Adam(learning_rate=3e-4),
    metrics=['accuracy']
)

model.fit(x_train, y_train, batch_size=64, epochs=2, verbose=2)
model.evaluate(x_test, y_test, batch_size=64, verbose=2)
