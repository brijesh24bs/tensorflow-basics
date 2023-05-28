import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras
from keras import  layers
from keras.datasets import mnist

(x_train, y_train), (x_test, y_test)  = mnist.load_data()
# print(x_train.shape, x_test.shape)
#(60000, 28, 28)    (10000, 28, 28)

train_images = x_train.reshape(60000, 28, 28, 1)
test_images = x_test.reshape(10000, 28, 28, 1)

x_train = x_train.astype("float32")/255.0
x_test = x_test.astype("float32")/255.0

# Functional API
def create_model():
    inputs = keras.layers.Input(shape=(28,28,1))
    x = layers.Conv2D(32, 5)(inputs)
    x = layers.BatchNormalization()(x)
    x = keras.activations.relu(x)
    x = layers.MaxPool2D(pool_size=(2,2))(x)

    x = layers.Conv2D(64, 3)(x)
    x = layers.BatchNormalization()(x)
    x = keras.activations.relu(x)
    x = layers.MaxPool2D(pool_size=(2,2))(x)

    x = layers.Flatten()(x)
    x = layers.Dense(64, activation='relu')(x)
    outputs = layers.Dense(10, activation='softmax')(x)

    model = keras.Model(inputs=inputs, outputs=outputs)
    return model

model = create_model()

model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(),
    optimizer=keras.optimizers.Adam(learning_rate=3e-1),
    metrics=['accuracy']
)

model.fit(x_train, y_train, batch_size=64 ,epochs=10, verbose=2)
model.evaluate(x_test, y_test, batch_size=64, verbose=2)
