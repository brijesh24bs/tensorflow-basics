import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras
from keras import layers, regularizers
from keras.datasets import cifar10   # 50000 train images, 10000 test images, 32x32x3

(x_train, y_train), (x_test, y_test) = cifar10.load_data() #load data
x_train = x_train.astype("float32")/255.0   #normalize the data between 0 and 1
x_test = x_test.astype("float32")/255.0


#Sequential api
# model = keras.Sequential(
#     [
#         keras.Input(shape=(32,32,3)),
#         keras.layers.Conv2D(32 , 3, padding='valid', activation='relu'),  # 32 out channels, 3x3 filter size, ...
#         keras.layers.MaxPool2D((2,2), strides=1),
#         keras.layers.Conv2D(64, 3, activation='relu'),
#         keras.layers.MaxPool2D((2,2)),
#         keras.layers.Conv2D(128, (3,3), activation='relu'),
#         keras.layers.Flatten(),
#         keras.layers.Dense(64, activation='relu'),
#         keras.layers.Dense(10, activation='softmax')
#     ]
# )


#Functional api
def my_model():
    inputs = keras.Input(shape=(32,32,3))
    x = keras.layers.Conv2D(32, 3, kernel_regularizer=regularizers.l2(0.01))(inputs)
    x = layers.BatchNormalization()(x)
    x = keras.activations.relu(x)
    x = layers.MaxPool2D()(x)

    x = layers.Conv2D(64, 5, kernel_regularizer=regularizers.l2(0.01))(x)
    x = layers.BatchNormalization()(x)
    x = keras.activations.relu(x)
    x = layers.MaxPool2D((2,2), padding='same')(x)

    x = layers.Conv2D(128, 3, kernel_regularizer=regularizers.l2(0.01))(x)
    x = layers.BatchNormalization()(x)
    x = keras.activations.relu(x)
    x = layers.MaxPool2D(pool_size=(2,2), padding='same')(x)

    x = layers.Flatten()(x)     ##Not flatten(x)
    x = layers.Dense(128, kernel_regularizer=regularizers.l2(0.01), activation='relu')(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(10 ,activation='softmax')(x)

    model = keras.Model(inputs=inputs, outputs=outputs)

    return model

model = my_model()
model.compile(
    loss = keras.losses.SparseCategoricalCrossentropy(),
    optimizer=keras.optimizers.Adam(learning_rate=3e-4),
    metrics=['accuracy']
)
model.fit(x_train, y_train, batch_size=64, epochs=50, verbose=2)
model.evaluate(x_test, y_test, batch_size=64 , verbose=2)

