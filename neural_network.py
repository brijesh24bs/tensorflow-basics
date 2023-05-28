import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
# print(x_train.shape)
# print(x_test.shape)

x_train = tf.reshape(x_train, (-1, 784))
x_train = tf.cast(x_train, dtype=tf.float32)
x_test = tf.reshape(x_test, (-1, 784))
x_test = tf.cast(x_test, dtype=tf.float32)

x_train = x_train/255.0
x_test = x_test/255.0

#Sequential API( Very convenient, not very flexible) one input one output at a time

# model = keras.Sequential(
#     [
#         keras.Input(shape=(784)), ###
#         layers.Dense(512, activation='relu', name='layer_1'),
#         layers.Dense(256, activation='relu'),   #-2
#         layers.Dense(10, activation='softmax')
#     ]
# )

# model = keras.Sequential()
# model.add(layers.Dense(512, activation='relu'))
# print(model.summary())

#extracting specific layer outputs
# model = keras.Model(inputs = model.inputs,
#                     outputs = [model.layers[-2].output])
#
# model = keras.Model(inputs = model.inputs,
#                     outputs = [model.get_layer(name='layer_1').output])
#
# model = keras.Model(inputs= model.inputs,
#                     outputs = [layer.output for layer in model.layers])
# feature = model.predict(x_train)
# for features in feature:
#     print((features.shape))

#Functional API (Very flexible,,,,,handles multiple inputs and outputs)
inputs = keras.Input(shape=(784))
x = layers.Dense(1024, activation='relu')(inputs)
x = layers.Dense(512, activation='relu')(x)
x = layers.Dense(256, activation='relu')(x)
outputs = layers.Dense(10, activation='softmax')(x)

model = keras.Model(inputs=inputs, outputs=outputs)

model.compile(
    loss = keras.losses.SparseCategoricalCrossentropy(from_logits=False),  #from_logits=True specifies that we have not used softmax in the last layer so treat it accordingly...
    optimizer = keras.optimizers.Adam(learning_rate=0.0001),
    metrics=['accuracy'],
)

model.fit(x_train, y_train, batch_size=32, epochs=10, verbose=2)
model.evaluate(x_test, y_test, batch_size=32, verbose=2)
# x_train = tf.convert_to_tensor(x_train, dtype=tf.float16)
# to use only when numpy is used above to reshape the train and test shape


