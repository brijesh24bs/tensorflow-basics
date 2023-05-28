import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.datasets import mnist
import tensorflow_hub as hub


#============================== Pretrained Model ==============================#
# (x_train, y_train), (x_test, y_test) = mnist.load_data()
# x_train = x_train.reshape(-1, 28 * 28).astype("float32") / 255.0
# x_test = x_test.reshape(-1, 28 * 28).astype("float32") / 255.0
#
#
# base_model = keras.models.load_model('saved_models/')
# base_model.trainable = False
#
# for layer in base_model.layers:
#     assert layer.trainable == False
#     layer.trainable = False
#

# Customizing the model

# base_inputs = base_model.layers[0].input
# base_outputs = base_model.layers[-2].output
# outputs = layers.Dense(10, activation='softmax')(base_outputs)
# new_model = keras.models.Model(inputs=base_inputs, outputs=outputs)
#
# new_model.compile(
#     optimizer=keras.optimizers.Adam(),
#     loss=keras.losses.SparseCategoricalCrossentropy(),
#     metrics=["accuracy"],
# )
# new_model.fit(x_train, y_train, batch_size=64,epochs=2, verbose=2)

#==================================== Pretrained Keras Model ==============================#
# x = tf.random.normal(shape=(5, 299, 299, 3))
# y = tf.constant([0, 1, 2, 3, 4])
#
# base_model = keras.applications.InceptionV3(include_top=True)##include_top=False removes last dense layer
# base_model.trainable = False
#
# base_inputs = base_model.layers[0].input
# base_outputs = base_model.layers[-2].output
# outputs = keras.layers.Dense(5, activation=keras.activations.softmax)(base_outputs)
#
# new_model = keras.Model(inputs=base_inputs, outputs=outputs)
#
# new_model.compile(
#     loss=keras.losses.SparseCategoricalCrossentropy(),
#     optimizer=keras.optimizers.Adam(learning_rate=3e-4),
#     metrics=['accuracy']
# )
# new_model.fit(x, y, epochs=5, verbose=2)



#================================= Tensorflow_hub models ==============================#
x = tf.random.normal(shape=(5,224,224,3))
y = tf.range(5)
url = "https://tfhub.dev/google/imagenet/mobilenet_v3_large_100_224/classification/5"

baseModel = hub.KerasLayer(url, input_shape=(224, 224, 3))
baseModel.trainable = False
model = keras.models.Sequential(
    [
        baseModel,
        layers.Dense(5, activation='softmax')
    ]
)

model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    optimizer=keras.optimizers.Adam(learning_rate=3e-4),
    metrics=['accuracy']
)

model.fit(x, y, epochs=1, verbose=2)