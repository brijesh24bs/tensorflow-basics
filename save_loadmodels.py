import os
import tensorflow as tf

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from tensorflow import keras
from keras import layers
from keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(-1, 28 * 28).astype("float32") / 255.0
x_test = x_test.reshape(-1, 28 * 28).astype("float32") / 255.0

# 1. How to save and load model weights
# 2. Save and loading entire model (Serializing model)
#   - Saves weights
#   - Model architecture
#   - Training Configuration (model.compile())
#   - Optimizer and states

# Using Functional Api
inputs = keras.Input(784)
x = layers.Dense(64, activation="relu")(inputs)
outputs = layers.Dense(10)(x)
model = keras.Model(inputs=inputs, outputs=outputs)

model.load_weights('checkpoint_folder/')
model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=keras.optimizers.Adam(),
    metrics=["accuracy"],
)

model.fit(x_train, y_train, batch_size=32, epochs=2, verbose=2)
model.evaluate(x_test, y_test, batch_size=32, verbose=2)

# model.save_weights('checkpoint_folder/')
# model.save('saved_models/')
