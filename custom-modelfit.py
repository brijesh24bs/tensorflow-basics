import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'

import tensorflow as tf
from tensorflow import keras
from keras import layers, models
from keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape)

x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype("float32") / 255.0
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype("float32") / 255.0

model = keras.Sequential(
    [
        layers.Input(shape=(28, 28, 1)),
        layers.Conv2D(64, (3, 3), padding="same"),
        layers.ReLU(),
        layers.Conv2D(128, (3, 3), padding="same"),
        layers.ReLU(),
        layers.Flatten(),
        layers.Dense(10),
    ],
    name="model",
)


class CustomFit(models.Model):
    def __init__(self, model):
        super(CustomFit, self).__init__()
        self.model = model

    def train_step(self, data):
        x, y = data

        with tf.GradientTape() as tape:
            y_pred = self.model(x, training=True)
            loss = self.loss(y, y_pred)

            trainable_vars = self.trainable_variables
            gradients = tape.gradient(loss, trainable_vars)

            self.optimizer.apply_gradients(zip(gradients, trainable_vars))
            self.compiled_metrics.update_state(y, y_pred)

        return {"loss":loss, "accuracy": acc_metrics.result()}


acc_metrics = keras.metrics.SparseCategoricalAccuracy(name='accuracy')

training = CustomFit(model)
training.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=keras.optimizers.Adam(learning_rate=3e-4),
    metrics=["accuracy"]
)
training.fit(x_train, y_train, batch_size=64, epochs=2)
training.evaluate(x_test, y_test, batch_size=64)