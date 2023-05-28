import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras
from keras import layers, regularizers
from keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(-1, 28, 28, 1).astype("float32") / 255.0
x_test = x_test.reshape(-1, 28, 28, 1).astype("float32") / 255.0


class CNNBlock(layers.Layer):
    def __init__(self, out_channels, kernel_size=3):
        super().__init__()
        self.conv = layers.Conv2D(out_channels, kernel_size, padding='same')
        self.bn = layers.BatchNormalization()

    def call(self, inputs, training=False):
        x = self.conv(inputs)
        x = self.bn(x, training=training)
        x = tf.nn.relu(x)
        return x


class ResBlock(layers.Layer):
    def __init__(self, channels):
        super().__init__()
        self.cnn1 = CNNBlock(channels[0])
        self.cnn2 = CNNBlock(channels[1])
        self.cnn3 = CNNBlock(channels[2])
        self.pooling = layers.MaxPool2D()
        self.identity_mapping = layers.Conv2D(channels[1], 3, padding='same')

    def call(self, inputs, training=False):
        x = self.cnn1(inputs, training=training)
        x = self.cnn2(x, training=training)
        x = self.cnn3(
            x + self.identity_mapping(inputs), training=training
        )
        x = self.pooling(x)
        return x


class ResModel(keras.models.Model):
    def __init__(self, num_classes=10):
        super().__init__()
        self.block1 = ResBlock([32, 32, 64])
        self.block2 = ResBlock([64, 64, 128])
        self.block3 = ResBlock([128, 128, 256])
        self.pool = layers.GlobalAveragePooling2D()
        self.classifier = layers.Dense(num_classes)

    def call(self, inputs, training=False):
        x = self.block1(inputs, training=training)
        x = self.block2(x, training=training)
        x = self.block3(x, training=training)
        x = self.pool(x)
        return self.classifier(x)

    def model(self):
        x = keras.Input(shape=(28,28,1))
        return keras.Model(inputs=x, outputs=self.call(x) )


model = ResModel(num_classes=10)

model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=keras.optimizers.Adam(learning_rate=3e-4),
    metrics=['accuracy']
)


model.fit(
    x_train,
    y_train,
    batch_size=64,
    epochs=1,
    verbose=2
)

print(model.summary())
model.evaluate(x_test, y_test, batch_size=64, verbose=2)

