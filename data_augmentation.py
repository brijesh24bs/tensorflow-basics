import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import  keras
from keras import layers

import tensorflow_datasets as tfds


(ds_train, ds_test), ds_info = tfds.load(
    "cifar10",
    split=['train', 'test'],
    with_info=True,
    as_supervised=True
)

IMG_SIZE = 32
AUTOTUNE = tf.data.AUTOTUNE
BATCH_SIZE = 32


ds_train = ds_train.cache()
ds_train = ds_train.shuffle(ds_info.splits["train"].num_examples)
ds_train = ds_train.batch(BATCH_SIZE)
ds_test = ds_train.batch(BATCH_SIZE)

ds_train = ds_train.prefetch(AUTOTUNE)
ds_test = ds_train.prefetch(AUTOTUNE)

resize_and_rescale = tf.keras.Sequential([
    layers.Resizing(height=IMG_SIZE, width=IMG_SIZE),
    #layers.Rescaling(1.0/127.5, offset=-1) #do this to scale your images between -1 and 1
    layers.Rescaling(1.0/255)
])


data_augmentation = tf.keras.Sequential([
  layers.RandomFlip("horizontal_and_vertical"),
  layers.RandomRotation(0.2),
])

model = keras.Sequential(
    [
        keras.Input(shape=(32,32,3)),
        resize_and_rescale,
        data_augmentation,
        layers.Conv2D(4, 3, padding="same", activation="relu"),
        layers.Conv2D(8, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(16, 3, padding="same", activation="relu"),
        layers.Flatten(),
        layers.Dense(64, activation="relu"),
        layers.Dense(10, activation='softmax'),
    ]
)

model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(),
    optimizer=keras.optimizers.Adam(learning_rate=3e-4),
    metrics=['accuracy']
)

model.fit(ds_train, epochs=5, verbose=2)
model.evaluate(ds_test)

