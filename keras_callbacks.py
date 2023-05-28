import os
import matplotlib.pyplot

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from tensorflow import keras
from keras import layers
import tensorflow_datasets as tfds

(ds_train, ds_test), ds_info = tfds.load(
    "mnist",
    split=["train", "test"],
    shuffle_files=True,
    as_supervised=True,  # will return tuple (img, label) otherwise dict
    with_info=True,  # able to get info about dataset
)


def normalize_img(image, label):
    """Normalizes images"""
    return tf.cast(image, tf.float32) / 255.0, label


AUTOTUNE = tf.data.experimental.AUTOTUNE
BATCH_SIZE = 128

#Setup for train datasets
ds_train = ds_train.map(normalize_img, num_parallel_calls=AUTOTUNE)
ds_train = ds_train.cache()
ds_train = ds_train.shuffle(ds_info.splits["train"].num_examples)
ds_train = ds_train.batch(BATCH_SIZE)
ds_train = ds_train.prefetch(AUTOTUNE)

model = keras.Sequential(
    [
        keras.Input(shape=(28, 28, 1)),
        layers.Conv2D(32, 3, activation="relu"),
        layers.Flatten(),
        tf.keras.layers.Dense(10, activation="softmax"),
    ]
)

save_callback = keras.callbacks.ModelCheckpoint(
    filepath='checkpoints/',
    save_weights_only=True,
    monitor='accuracy',
)

#custom scheduler
# def scheduler(epoch):
#     if epoch<5:
#         return 0.001
#     else:
#         return 0.1
#
#
# lr_scheduler = keras.callbacks.LearningRateScheduler(schedule=scheduler,verbose=1)


class MyCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs.get('accuracy') > .95:
            print("Training Stopped")
            self.model.stop_training = True


model.compile(
    optimizer=keras.optimizers.Adam(0.01),
    loss=keras.losses.SparseCategoricalCrossentropy(),
    metrics=["accuracy"],
)

model.fit(
    ds_train,
    epochs=10,
    callbacks=[save_callback, MyCallback()],
    verbose=2,
)