import os
import tensorflow as tf
from tensorflow import keras
from keras import layers
import tensorflow_datasets as tfds

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


#===================== mnist dataset ===========================#

# (ds_train, ds_test), ds_info = tfds.load(
#     "mnist",
#     split=["train", "test"],
#     shuffle_files=True,
#     as_supervised=True,
#     with_info=True
# )
#
#
# def normalize_img(image, label):
#     return tf.cast(image, dtype=tf.float32)/255.0, label
#
#
# AUTOTUNE = tf.data.AUTOTUNE
# BATCH_SIZE = 64
# ds_train = ds_train.map(normalize_img, num_parallel_calls=AUTOTUNE)
# ds_train = ds_train.cache()  #keep track of some datasets to load faster next time
# ds_train = ds_train.shuffle(ds_info.splits["train"].num_examples)
# ds_train = ds_train.batch(BATCH_SIZE)
# ds_train = ds_train.prefetch(AUTOTUNE)
#
# ds_test = ds_test.map(normalize_img, num_parallel_calls=AUTOTUNE)
# ds_test = ds_test.batch(128)
# ds_test = ds_test.prefetch(AUTOTUNE)
#
# model = keras.Sequential(
#     [
#         keras.Input((28,28,1)),
#         layers.Conv2D(32, 3, activation='relu'),
#         layers.Flatten(),
#         layers.Dense(10, activation='softmax')
#     ]
# )
# model.compile(
#     loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
#     optimizer=keras.optimizers.Adam(learning_rate=3e-4),
#     metrics=['accuracy']
# )
# model.fit(ds_train, epochs=5, verbose=2)
# model.evaluate(ds_test)


print(tf.__version__)
#=============================== imdb reviews =================================#

#sikhvanu baaki che...thoduk advanced che etle matha mathi gyu