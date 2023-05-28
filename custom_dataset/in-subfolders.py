import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.preprocessing.image import ImageDataGenerator

img_height = 28
img_width = 28
batch_size= 2

model = keras.Sequential(
    [
        layers.Input((28, 28, 1)),
        layers.Conv2D(16, 3, padding="same"),
        layers.Conv2D(32, 3, padding="same"),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(10),
    ]
)

#                      METHOD 1
# ==================================================== #
#             Using dataset_from_directory             #
# ==================================================== #

# ds_train = tf.keras.preprocessing.image_dataset_from_directory(
#     directory='/home/brijesh/PycharmProjects/tensorflow-basics/custom_dataset/data/mnist_subfolders',
#     labels="inferred",
#     label_mode='int',
#     color_mode="grayscale",
#     batch_size=batch_size,
#     image_size=(img_height, img_width),
#     shuffle=True,
#     seed=123,
#     validation_split=0.1,
#     subset="training",
# )
#
# ds_validation = tf.keras.preprocessing.image_dataset_from_directory(
#     "/home/brijesh/PycharmProjects/tensorflow-basics/custom_dataset/data/mnist_subfolders",
#     labels="inferred",
#     label_mode="int",  # categorical, binary
#     color_mode="grayscale",
#     batch_size=batch_size,
#     image_size=(img_height, img_width),  # reshape if not in this size
#     shuffle=True,
#     seed=123,
#     validation_split=0.1,
#     subset="validation",
# )


# def augment(image, y):
#     image = tf.image.random_brightness(image, max_delta=0.05)
#     return image, y
#
#
# ds_train = ds_train.map(augment)

# model.compile(
#     optimizer=keras.optimizers.Adam(),
#     loss=[keras.losses.SparseCategoricalCrossentropy(from_logits=True),],
#     metrics=["accuracy"],
# )

# model.fit(ds_train, epochs=1, verbose=2)

#                           METHOD 2
# ================================================================== #
#             ImageDataGenerator and flow_from_directory             #
# ================================================================== #

datagen = ImageDataGenerator(
    rescale=1.0/ 255,
    rotation_range=5,
    zoom_range=(0.95, 0.95),
    horizontal_flip=False,
    vertical_flip=False,
    data_format="channels_last",
    validation_split=0.2,
)

train_generator = datagen.flow_from_directory(
    "/home/brijesh/PycharmProjects/tensorflow-basics/custom_dataset/data/mnist_subfolders",
    target_size=(img_height, img_width),
    batch_size=batch_size,
    color_mode="grayscale",
    class_mode="sparse",
    shuffle=True,
    subset="training",
    seed=123,
)

valid_generator = datagen.flow_from_directory(
    "/home/brijesh/PycharmProjects/tensorflow-basics/custom_dataset/data/mnist_subfolders",
    target_size=(img_height, img_width),
    batch_size=batch_size,
    color_mode="grayscale",
    class_mode="sparse",
    shuffle=True,
    subset="validation",
    seed=123,
)

model.compile(
    optimizer=keras.optimizers.Adam(),
    loss=[keras.losses.SparseCategoricalCrossentropy(from_logits=True),],
    metrics=["accuracy"],
)
model.fit(
    train_generator,
    epochs=1,
    steps_per_epoch=20,
    verbose=2,
    # if we had a validation generator:
    validation_data=valid_generator,
    validation_steps=5
)