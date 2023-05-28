import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
import pandas as pd
from tensorflow import keras
from keras import layers

directory = "/home/brijesh/PycharmProjects/tensorflow-basics/custom_dataset/data/mnist_images_csv/"
df = pd.read_csv(directory + "train.csv")
file_paths = df["file_name"].values
labels = df["label"].values

ds_train = tf.data.Dataset.from_tensor_slices((file_paths,labels))

def read_img(image_file, label):
    image = tf.io.read_file(directory+image_file)
    image = tf.image.decode_image(image, channels=1, dtype=tf.float32)
    return image, label

def augment(image, label):
    #data augmentation here
    return image, label

ds_train = ds_train.map(read_img).map(augment).batch(2)

###conitnue compile, fit ...
