import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,Activation, Dropout, Flatten, Dense
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
# from tqdm import tqdm
from PIL import Image

# To do: add argparse


train_csv = pd.read_csv('/home/sharma.kin/capstone2/Human Action Recognition/Training_set.csv')
train_path = '/home/sharma.kin/capstone2/Human Action Recognition/train/'

img = []

for i in range(len(train_csv)):
    im = train_path + train_csv['filename'].iloc[i]
    temp = Image.open(im)
    img.append(np.array(temp.resize((200,200))))
    
labels = to_categorical(np.asarray(train_csv['label'].factorize()[0]))
images = np.asarray(img)

x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size=0.10, random_state=29, stratify=labels)

del images, labels

data_augmentation = keras.Sequential(
    [layers.RandomFlip("horizontal"), layers.RandomRotation(0.2),
     layers.RandomZoom(0.2), layers.RandomContrast(0.1), 
     layers.RandomTranslation(0.1, 0.1),] 
)

# pretrained = tf.keras.applications.VGG16(input_shape=(200,200,3), 
#                                          include_top=False, 
#                                          weights='imagenet',
#                                          pooling='avg')

# trying resnet50

# pretrained = tf.keras.applications.ResNet50(input_shape=(200,200,3),
#                                             include_top=False,
#                                             weights='imagenet',
#                                             pooling='avg')

# trying xception

# pretrained = tf.keras.applications.Xception(input_shape=(200,200,3),
#                                             include_top=False,
#                                             weights='imagenet',
#                                             pooling='avg')


# trying efficientnetb4

# pretrained = tf.keras.applications.EfficientNetB4(input_shape=(200,200,3),
#                                                   include_top=False,
#                                                   weights='imagenet',
#                                                   pooling='avg')

# trying efficientnetv2m

# pretrained = tf.keras.applications.EfficientNetV2M(input_shape=(200,200,3),
#                                                    include_top=False,
#                                                    weights='imagenet',
#                                                    pooling='avg')

# trying convnextlarge

pretrained = tf.keras.applications.ConvNeXtLarge(input_shape=(200,200,3),
                                                 include_top=False,
                                                 weights='imagenet',
                                                 pooling='avg')

pretrained.trainable = False

inputs = keras.Input(shape=(200,200,3))
# x = data_augmentation(inputs) # uncomment this and comment the next line to use data augmentation
x = inputs
# x = tf.keras.applications.xception.preprocess_input(x) # for input scaling, not for all models
x = pretrained(x, training=False)
x = keras.layers.Flatten()(x)
# to do: try batch normalization
x = keras.layers.Dense(512, activation='relu')(x)
x = keras.layers.Dropout(0.2, seed=29)(x)
outputs = keras.layers.Dense(15, activation='softmax')(x)
model = keras.Model(inputs, outputs)

model.compile(
    optimizer=keras.optimizers.Adam(),
    loss=keras.losses.CategoricalCrossentropy(),
    metrics=[keras.metrics.CategoricalAccuracy()],
)

hist = model.fit(x_train, y_train, batch_size=32, epochs=20, validation_data=(x_test, y_test))


# Fine-tuning

pretrained.trainable = True

model.compile(
    optimizer=keras.optimizers.Adam(1e-5),
    loss=keras.losses.CategoricalCrossentropy(),
    metrics=[keras.metrics.CategoricalAccuracy()],
)

epochs = 5

model.fit(x_train, y_train, batch_size=32, epochs=epochs, validation_data=(x_test, y_test))


# To do: save model
# To do: plot accuracy and loss
