import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras import layers
import tensorflow_addons as tfa

from PIL import Image
from sklearn.model_selection import train_test_split
from typing import Tuple

def split_data(train_df: pd.DataFrame, 
               train_path: str, 
               random_state: int = 42) -> Tuple[np.ndarray,
                                                np.ndarray,
                                                np.ndarray,
                                                np.ndarray]:

    '''
        Wrapper function for sklearn.model_selection.train_test_split() 

        Arguments:
            train_df: dataframe containing locations and labels for training images
            train_path: the location of the training images 
            random_state: random seed to allow for replicable splits 

        Returns:
            a tuple containing the following objects:
                - x_train: a numpy array containing the training images in matrix form 
                - x_test:  a numpy array containing the test images in matrix form 
                - y_train: a numpy array containing the training set labels 
                - y_test:  a numpy array containing the test set labels

    '''
    img = []

    for i in range(len(train_df)):
        im = train_path + train_df['filename'].iloc[i]
        temp = Image.open(im)
        img.append(np.array(temp.resize((200,200))))
        
    labels = tf.keras.utils.to_categorical(np.asarray(train_df['label'].factorize()[0]))

    images = np.asarray(img)

    x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size=0.10, random_state=random_state, stratify=labels)

    return x_train, x_test, y_train, y_test

# Some helper functions for ViT

def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = layers.Dense(units, activation=tf.nn.gelu)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x

class Patches(layers.Layer):
    def __init__(self, patch_size):
        super().__init__()
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches

class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super().__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded

def create_vit_classifier():
    num_classes = 15
    image_size = 200  # We'll resize input images to this size
    input_shape = (200, 200, 3)
    patch_size = 10  
    num_patches = (image_size // patch_size) ** 2
    projection_dim = 64
    num_heads = 4
    transformer_units = [
        projection_dim * 2,
        projection_dim,
    ]  # Size of the transformer layers
    transformer_layers = 8
    mlp_head_units = [512, 15]  # Size of the dense layers of the final classifier

    inputs = layers.Input(shape=input_shape)
    # Augment data.
    # augmented = data_augmentation(inputs)
    augmented = inputs # No data augmentation for now
    # Create patches.
    patches = Patches(patch_size)(augmented)
    # Encode patches.
    encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)

    # Create multiple layers of the Transformer block.
    for _ in range(transformer_layers):
        # Layer normalization 1.
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        # Create a multi-head attention layer.
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=0.1
        )(x1, x1)
        # Skip connection 1.
        x2 = layers.Add()([attention_output, encoded_patches])
        # Layer normalization 2.
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        # MLP.
        x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=0.1)
        # Skip connection 2.
        encoded_patches = layers.Add()([x3, x2])

    # Create a [batch_size, projection_dim] tensor.
    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    representation = layers.Flatten()(representation)
    representation = layers.Dropout(0.5)(representation)
    # Add MLP.
    features = mlp(representation, hidden_units=mlp_head_units, dropout_rate=0.5)
    # Classify outputs.
    logits = layers.Dense(num_classes, activation='softmax')(features)
    # Create the Keras model.
    model = keras.Model(inputs=inputs, outputs=logits)
    return model


# Build the model for predicting the action
def train_model(model_config: str, 
                x_train: np.ndarray, 
                y_train: np.ndarray, 
                x_test: np.ndarray, 
                y_test: np.ndarray, 
                augment: bool = False,
                batch_size: int = 32, 
                epochs: int = 5, 
                random_seed: int = 42) -> Tuple[tf.keras.Model, 
                                                tf.keras.callbacks.History, 
                                                tf.keras.callbacks.History]:
    
    '''
        Convenience function for compiling and training the model 

        Arguments:
            model_config: a string indicating the network architecture to be used
            x_train:      an array containing matrix representations of the training images 
            y_train:      an array containing the training labels 
            x_test:       an array containing matrix representations of the test images 
            y_test:       an array containing the test labels 
            batch_size:   an int indicating the batch size for model training 
            epochs:       the number of epochs to train the model 
            random_seed:  random seed for replicable training results 

        The function will throw an exception if an invalid model configuration is passed to it.
        Valid configuratinos include:
            - VGG16
            - ResNet50
            - Xception
            - EffecientNet84 
            - EfficientNetV2M
            - ConvNeXtLarge
            - CNN
            - ViT

        Returns:
            a tuple containing the following objects:
                - model: the trained model
                - hist1: the history for the first round of training 
                - hist2: the history for the second round of training (if applicable)
                         if only one round of training ran, hist2 = hist1

        TODO: Add data augmentation layers 

    '''

    tf.random.set_seed(random_seed)

    if model_config == 'VGG16':
        pretrained = tf.keras.applications.VGG16(input_shape=(200,200,3), 
                                                include_top=False, 
                                                weights='imagenet',
                                                pooling='avg')
    elif model_config == 'ResNet50':
        pretrained = tf.keras.applications.ResNet50(input_shape=(200,200,3), 
                                                include_top=False, 
                                                weights='imagenet',
                                                pooling='avg')
    elif model_config == 'Xception':
        pretrained = tf.keras.applications.Xception(input_shape=(200,200,3), 
                                                include_top=False, 
                                                weights='imagenet',
                                                pooling='avg')
    elif model_config == 'EfficientNetB4':
        pretrained = tf.keras.applications.EfficientNetB4(input_shape=(200,200,3), 
                                                include_top=False, 
                                                weights='imagenet',
                                                pooling='avg')
    elif model_config == 'EfficientNetV2M':
        pretrained = tf.keras.applications.EfficientNetV2M(input_shape=(200,200,3), 
                                                include_top=False, 
                                                weights='imagenet',
                                                pooling='avg')
    elif model_config == 'ConvNeXtLarge':
        pretrained = tf.keras.applications.ConvNeXtLarge(input_shape=(200,200,3), 
                                                include_top=False, 
                                                weights='imagenet',
                                                pooling='avg')
    elif model_config == 'CNN':
        model = tf.keras.Sequential([
                    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(200, 200, 3)),
                    tf.keras.layers.MaxPooling2D((2, 2)),
                    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
                    tf.keras.layers.MaxPooling2D((2, 2)),
                    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
                    tf.keras.layers.MaxPooling2D((2, 2)),
                    tf.keras.layers.Flatten(),
                    tf.keras.layers.Dense(128, activation='relu'),
                    tf.keras.layers. Dropout(0.2),
                    tf.keras.layers. Dense(15, activation='softmax')
        ])
        
    elif model_config == 'ViT':
        model = create_vit_classifier()

        model.compile(
            optimizer=tf.keras.optimizers.Adam(),
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=[tf.keras.metrics.CategoricalAccuracy()],
        )

        model.summary()

        hist1 = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test, y_test))

        hist2 = hist1

        return model, hist1, hist2
    else:
        raise Exception('Invalid model configuration')
    
    # pretrained layers


    # data augmentation (to be used if applicable)
    data_augmentation = tf.keras.Sequential(
        [tf.keras.layers.RandomFlip("horizontal"), tf.keras.layers.RandomRotation(0.2),
        tf.keras.layers.RandomZoom(0.2), tf.keras.layers.RandomContrast(0.1), 
        tf.keras.layers.RandomTranslation(0.1, 0.1),]
    )

    # note: this will only run if you are fine-tuning one of the pretrained networks
    pretrained.trainable = False
    inputs = tf.keras.Input(shape=(200,200,3))
    if augment:
        x = data_augmentation(inputs)
    else:
        x = inputs
    x = pretrained(x, training=False)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(512, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.2, seed=29)(x)
    outputs = tf.keras.layers.Dense(15, activation='softmax')(x)
    model = tf.keras.Model(inputs, outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=[tf.keras.metrics.CategoricalAccuracy()],
    )

    model.summary()

    hist1 = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test, y_test))
    
    pretrained.trainable = True
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-5),
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=[tf.keras.metrics.CategoricalAccuracy()],
    )

    epochs = 5

    hist2 = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test, y_test))

    # return the model histories and the model itself
    return model, hist1, hist2

def plot_history(history: tf.keras.callbacks.History, 
                 config: str) -> None:

    '''
        Saves plots of model history in a visualizations folder 

        Arguments:
            - history: the model history 
            - config: a string representing the model configuration 

    '''
    fig = plt.figure(figsize=(15,4))

    fig.add_subplot(121)
    plt.plot(history.history['categorical_accuracy'])
    plt.plot(history.history['val_categorical_accuracy'])
    plt.legend(['Training accuracy', 'Validation accuracy'])
    plt.title('Training and validation accuracy')
    plt.xlabel('Epoch Number')
    plt.ylabel('Accuracy')

    fig.add_subplot(122)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.legend(['Training loss', 'Validation loss'])
    plt.title('Training and validation loss')
    plt.xlabel('Epoch Number')
    plt.ylabel('Loss')
    
    plt.savefig('./visualizations/'+config+'.png')
    plt.show()