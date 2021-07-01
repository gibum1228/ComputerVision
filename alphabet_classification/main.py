import csv

import cv2
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import models, layers

import read_data

def create_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPool2D(2, 2),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPool2D(2, 2),
        tf.keras.layers.Conv2D(512, (3, 3), activation='relu'),
        tf.keras.layers.MaxPool2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1024, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(25, activation='softmax')
    ])

    # Compiling the Model.
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=tf.keras.optimizers.Adam(),
                  metrics=['accuracy'])

    return model

def create_lenet5():
    model = models.Sequential([
        layers.Conv2D(6, kernel_size=5, strides=1, activation='relu', input_shape=(32, 32, 1)),
        layers.MaxPool2D(pool_size=2, strides=2),
        layers.Conv2D(16, kernel_size=5, strides=1, activation='relu'),
        layers.MaxPool2D(pool_size=2, strides=2),
        layers.Flatten(),
        layers.Dense(120, activation='relu'),
        layers.Dense(84, activation='relu'),
        layers.Dense(25, activation='softmax')
    ])

    # Compiling the Model.
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='Adam',
                  metrics=['accuracy'])

    # tf.keras.utils.plot_model(model,
    #                           to_file='lenet5.png',
    #                           show_shapes=True,
    #                           show_dtype=False,
    #                           show_layer_names=True,
    #                           rankdir='TB',
    #                           expand_nested=True,
    #                           dpi=96)

    return model

if __name__ == "__main__":
    # read data
    root = "/home/gibeom/dataset/asl_mnist"
    train_images, train_labels = read_data.get_data(root + "/sign_mnist_train.csv")
    test_images, test_labels = read_data.get_data(root + "/sign_mnist_test.csv")

    copy_train_images = np.empty((train_images.shape[0], 32, 32), dtype='float64')
    copy_test_images = np.empty((test_images.shape[0], 32, 32), dtype='float64')

    # lenet5 input (32, 32, 1), so images resize
    for i in range(len(train_images)):
        copy_train_images[i] = cv2.resize(train_images[i], (32, 32), interpolation=cv2.INTER_AREA)
    for i in range(len(test_images)):
        copy_test_images[i] = cv2.resize(test_images[i], (32, 32), interpolation=cv2.INTER_AREA)

    # train_images = np.expand_dims(train_images, axis=3)
    # test_images = np.expand_dims(test_images, axis=3)

    copy_train_images = np.expand_dims(copy_train_images, axis=3)
    copy_test_images = np.expand_dims(copy_test_images, axis=3)
    train_images = copy_train_images
    test_images = copy_test_images

    print("Total Training images", train_images.shape)
    print("Total Training labels", train_labels.shape)
    print("Total Testing images", test_images.shape)
    print("Total Testing labels", test_labels.shape)
    print("copy train", copy_train_images.shape)
    print("copy test", copy_test_images.shape)

    # Create an ImageDataGenerator and do Image Augmentation
    train_data = ImageDataGenerator(rescale=1.0 / 255.0,
                                       height_shift_range=0.1,
                                       width_shift_range=0.1,
                                       zoom_range=0.1,
                                       shear_range=0.1,
                                       rotation_range=10,
                                       fill_mode='nearest',
                                       horizontal_flip=True)
    # Image Augmentation is not done on the testing data
    val_data = ImageDataGenerator(rescale=1.0 / 255)
    train_datagenerator = train_data.flow(train_images,
                                          train_labels,
                                          batch_size=32)
    val_datagenerator = val_data.flow(test_images,
                             test_labels,
                             batch_size=32)

    model = create_lenet5()

    history = model.fit(train_datagenerator,
                        validation_data=val_datagenerator,
                        steps_per_epoch=len(train_labels) // 32,
                        epochs=100,
                        validation_steps=len(test_labels) // 32)

    # Plot the chart for accuracy and loss on both training and validation
    # acc = history.history['accuracy']
    # val_acc = history.history['val_accuracy']
    # loss = history.history['loss']
    # val_loss = history.history['val_loss']
    #
    # epochs = range(len(acc))
    #
    # plt.plot(epochs, acc, 'r', label='Training accuracy')
    # plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
    # plt.title('Training and validation accuracy')
    # plt.legend()
    # plt.figure()
    #
    # plt.plot(epochs, loss, 'r', label='Training Loss')
    # plt.plot(epochs, val_loss, 'b', label='Validation Loss')
    # plt.title('Training and validation loss')
    # plt.legend()
    #
    # plt.show()

    val_loss, val_acc = model.evaluate(test_images, test_labels, verbose=0)
    model.save('alphabet_lenet5.h5')
    print()
    print("val_acc =>", val_acc)