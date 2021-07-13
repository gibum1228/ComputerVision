import csv

import cv2
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import read_data
import create_model

if __name__ == "__main__":
    # read data
    root = "/home/gibeom/dataset/asl_image_recognition"
    train_images_alphabet, train_labels_alphabet = read_data.get_32by32_data_for_28by28_data(root + "/asl_alphabet/sign_mnist_train.csv")
    test_images_alphabet, test_labels_alphabet = read_data.get_32by32_data_for_28by28_data(root + "/asl_alphabet/sign_mnist_test.csv")
    train_images_digit, train_labels_digit = read_data.get_image_data_for_dataloader(root + "/asl_digit/train")
    test_images_digit, test_labels_digit = read_data.get_image_data_for_dataloader(root + "/asl_digit/test")


    # case 1
    # train_images = train_images_alphabet
    # train_labels = train_labels_alphabet
    # test_images = test_images_alphabet
    # test_labels = test_labels_alphabet
    # case 2
    # train_images = train_images_digit
    # train_labels = train_labels_digit
    # test_images = test_images_digit
    # test_labels = test_labels_digit
    # case 3
    train_images = np.concatenate([train_images_alphabet, train_images_digit])
    train_labels = np.concatenate([train_labels_alphabet, train_labels_digit])
    test_images = np.concatenate([test_images_alphabet, test_images_digit])
    test_labels = np.concatenate([test_labels_alphabet, test_labels_digit])

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

    # model init
    model = create_model.lenet5() # LeNet-5

    # Compiling the Model.
    # model = tf.keras.applications.DenseNet201(input_shape=(32, 32, 3), include_top=False, pooling='avg')
    # model.compile(loss='sparse_categorical_crossentropy',
    #               optimizer='Adam',
    #               metrics=['accuracy'])

    history = model.fit(train_datagenerator,
                        validation_data=val_datagenerator,
                        steps_per_epoch=len(train_labels) // 32,
                        epochs=500,
                        validation_steps=len(test_labels) // 32)

    # history = model.fit(train_images,
    #                     train_labels,
    #                     validation_data=(test_images, test_labels),
    #                     epochs=50)

    # 훈련 과정 시각화 (정확도)
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('acc')
    plt.legend(['Train', 'Test'], loc='upper left')
    # 훈련 과정 시각화 (손실)
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('loss')
    plt.legend(['Train', 'Test'], loc='upper left')
    # plt.show()

    val_loss, val_acc = model.evaluate(val_datagenerator, verbose=0)
    model.save('./data/lenet5_asl_recognition_500.h5')
    print()
    print("val_acc =>", val_acc)
    print("val_loss =>", val_loss)
    print()
    print("train >>", len(train_images))
    print("test >>", len(test_images))
    plt.show()