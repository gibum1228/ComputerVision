import csv

import cv2
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

import read_data
import create_model

if __name__ == "__main__":
    # read data
    root = "/home/gibeom/dataset/asl_image_recognition"
    train_images_alphabet, train_labels_alphabet = read_data.get_32by32_data_for_28by28_data(root + "/asl_alphabet/sign_mnist_train.csv")
    test_images_alphabet, test_labels_alphabet = read_data.get_32by32_data_for_28by28_data(root + "/asl_alphabet/sign_mnist_test.csv")
    images_1, labels_1 = read_data.get_image_data_for_dataloader(root + "/asl_digit/1")
    images_2, labels_2 = read_data.get_image_data_for_dataloader(root + "/asl_digit/2")

    # case 1
    # train_images = train_images_alphabet
    # train_labels = train_labels_alphabet
    # test_images = test_images_alphabet
    # test_labels = test_labels_alphabet
    # case 2 original
    train_images_1, test_images_1, train_labels_1, test_labels_1 = train_test_split(images_1,
                                                                            labels_1,
                                                                            test_size=0.1,
                                                                            shuffle=True,
                                                                            stratify=labels_1)
    train_images_2, test_images_2, train_labels_2, test_labels_2 = train_test_split(images_2,
                                                                                    labels_2,
                                                                                    test_size=0.1,
                                                                                    shuffle=True,
                                                                                    stratify=labels_2)
    # train_images = np.concatenate([train_images_1, train_images_2])
    # train_labels = np.concatenate([train_labels_1, train_labels_2])
    # test_images = np.concatenate([test_images_1, test_images_2])
    # test_labels = np.concatenate([test_labels_1, test_labels_2])

    # print(train_images.shape) # (28855, 32, 32, 1)
    # print(train_labels.shape) # (28855, )
    # print(test_images.shape) # (3207, 32, 32, 1)
    # print(test_labels.shape) # (3207, )
    # exit()
    # case 3
    train_images_digit = np.concatenate([train_images_1, train_images_2])
    train_labels_digit = np.concatenate([train_labels_1, train_labels_2])
    test_images_digit = np.concatenate([test_images_1, test_images_2])
    test_labels_digit = np.concatenate([test_labels_1, test_labels_2])
    train_labels_alphabet = train_labels_alphabet + 10
    test_labels_alphabet = test_labels_alphabet + 10

    train_images = np.concatenate([train_images_alphabet, train_images_digit])
    train_labels = np.concatenate([train_labels_alphabet, train_labels_digit])
    test_images = np.concatenate([test_images_alphabet, test_images_digit])
    test_labels = np.concatenate([test_labels_alphabet, test_labels_digit])

    # Create an ImageDataGenerator and do Image Augmentation
    train_data = ImageDataGenerator(rescale=1.0 / 1.0,
                                       height_shift_range=0.1,
                                       width_shift_range=0.1,
                                       zoom_range=0.1,
                                       shear_range=0.1,
                                       rotation_range=10,
                                       fill_mode='nearest',
                                       horizontal_flip=True)
    # Image Augmentation is not done on the testing data
    val_data = ImageDataGenerator(rescale=1.0 / 1.0)
    train_datagenerator = train_data.flow(train_images,
                                          train_labels,
                                          batch_size=64)
    val_datagenerator = val_data.flow(test_images,
                             test_labels,
                             batch_size=64)

    # model init
    model = create_model.lenet5() # LeNet-5

    # Compiling the Model.
    # model = tf.keras.applications.DenseNet201(input_shape=(32, 32, 3), include_top=False, pooling='avg')
    # model.compile(loss='sparse_categorical_crossentropy',
    #               optimizer='Adam',
    #               metrics=['accuracy'])

    history = model.fit(train_datagenerator,
                        validation_data=val_datagenerator,
                        steps_per_epoch=len(train_labels) // 64,
                        epochs=1000,
                        validation_steps=len(test_labels) // 64)

    # history = model.fit(train_images,
    #                     train_labels,
    #                     validation_data=(test_images, test_labels),
    #                     epochs=50)

    # 훈련 과정 시각화 (정확도)
    # plt.subplot(1, 2, 1)
    # plt.plot(history.history['accuracy'])
    # plt.plot(history.history['val_accuracy'])
    # plt.title('Model Accuracy')
    # plt.xlabel('Epoch')
    # plt.ylabel('acc')
    # plt.legend(['Train', 'Test'], loc='upper left')
    # 훈련 과정 시각화 (손실)
    # plt.subplot(1, 2, 2)
    # plt.plot(history.history['loss'])
    # plt.plot(history.history['val_loss'])
    # plt.title('Model Loss')
    # plt.xlabel('Epoch')
    # plt.ylabel('loss')
    # plt.legend(['Train', 'Test'], loc='upper left')
    # plt.show()

    # evaluate case 1
    val_loss, val_acc = model.evaluate(test_images, test_labels, verbose=0)
    model.save('./data/lenet5_recognition_1000_v2.h5')
    print()
    print("val_acc =>", val_acc)
    print("val_loss =>", val_loss)

    # evaluate case 2
    pred = model.predict(test_images)
    pred = np.argmax(pred, axis=1)

    test_labels = test_labels.astype('int64')
    # classification_report
    y_test_word = [i for i in test_labels]
    pred_word = [i for i in pred]

    print(classification_report(y_test_word, pred_word))

    # confusion matrix
    cf_matrix = confusion_matrix(y_test_word, pred_word, normalize='true')
    plt.figure(figsize=(30, 15))
    sns.heatmap(cf_matrix, annot=True, xticklabels=sorted(set(y_test_word)), yticklabels=sorted(set(y_test_word)),
                cbar=False)
    plt.title("ASL Recognition Confusion Matrix\n", fontsize=25)
    plt.xlabel("Predict", fontsize=20)
    plt.ylabel("True", fontsize=20)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15, rotation=0)
    plt.show()