import csv
import cv2
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import collections
from PIL import Image
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms


def get_32by32_data_for_28by28_data(path):
    print(f"start>> get_32by32_data_for_28by28_data({path})")

    # read for csv
    with open(path) as training_file:
        training_reader = csv.reader(training_file, delimiter=',')
        image = []
        labels = []
        line_count = 0
        for row in training_reader:
            if line_count == 0:
                line_count += 1
            else:
                labels.append(row[0])
                temp_image = row[1:785]
                image_data_as_array = np.array_split(temp_image, 28)
                image.append(image_data_as_array)
                line_count += 1
        images = np.array(image).astype('float')
        labels = np.array(labels).astype('float')

    # 32 by 32 list
    result_images = np.empty((images.shape[0], 32, 32), dtype='float64')

    # (28, 28) to (32, 32)
    for i in range(len(images)):
        result_images[i] = cv2.resize(images[i], (32, 32), interpolation=cv2.INTER_AREA)

    result_images = np.expand_dims(result_images, axis=3) # (samples, width, height, channel)

    print("end>> get_32by32_data_for_28by28_data() success")

    return result_images, labels


def get_image_data_for_dataloader(path):
    print(f"start>> get_image_data_for_dataloader({path})")

    # 데이터 전처리
    trans = transforms.Compose([transforms.Resize((32, 32)),
                                transforms.ToTensor(),
                                #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), # 3 channel
                                transforms.Normalize(0.5, 0.5), # grayscale, 1 channel
                                transforms.Grayscale()])
    custom_dataset = torchvision.datasets.ImageFolder(root=path,
                                                      transform=trans)

    # 데이터 읽어 오기
    data_loader = DataLoader(custom_dataset,
                             batch_size=2000,
                             shuffle=False,
                             num_workers=10)

    data_iter = iter(data_loader)
    images, labels = np.empty((0, 32, 32, 1)), np.empty((0,)) # 데이터 집합

    # concat images and labels
    try:
        for _ in range(len(data_iter)):
            image, label = data_iter.next()

            images = np.append(images, image.numpy().reshape(-1, 32, 32, 1), axis=0)
            labels = np.append(labels, label.numpy(), axis=0)
    except Exception:
        print()

    print("end>> get_image_data_for_dataloader() success")

    return images, labels


def save_data_to_csv(images, labels, filename, trigger=True):
    print("start>> save .csv file")

    if trigger:
        folder = "train"
    else:
        folder = "test"
    path = f"./data/{folder}/{filename}.csv"

    images = images.reshape(-1, 1024)
    pd.DataFrame(images).to_csv(path)

    print("end>> save .csv file success")


def get_32by32_data_for_csv(path):
    print(f"start>> get_32by32_data({path})")

    # read for csv
    with open(path) as training_file:
        training_reader = csv.reader(training_file, delimiter=',')
        image = []
        label = []

        for row in training_reader:
            label.append(row[0])
            temp_image = row[1:1025]
            image_data_as_array = np.array_split(temp_image, 32)
            image.append(image_data_as_array)

        images = np.array(image).astype('float')
        labels = np.array(label).astype('float')

    print("end>> get_32by32_data() success")

    return images, labels


if __name__ == '__main__':
    labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
              'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
    # read data
    root = "/home/gibeom/dataset/asl_image_recognition"
    # case 1
    train_images_alphabet, train_labels_alphabet = get_32by32_data_for_28by28_data(root + "/asl_alphabet/sign_mnist_train.csv")
    test_images_alphabet, test_labels_alphabet = get_32by32_data_for_28by28_data(root + "/asl_alphabet/sign_mnist_test.csv")
    # case 2
    train_images_digit, train_labels_digit = get_image_data_for_dataloader(root + "/asl_digit/train")
    test_images_digit, test_labels_digit = get_image_data_for_dataloader(root + "/asl_digit/test")
    # # case 3
    # train_images = np.concatenate([train_images_alphabet, train_images_digit])
    # train_labels = np.concatenate([train_labels_alphabet, train_labels_digit])
    # test_images = np.concatenate([test_images_alphabet, test_images_digit])
    # test_labels = np.concatenate([test_labels_alphabet, test_labels_digit])

    # save .csv file
    # save_data_to_csv(train_images_alphabet, train_labels_alphabet, "test1")

    # Alphabet Dataset Show Table
    #
    # # DB 테이블
    # alphabet_labels = []
    # for i in range(26):
    #     alphabet_labels.append(chr(i + 65))
    #
    # plt.figure(figsize=(30, 10))
    # for i in range(4): # (4, 24) graph, j와 z는 단일 이미지 동작이 아니기 때문에 제외
    #     for j in range(25):
    #         if j == 9:
    #             continue
    #
    #         plt.subplot(4, 25, (i * 25) + j + 1)
    #         plt.xticks([])
    #         plt.yticks([])
    #         plt.grid(False)
    #
    #         train_find_index = np.where(train_labels_alphabet == j)[0]
    #         test_find_index = np.where(test_labels_alphabet == j)[0]
    #
    #         if i == 0:
    #             plt.imshow(train_images_alphabet[train_find_index[0]], cmap='gray')
    #             plt.xlabel(alphabet_labels[j])
    #
    #             if j == 0:
    #                 plt.ylabel("train")
    #         elif i == 1:
    #             plt.imshow(train_images_alphabet[train_find_index[len(train_find_index) - 1]], cmap='gray')
    #             plt.xlabel(alphabet_labels[j])
    #
    #             if j == 0:
    #                 plt.ylabel("train")
    #         elif i == 2:
    #             plt.imshow(test_images_alphabet[test_find_index[0]], cmap='gray')
    #             plt.xlabel(alphabet_labels[j])
    #
    #             if j == 0:
    #                 plt.ylabel("test")
    #         else:
    #             plt.imshow(test_images_alphabet[test_find_index[len(test_find_index) - 1]], cmap='gray')
    #             plt.xlabel(alphabet_labels[j])
    #
    #             if j == 0:
    #                 plt.ylabel("test")
    # plt.show()

    # Digit Dataset Show Table
    # plt.figure(figsize=(30, 10))
    #
    # for i in range(4):
    #     for j in range(10):
    #         plt.subplot(4, 10, (i * 10) + j + 1)
    #         plt.xticks([])
    #         plt.yticks([])
    #         plt.grid(False)
    #
    #         train_find_index = np.where(train_labels_digit == j)[0]
    #         test_find_index = np.where(test_labels_digit == j)[0]
    #
    #         if i == 0:
    #             plt.imshow(train_images_digit[train_find_index[0]], cmap='gray')
    #             plt.xlabel(j)
    #
    #             if j == 0:
    #                 plt.ylabel("train")
    #         elif i == 1:
    #             plt.imshow(train_images_digit[train_find_index[len(train_find_index) - 1]], cmap='gray')
    #             plt.xlabel(j)
    #
    #         elif i == 2:
    #             plt.imshow(test_images_digit[test_find_index[0]], cmap='gray')
    #             plt.xlabel(j)
    #
    #             if j == 0:
    #                 plt.ylabel("test")
    #         else:
    #             plt.imshow(test_images_digit[test_find_index[len(test_find_index) - 1]], cmap='gray')
    #             plt.xlabel(j)
    #
    # plt.show()

    # data info
    train, test = [], []
    for i in range(26):
        train.append(collections.Counter(train_labels_alphabet)[i])
        test.append(collections.Counter(test_labels_alphabet)[i])

    train.append(" ")
    test.append(" ")
    train.append(len(train_labels_alphabet))
    test.append(len(test_labels_alphabet))
    df = pd.DataFrame({
        "train data": train,
        "test data": test
    })

    print(df)