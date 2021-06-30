import csv
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def get_data(filename):
    with open(filename) as training_file:
        training_reader = csv.reader(training_file, delimiter=',')
        image = []
        labels = []
        line_count = 0
        for row in training_reader:
            if line_count == 0:
                line_count +=1
            else:
                labels.append(row[0])
                temp_image = row[1:785]
                image_data_as_array = np.array_split(temp_image, 28)
                image.append(image_data_as_array)
                line_count += 1
        images = np.array(image).astype('float')
        labels = np.array(labels).astype('float')
        print(f'Processed {line_count} lines.')

    return images, labels

# root = "/home/gibeom/dataset/asl_mnist"
# train_images, train_labels = get_data(root + "/sign_mnist_train.csv")
# test_images, test_labels = get_data(root + "/sign_mnist_test.csv")
#
# print("Total Training images", train_images.shape)
# print("Total Training labels", train_labels.shape)
# print("Total Testing images", test_images.shape)
# print("Total Testing labels", test_labels.shape)