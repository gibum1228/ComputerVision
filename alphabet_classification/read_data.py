import csv
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from PIL import Image
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms

def get_data_for_csv(path):
    with open(path) as training_file:
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

def get_data_for_dataloader(path):
    # 데이터 전처리
    trans = transforms.Compose([transforms.Resize((32, 32)),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    custom_dataset = torchvision.datasets.ImageFolder(root=path,
                                                      transform=trans)

    data_loader = DataLoader(custom_dataset,
                             batch_size=2000,
                             shuffle=False,
                             num_workers=10)

    data_iter = iter(data_loader)
    images, labels = np.empty((0, 32, 32, 3)), np.empty((0,))
    try:
        for _ in range(len(data_iter)):
            i, l = data_iter.next()
            images = np.append(images, i.numpy().reshape(-1, 32, 32, 3), axis=0)
            labels = np.append(labels, l.numpy(), axis=0)
    except Exception:
        print()

    return images, labels
