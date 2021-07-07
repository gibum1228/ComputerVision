import math

import cv2 as cv
import numpy as np
from tensorflow.keras import models

if __name__ == '__main__':
    model = models.load_model('alphabet_lenet5.h5')

    img = cv.imread("/home/gibeom/dataset/asl_alphabet/asl_alphabet_train/asl_alphabet_train/R/R85.jpg", cv.IMREAD_GRAYSCALE)

    predict_img = cv.resize(img, (32, 32), cv.INTER_AREA)
    predict_img = predict_img / 255.0
    predict_img = np.expand_dims(predict_img, axis=0)
    predict_img = np.expand_dims(predict_img, axis=3)

    print(img.shape)
    print(predict_img.shape)

    predict_acc = model.predict(predict_img)[0]
    print(predict_acc)
    print(max(predict_acc))
    for i in range(len(predict_acc)):
        if math.floor(predict_acc[i] * 100) > 0:
            print(math.floor(predict_acc[i] * 100), i)