import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report

import read_data

if __name__ == "__main__":
    model = tf.keras.models.load_model('data/lenet5_digit_1000.h5')
    labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
              'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

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

    pred = model.predict(test_images)
    print(pred)
    pred = np.argmax(pred, axis=1)
    print(pred)

    test_labels = test_labels.astype('int64')
    # classification_report
    y_test_word = [i for i in test_labels]
    pred_word = [i for i in pred]

    print(classification_report(y_test_word, pred_word))