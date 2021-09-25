import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

import read_data

if __name__ == "__main__":
    model_case1 = tf.keras.models.load_model('data/lenet5_alphabet_1000.h5')
    model_case2 = tf.keras.models.load_model('data/lenet5_digit_1000.h5')
    model_case3 = tf.keras.models.load_model('data/lenet5_recognition_1000_v2_acc99.h5')
    model = tf.keras.models.load_model('data/lenet5_recognition_1000_v2_acc99.h5')
    labels = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
              'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
              'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

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

    # train accuracy 구하기
    print(model_case1.evaluate(train_images_alphabet, train_labels_alphabet))
    print(model_case2.evaluate(train_images_digit, train_labels_digit))
    print(model_case3.evaluate(train_images, train_labels))
    exit()

    # predict
    pred = model.predict(test_images)
    pred = np.argmax(pred, axis=1)

    # classification_report
    test_labels = test_labels.astype('int64')
    # y_test_word = [labels[i] for i in test_labels]
    # pred_word = [labels[i] for i in pred]
    y_test_word = [labels[i] for i in test_labels]
    pred_word = [labels[i] for i in pred]
    # print table
    # print(classification_report(y_test_word, pred_word))
    #
    # val_loss, val_acc = model.evaluate(test_images, test_labels)
    # print(val_acc)
    # exit()

    # confusion matrix
    cf_matrix = confusion_matrix(y_test_word, pred_word, normalize='true')
    plt.figure(figsize=(30, 15))
    sns.heatmap(cf_matrix, annot=True, xticklabels=sorted(set(y_test_word)), yticklabels=sorted(set(y_test_word)), cbar=False)
    plt.title("CASE 3 Confusion Matrix\n", fontsize=25)
    plt.xlabel("Predict", fontsize=20)
    plt.ylabel("True", fontsize=20)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15, rotation=0)
    plt.show()