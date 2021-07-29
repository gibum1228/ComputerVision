import tensorflow as tf
from tensorflow.keras import models, layers


def custom_model():
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


def lenet5():
    model = models.Sequential([
        layers.Conv2D(6, kernel_size=5, strides=1, activation='relu', input_shape=(32, 32, 1)),
        layers.MaxPool2D(pool_size=2, strides=2),
        layers.Conv2D(16, kernel_size=5, strides=1, activation='relu'),
        layers.MaxPool2D(pool_size=2, strides=2),
        layers.Flatten(),
        layers.Dense(120, activation='relu'),
        layers.Dense(84, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])

    # Compiling the Model.
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=tf.keras.optimizers.Adam(),
                  metrics=['accuracy'])

    # create lenet5 network structure .png
    # tf.keras.utils.plot_model(model,
    #                           to_file='lenet5.png',
    #                           show_shapes=True,
    #                           show_dtype=False,
    #                           show_layer_names=True,
    #                           rankdir='TB',
    #                           expand_nested=True,
    #                           dpi=96)

    return model
