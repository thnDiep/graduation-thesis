from keras import Sequential, Input
from keras.layers import Dense, Conv2D, Flatten, MaxPool2D


def cnnpred_2d(sequence_length, n_feature, n_filters):
    """
    Build model using architecture that is specified on the paper
    (Hoseinzade and Haratizadeh).
    """

    model = Sequential([
        # Layer 1
        Input(shape=(sequence_length, n_feature, 1)),
        Conv2D(n_filters[0], (1, n_feature), activation="relu"),

        # Layer 2
        Conv2D(n_filters[1], (3, 1), activation="relu"),
        MaxPool2D(pool_size=(2, 1)),

        # Layer 3
        Conv2D(n_filters[2], (3, 1), activation="relu"),
        MaxPool2D(pool_size=(2, 1)),

        # FFNN
        Flatten(),
        Dense(3, activation='softmax')   # 3 giá trị action
    ])

    return model