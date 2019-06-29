import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical 

def preprocess_data(train, test, train_val_prop = 0.25, rnd_seed = 0):
    # Shuffle
    train = shuffle(train, random_state = rnd_seed)
    test = shuffle(test, random_state = rnd_seed)

    # Split x, y
    y_train = train["label"]
    x_train = train.drop(labels = ["label"], axis = 1)
    del train

    # Reshape images (from vector to matrix)
    x_train = x_train.values.reshape(-1, 28, 28, 1)
    test = test.values.reshape(-1, 28, 28, 1)

    # Value to categorical variable
    y_train = to_categorical(y_train, num_classes=10)

    # Split train/val
    x_train, x_val, y_train, y_val = train_test_split(
        x_train, y_train,
        test_size = train_val_prop,
        random_state = rnd_seed)

    return x_train, y_train, x_val, y_val, test