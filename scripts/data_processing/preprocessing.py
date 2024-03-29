import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical 
import seaborn as sns

def common_preprocessing(df):
    df = df.values.reshape(-1, 28, 28, 1)
    return df

def preprocess_data(train, train_val_prop = 0.25, rnd_seed = 1):
    # Shuffle
    train = shuffle(train, random_state = rnd_seed)

    # Split x, y
    y_train = train["label"]
    x_train = train.drop(labels = ["label"], axis = 1)
    del train

    # Reshape images (from vector to matrix)
    x_train = common_preprocessing(x_train)

    # sns.countplot(y_train)

    # Value to categorical variable
    y_train = to_categorical(y_train, num_classes=10)

    # Split train/val
    x_train, x_val, y_train, y_val = train_test_split(
        x_train, y_train,
        test_size = train_val_prop,
        random_state = rnd_seed)

    return x_train, y_train, x_val, y_val