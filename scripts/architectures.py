from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense

def SimpleCnnClassification():
    model = Sequential()

    # Feature Extraction Block 1
    model.add(Conv2D(filters = 32,
                    kernel_size = (5, 5),
                    strides = (1, 1),
                    padding = "same",
                    activation = "relu"))
    model.add(Conv2D(filters = 32,
                    kernel_size = (5, 5),
                    strides = (1, 1),
                    padding = "same",
                    activation = "relu"))
    model.add(MaxPooling2D(pool_size = (2, 2),
                    strides=2,
                    padding="valid"))
    model.add(Dropout(rate=0.2))

    # Feature Extraction Block 2
    model.add(Conv2D(filters = 64,
                    kernel_size = (3, 3),
                    strides = (1, 1),
                    padding = "same",
                    activation = "relu"))
    model.add(Conv2D(filters = 64,
                    kernel_size = (3, 3),
                    strides = (1, 1),
                    padding = "same",
                    activation = "relu"))
    model.add(MaxPooling2D(pool_size = (2, 2),
                    strides = 2,
                    padding = "valid"))
    model.add(Dropout(rate = 0.2))

    # Flatten & connect to classes
    model.add(Flatten())
    model.add(Dense(units = 256, activation = "tanh")
    model.add(Dropout(rate = 0.2))
    model.add(Dense(units = 10, activation = "softmax"))
    return model