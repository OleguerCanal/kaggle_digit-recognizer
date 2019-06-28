import pandas as pd
import numpy as np
from preprocessing import preprocess_data
from architectures import simple_cnn_classification
from keras.optimizers import RMSprop
from datagenerators import simple_image_augmentation
from results_analysis import plot_history, plot_confusion_matrix

if __name__ == "__main__":
    # 1. Load data
    raw_train = pd.read_csv("../input/train.csv")
    raw_test =  pd.read_csv("../input/test.csv")

    # raw_train = raw_train.sample(frac=0.001)  # Only to test pipeline
    print("Train shape:", raw_train.shape)
    b = input("enter")

    # 2. Process data
    x_train, y_train, x_val, y_val, x_test = preprocess_data(raw_train, raw_test)
    del raw_train, raw_test

    # 3. Define Model
    epochs = 10 # Turn epochs to 30 to get 0.9967 accuracy
    batch_size = 60
    optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
    loss = "categorical_crossentropy"
    metrics = ["accuracy"]

    model = simple_cnn_classification(input_shape = x_train[0].shape)
    model.compile(optimizer = optimizer, loss = loss, metrics = metrics)

    # 4. Get DataGenerator (image augmentation)
    datagen = simple_image_augmentation()
    datagen.fit(x_train)

    # 5. Fit Model
    history = model.fit_generator(
                        generator = datagen.flow(x_train, y_train, batch_size=batch_size),
                        epochs = epochs,
                        validation_data = (x_val, y_val),
                        verbose = 2,
                        steps_per_epoch = x_train.shape[0])
    
    a = input("enter")

    # 6 Analyze results
    plot_history(history)

    y_pred = model.predict(x_val)
    plot_confusion_matrix(y_pred = y_pred, y_val = y_val, classes = range(10))