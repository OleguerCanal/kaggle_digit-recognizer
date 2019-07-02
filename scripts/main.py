import pandas as pd
import numpy as np
import os
from time import time
from preprocessing import preprocess_data
from architectures import simple_cnn_classification
from keras.optimizers import RMSprop
from keras.models import model_from_json
from datagenerators import simple_image_augmentation
from results_analysis import plot_history, plot_confusion_matrix
from callbacks import TelegramSummary, TensorBoard, ReduceLROnPlateau, ModelCheckpoint

def load_saved_model():
    # load json and create model
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("model.h5")
    print("Loaded model from disk")

def get_submission(model, test, csv_path = "../input/solution.csv"):
    results = model.predict(test)
    results = np.argmax(results, axis = 1)
    results = pd.Series(results, name="Label")
    submission = pd.concat([pd.Series(range(1, 28001), name = "ImageId"), results], axis = 1)
    submission.to_csv(csv_path, index = False)

if __name__ == "__main__":
    # 1. Load data
    raw_train = pd.read_csv("../input/train.csv")
    raw_test =  pd.read_csv("../input/test.csv")

    # raw_train = raw_train.sample(frac=0.01)  # Only to test pipeline

    # 2. Process data
    x_train, y_train, x_val, y_val, x_test = preprocess_data(raw_train, raw_test)
    del raw_train, raw_test

    # 3. Define Model
    epochs = 30 # Turn epochs to 30 to get 0.9967 accuracy
    batch_size = 80
    optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
    loss = "categorical_crossentropy"
    metrics = ["accuracy"]

    model = simple_cnn_classification(input_shape = x_train[0].shape)
    model.compile(optimizer = optimizer, loss = loss, metrics = metrics)

    # serialize model to JSON
    model_saving_path = "../models/architecture.json"
    model_json = model.to_json()
    with open(model_saving_path, "w") as json_file:
        json_file.write(model_json)

    # Datagen
    datagen = simple_image_augmentation()
    datagen.fit(x_train)

    # Callbacks:
    weights_filepath = "../models/weights-{epoch:02d}-{val_acc:.2f}.hdf5"
    checkpoint = ModelCheckpoint(  # Save model weights after each epoch
                                filepath=weights_filepath,
                                monitor='val_acc',
                                verbose=1,
                                save_best_only=True,
                                mode='max')
    telegram_summary = TelegramSummary()
    tensorboard = TensorBoard(log_dir="../logs/{}".format(time()))
    learning_rate_reduction = ReduceLROnPlateau(
                                            monitor = 'val_acc', 
                                            patience = 3,
                                            verbose = 1,
                                            factor = 0.5,  # Each epoch reduce lr by half
                                            min_lr = 0.00001)
    callbacks = [telegram_summary, tensorboard, learning_rate_reduction, checkpoint]

    # 4. Fit Model
    model.summary()
    history = model.fit_generator(
                        generator = datagen.flow(x_train, y_train, batch_size=batch_size),
                        epochs = epochs,
                        validation_data = (x_val, y_val),
                        verbose = 1,
                        callbacks = callbacks,
                        steps_per_epoch = x_train.shape[0])
    
    # 5. Analyze results
    # plot_history(history)

    y_pred = model.predict(x_val)
    plot_confusion_matrix(y_pred = y_pred, y_val = y_val, classes = range(10))