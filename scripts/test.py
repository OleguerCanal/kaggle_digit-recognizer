from preprocessing import common_preprocessing
from keras.models import model_from_json
import numpy as np
import os
import pandas as pd

def load_model(architecture_path, weights_path):
    json_file = open(architecture_path, 'r')
    model_json = json_file.read()
    json_file.close()
    model = model_from_json(model_json)
    model.load_weights(weights_path)
    # Dont think these params mater but we have to compile the model
    optimizer = "sgd"
    loss = "categorical_crossentropy"
    metrics = ["accuracy"]
    model.compile(optimizer = optimizer, loss = loss, metrics = metrics)
    return model

def get_submission(model, test, csv_path = "../input/solution.csv"):
    results = model.predict(test)
    results = np.argmax(results, axis = 1)
    results = pd.Series(results, name="Label")
    submission = pd.concat([pd.Series(range(1, 28001), name = "ImageId"), results], axis = 1)
    submission.to_csv(csv_path, index = False)

if __name__ == "__main__":
    architecture_path = "../models/architecture.json"
    weights_path = "../models/weights-25-0.93.hdf5"
    input_path = "../input/test.csv"

    model = load_model(architecture_path = architecture_path,
                       weights_path = weights_path)

    test = pd.read_csv(input_path)
    test = common_preprocessing(df = test)
    get_submission(model = model, test = test)
