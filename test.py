from scripts.data_processing.preprocessing import common_preprocessing
from scripts.model import Model
# from keras.models import model_from_json
import numpy as np
import os
import pandas as pd

# def load_model(architecture_path, weights_path):
#     json_file = open(architecture_path, 'r')
#     model_json = json_file.read()
#     json_file.close()
#     model = model_from_json(model_json)
#     model.load_weights(weights_path)
#     # Dont think these params mater but we have to compile the model
#     optimizer = "sgd"
#     loss = "categorical_crossentropy"
#     metrics = ["accuracy"]
#     model.compile(optimizer = optimizer, loss = loss, metrics = metrics)
#     return model

# def get_submission(model, test, csv_path = "../input/solution.csv"):
#     results = model.predict(test)
#     results = np.argmax(results, axis = 1)
#     results = pd.Series(results, name="Label")
#     submission = pd.concat([pd.Series(range(1, 28001), name = "ImageId"), results], axis = 1)
#     submission.to_csv(csv_path, index = False)

# if __name__ == "__main__":
#     architecture_path = "../models/architecture.json"
#     weights_path = "../models/weights-14-0.97.hdf5"
#     input_path = "../input/test.csv"

#     model = load_model(architecture_path = architecture_path,
#                        weights_path = weights_path)

#     test = pd.read_csv(input_path)
#     test = common_preprocessing(df = test)
#     get_submission(model = model, test = test)

if __name__ == "__main__":
    # TODO(oleguer): This needs a major refactor

    param_yaml_path = "params/basic_test.yaml"
    test_path = "/home/oleguer/projects/kaggle_digit-recognizer/input/test.csv"
    weights_path = "/home/oleguer/projects/kaggle_digit-recognizer/models/2019-07-11_12:00:42/weights-02-1.00.hdf5"
    solution_path = "/home/oleguer/projects/kaggle_digit-recognizer/input/solution.csv"

    model = Model(param_yaml = param_yaml_path)

    test = pd.read_csv(test_path)
    test = common_preprocessing(df = test)


    loaded_model = model.recover_logged_model(weights_path)
    # print(type(loaded_model))
    model.get_submission(mod = loaded_model, test = test, csv_path = solution_path)