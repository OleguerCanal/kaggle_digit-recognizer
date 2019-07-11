import datetime
import os
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import time
import yaml

# Keras
from keras.models import model_from_json
from keras.optimizers import RMSprop, Adam
from keras.preprocessing.image import ImageDataGenerator

# Own imports TODO(oleguer): Fix this path problem
sys.path.append(str(Path(__file__).parent))
from architectures.simple_cnn import simple_cnn_classification
from architectures.model2 import model2
from data_processing.preprocessing import preprocess_data
from helpers.callbacks import TensorBoard, ReduceLROnPlateau, ModelCheckpoint, TelegramSummary


import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

class Model():
    def __init__(self, param_yaml):
        self.__load_params(param_yaml)
    
    def __load_params(self, param_yaml):
        stream = open(param_yaml, 'r')
        self.params = yaml.load(stream, Loader = yaml.FullLoader)

    def recover_logged_model(self, weights_path):
        weights_name = weights_path.split("/")[-1]
        full_model_path = weights_path.replace("/" + weights_name, "")
        json_file = open(full_model_path + "/architecture.json", "r")
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)

        # load weights into new model
        loaded_model.load_weights(weights_path)
        print("Loaded model from disk")
        return loaded_model

    def __log_model(self, path):
        # Make sure dir exists
        if not os.path.exists(path):
            os.makedirs(path)

        # Serialize model to JSON
        model_json = self.model.to_json()
        with open(path + "/architecture.json", "w") as json_file:
            json_file.write(model_json)

        # Save model params
        with open(path + "/params.yaml", 'w') as outfile:
            yaml.dump(self.params, outfile, default_flow_style=False)

    def get_submission(self, mod, test, csv_path = "../input/solution.csv"):
        results = mod.predict(test)
        results = np.argmax(results, axis = 1)
        results = pd.Series(results, name="Label")
        submission = pd.concat([pd.Series(range(1, 28001), name = "ImageId"), results], axis = 1)
        submission.to_csv(csv_path, index = False)

    def train(self):
        # 1. Load data
        raw_train = pd.read_csv(self.params["data_path"])
        raw_train = raw_train.sample(frac = self.params["sample_data"])

        # 2. Process data
        x_train, y_train, x_val, y_val = preprocess_data(raw_train)
        del raw_train

        # 3. Define Model
        optimizer = RMSprop(
                        lr = float(self.params["learning_rate"]),
                        rho = float(self.params["rho"]),
                        epsilon = float(self.params["epsilon"]),
                        decay = float(self.params["decay"]))
        if str(self.params["optimizer"]) == "Adam":    
            opimizer = Adam(float(self.params["learning_rate"]))

        # self.model = simple_cnn_classification(input_shape = x_train[0].shape)  # Default: Start with random weights
        self.model = model2(input_shape = x_train[0].shape)  # Default: Start with random weights
        if self.params["train_from_saved_weights"]:
            self.model = self.recover_logged_model(self.params["saved_weights_path"])
        
        self.model.compile(
                        optimizer = optimizer,
                        loss = self.params["loss"],
                        metrics = self.params["metrics"])

        # 4. Log model
        time_stamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H:%M:%S')
        save_path = str(self.params["model_logging_path"]) + "/" + str(time_stamp)
        self.__log_model(path = save_path)

        # Datagen
        datagen_args = dict(rotation_range = 20,
                        width_shift_range = 0.1,
                        height_shift_range = 0.1,
                        shear_range = 0.1,
                        zoom_range = 0.1)
        datagen = ImageDataGenerator(**datagen_args)
        datagen.fit(x_train)

        # Callbacks:
        weights_filepath = save_path + "/weights-{epoch:0f}-{val_acc:.4f}.hdf5"
        checkpoint = ModelCheckpoint(  # Save model weights after each epoch
                                    filepath=weights_filepath,
                                    monitor='val_acc',
                                    verbose=1,
                                    save_best_only=True,
                                    mode='max')
        telegram_summary = TelegramSummary()
        log_dir = str(self.params["tensorboard_logging_path"]) + "/{}".format(time.time())
        tensorboard = TensorBoard(log_dir = log_dir)
        learning_rate_reduction = ReduceLROnPlateau(
                                                monitor = 'val_acc', 
                                                patience = 5,
                                                verbose = 1,
                                                factor = 0.85,  # Each patience epoch reduce lr by half
                                                min_lr = 1e-10)
        callbacks = [checkpoint, learning_rate_reduction, tensorboard, telegram_summary]

        # 4. Fit Model
        self.model.summary()
        history = self.model.fit_generator(
                            generator = datagen.flow(x_train, y_train, batch_size = self.params["batch_size"]),
                            epochs = self.params["epochs"],
                            validation_data = (x_val, y_val),
                            verbose = 1,
                            callbacks = callbacks,
                            steps_per_epoch = x_train.shape[0] // self.params["batch_size"])  # // is floor division
        # TODO(oleguer): Log history?
        return
    
    def test(self, data):
        #TODO(oleguer): self.model.predict
        pass

    def analyze(self):
        pass
