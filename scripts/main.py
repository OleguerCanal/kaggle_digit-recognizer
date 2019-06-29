import pandas as pd
import numpy as np
from preprocessing import GetProcessedData
from architectures import SimpleCnnClassification
from keras.optimizers import RMSprop
from datagenerators import GetSimpleImageAugmentation
from results_analysis import PlotHistory

if __name__ == "__main__":
    # 1. Load data
    raw_train = pd.read_csv("../input/train.csv")
    raw_test =  pd.read_csv("../input/test.csv")

    raw_train = raw_train.sample(frac=0.01)

    # 2. Process data
    x_train, y_train, x_val, y_val, x_test = GetProcessedData(raw_train, raw_test)
    del raw_train, raw_test

    # 3. Define Model
    epochs = 2 # Turn epochs to 30 to get 0.9967 accuracy
    batch_size = 10
    optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
    loss = "categorical_crossentropy"
    metrics = ["accuracy"]

    model = SimpleCnnClassification(input_shape = x_train[0].shape)
    model.compile(optimizer = optimizer, loss = loss, metrics = metrics)

    # 4. Get DataGenerator (image augmentation)
    datagen = GetSimpleImageAugmentation()
    datagen.fit(x_train)

    # 5. Fit Model
    history = model.fit_generator(
                        generator = datagen.flow(x_train, y_train, batch_size=batch_size),
                        epochs = epochs,
                        validation_data = (x_val, y_val),
                        verbose = 2,
                        steps_per_epoch = x_train.shape[0])
    
    # 6 Analyze results
    PlotHistory(history)
    y_pred = model.predict(x_val)
    y_pred_classes = np.argmax(y_pred,axis = 1)  # One-hot vector (get max prob)
    y_true = np.argmax(y_val,axis = 1)  # One-hot vecotr
    confusion_mtx = confusion_matrix(y_true, y_pred_classes) 
    plot_confusion_matrix(confusion_mtx, classes = range(10))