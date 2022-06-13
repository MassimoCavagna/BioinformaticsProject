
from typing import Tuple

from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization, ReLU, Concatenate, Layer
from tensorflow.keras.layers import Conv1D, MaxPool1D, AveragePooling1D, GlobalAveragePooling1D, GlobalMaxPool1D, Flatten

from tensorflow.keras import Sequential
from tensorflow.keras.models import Model

from tensorflow.keras import regularizers

from tensorflow.keras.optimizers import Optimizer
from tensorflow.keras.optimizers import SGD, Nadam, Adam

from tensorflow.keras.losses import Loss
from tensorflow.keras.losses import MeanSquaredError, BinaryCrossentropy

from tensorflow.keras.metrics import Metric
from tensorflow.keras.metrics import Accuracy, TruePositives, TrueNegatives, FalsePositives, FalseNegatives, BinaryAccuracy, Precision, Recall, AUC

from sklearn.model_selection import StratifiedShuffleSplit, ShuffleSplit

from tensorflow.data import Dataset

import tensorflow as tf

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

def plot_evaluation_result(results : dict, 
                           train_results : dict, 
                           metrics : list = ["loss", "accuracy"],
                           save : str = "perceptron_results/perceptron_barplots.png",
                           path : str = "/content/gdrive/MyDrive/BioinformaticsData/"
                           ):
  res = pd.DataFrame(results[list(results.keys())[0]])
  res["dataset"] = list(results.keys())[0]
  res["run_type"] = "test"
  res["model"] = "ffnn"
  for k in list(results.keys())[1:]:
    tmp = pd.DataFrame(results[k])
    tmp["dataset"] = k
    tmp["run_type"] = "test"
    tmp["model"] = "ffnn"
    res = pd.concat([res, tmp])

  for k in list(train_results.keys()):
    r = pd.DataFrame(train_results[k])
    validation_results = r[[c for c in r.columns if "val_" in c]].rename(lambda c: c.replace("val_", ""), axis = 'columns')
    r = r[[c for c in r.columns if "val_" not in c]]
    tmp = pd.DataFrame(validation_results)
    tmp["dataset"] = k
    tmp["run_type"] = "validation"
    tmp["model"] = "ffnn"

    res = pd.concat([res, tmp])

    tmp = pd.DataFrame(r)
    tmp["dataset"] = k
    tmp["run_type"] = "train"
    tmp["model"] = "ffnn"

    res = pd.concat([res, tmp])

  res["holdout_n"] = res.index

  l = list(res.columns)
  l.remove("holdout_n")
  l.remove("dataset")
  l.remove("run_type")
  l.remove("model")

  res = res.explode(l)
  res.index = [x//5 for x in res.index]

  res = res[["run_type", "model", "dataset", ] + metrics]

  fig, axs = plt.subplots(2, 1, figsize = (15, 25), squeeze = False)

  for j, label in enumerate(metrics):
    brp = sns.barplot(x = label, y = "dataset", hue = "run_type", data = res[::-1], ax = axs[j][0], capsize=.2)
    axs[j][0].legend(loc = "lower left", fontsize=20)
    axs[j][0].set_title(label)
  plt.savefig(path + save)


def binary_FFNN_model(input_shape : int, 
                      hidden_layers: list,
                      hid_layers_act: str = 'ReLU',
                      outp_layer_act: str = 'sigmoid',
                      optimizer : Optimizer = Adam(learning_rate=.01, ),
                      loss: Loss = BinaryCrossentropy(),
                      metrs: list = [
                                        TruePositives(), 
                                        TrueNegatives(),
                                        FalsePositives(),
                                        FalseNegatives(),
                                        BinaryAccuracy(),
                                        Precision(),
                                        Recall(),
                                        AUC()
                                      ]
                      ) -> Model:
  """ 
  Build the structure of the ffnn classification model
  Parameters:
    - input_shape: the number of input that the model must handle
    - hidden_layers: an iterator containing the amount of neurons in each hidden layer
  Return:
    The compiled model
  """

  # Definition of the input and output (dense) layer
  
  ffnn = Sequential()

  input_layer = Input(shape=(input_shape,))
  output_layer = Dense(1, activation = outp_layer_act)

  # Define and compile the model
  ffnn.add(input_layer)

  for i in hidden_layers:
    ffnn.add( Dense(i, activation = hid_layers_act) )
    ffnn.add( Dropout(0.5))

  ffnn.add(output_layer)

  ffnn.compile(
        optimizer = optimizer,
        loss = loss,
        metrics = metrs
    )
  return ffnn, input_layer

def nested_cross_val(data: pd.DataFrame, labels: pd.DataFrame,
                     learning_rates: list,
                     test_splits: int, validation_splits: int,
                     test_size: float = .1, validation_size: float = .1,
                     ):

  holdout_test = StratifiedShuffleSplit(
      n_splits = test_splits,
      test_size = test_size
  )
  holdout_validation = StratifiedShuffleSplit(
      n_splits = validation_splits,
      test_size = validation_size
  )

  tested_losses = []

  for train_val, test in tqdm(holdout_test.split(x, y), desc= "External",  position = 0, leave = True):

    best_eta = (-1, np.inf) # this tuple keeps track of the best eta for a given 'external' split
    train_val_x, test_x = x[train_val], x[test]
    train_val_y, test_y = y[train_val], y[test]

    
    for train, val in tqdm(holdout_validation.split(train_val_x, train_val_y), desc = "Internal"):

      # the training set is divided again in two sets
      train_x, val_x = train_val_x[train], train_val_x[val]
      train_y, val_y = train_val_y[train], train_val_y[val]

      internal_split_losses = (-1, np.inf) # this tuple keeps track of the best eta for a given 'internal' split

      for eta in learning_rates:
        
        # the network is initialized with the 'eta' in 'learning_rates'
        ffnn, _ = binary_FFNN_model(x.shape[1], 
                                 hidden_layers = [x.shape[1], 32, 16],
                                 optimizer = Adam(learning_rate = eta),
                                 metrs = [ BinaryAccuracy() ])
        
        trained = ffnn.fit(train_x, train_y,
                           epochs=20,
                           verbose = 0,
                           batch_size = int(len(train_x)/100))
        
        loss, _ = ffnn.evaluate(val_x, val_y, verbose = 0)
        
        # here it is checked if the network obtained a better result with the 'eta' it's been fitted with.
        # if so, the learning rate is saved with the loss value.
        if loss < internal_split_losses[1]:
          internal_split_losses = (eta, loss)

      if internal_split_losses[1] < best_eta[1]:
        best_eta = internal_split_losses

    ffnn, _= binary_FFNN_model(x.shape[1], hidden_layers = [x.shape[1], 32, 16], optimizer = Adam(learning_rate = best_eta[0]), metrs = [ BinaryAccuracy()])
    tested = ffnn.fit(train_val_x, train_val_y, epochs=20, validation_data = (test_x, test_y), verbose = 0, batch_size = int(len(train_val_x)/100))
    
    tested_losses.append([best_eta[0], ffnn.evaluate(test_x, test_y, verbose = 0)])

  return tested_losses

# nested_cross_val(x, y, 
#                  list(np.arange(.0, .16, .05)),
#                  test_splits = 5,
#                  validation_splits = 3)

def binary_CNN_model( input_shape : int, 
                      hidden_layers: list = [(32, 16, 4), (32, 16, 4)],
                      hid_layers_act: str = 'ReLU',
                      outp_layer_act: str = 'sigmoid',
                      optimizer : Optimizer = Nadam(learning_rate=.05, ),
                      loss: Loss = BinaryCrossentropy(),
                      metrs: list = [
                                        TruePositives(), 
                                        TrueNegatives(),
                                        FalsePositives(),
                                        FalseNegatives(),
                                        BinaryAccuracy(),
                                        Precision(),
                                        Recall(),
                                        AUC()
                                      ]
                      ) -> Model:
  """ 
  Build the structure of the cnn classification model
  Parameters:
    - input_shape: the number of input that the model must handle
    - hidden_layers: an iterator containing the amount of neurons in each hidden layer, the kernel size and the pooling size
  Return:
    The compiled model
  """   

  # Definition of the input and output (dense) layer
  
  cnn = Sequential()

  input_layer = Input(shape = (input_shape, 1) )

  # Define and compile the model
  cnn.add(input_layer)

  for (size, ks, do) in hidden_layers:

    cnn.add(Conv1D(size, kernel_size = ks, activation="relu"))
    cnn.add(Dropout(.3))
    cnn.add(MaxPool1D(pool_size = do))

  cnn.add(Flatten())
  cnn.add(Dense(32, activation="relu"))

  output_layer = Dense(1, activation = outp_layer_act)
  cnn.add(output_layer)

  cnn.compile(
        optimizer = optimizer,
        loss = loss,
        metrics = metrs
    )
  return cnn, input_layer
