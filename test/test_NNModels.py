import pytest
import json
import sys, os
import pandas as pd
import numpy as np
from tensorflow.keras.optimizers import Adam, Nadam
from tensorflow.keras.metrics import MeanSquaredError
sys.path.insert(0, os.path.abspath('/content/BioinformaticsProject'))

import NNModels as nnm

def test_plot_evaluation_result():
  try:
    with open("perceptron_results.json", "r") as f:
      results = json.load(f)
    with open("perceptron_train_results.json", "r") as f:
      train_results = json.load(f)
    
    nnm.plot_evaluation_result(results, train_results, ["loss", "binary_accuracy"], path = "", save = "perceptron_barplots.png")

    t1 = False
    with open("perceptron_barplots.png", "r") as p:
      pass
    t1 = True
    with open("fake_file.fk", "r") as p:
      pass
    t2 = False
  except:
    t2 = True

  assert t1 and t2

def test_binary_FFNN_model():
  ffnn, _ = nnm.binary_FFNN_model(64, 
                                  hidden_layers = [32, 16, 6],
                                  optimizer = Adam(learning_rate = 0.01),
                                  metrs = MeanSquaredError()
                                 )
  assert (len(ffnn.layers) == 7 and 
          ("Adam" in str(ffnn.optimizer)) and 
          "MeanSquaredError" in str(ffnn.compiled_metrics._metrics)
         )

def test_binary_CNN_model():
  cnn, _ = nnm.binary_CNN_model(1024, 
                                  hidden_layers = [(32, 4, 2), (32, 4, 2)],
                                  optimizer = Nadam(learning_rate = 0.01),
                                  metrs = MeanSquaredError()
                                 )
  assert (len(cnn.layers) == 9 and 
          ("Nadam" in str(cnn.optimizer)) and 
          "MeanSquaredError" in str(cnn.compiled_metrics._metrics)
         )


