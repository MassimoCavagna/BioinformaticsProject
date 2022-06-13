import numpy as np
import pytest
import pandas as pd
import sys, os

sys.path.insert(0, os.path.abspath('/content/BioinformaticsProject/v_test'))

import DataVisualization as dv


def test_pca():
  data = np.array([np.random.randint(0,100, 50) for _ in range(50)])
  data = dv.pca(data, 2)
  assert data.shape == (50,2)
  
def test_cannylab_tsne():
  data = np.array([np.random.randint(0,100, 50) for _ in range(60)])
  res = dv.cannylab_tsne(data, 50)
  assert res.shape == (60, 2)
