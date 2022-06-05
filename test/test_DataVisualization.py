import numpy as np
import pytest
import pandas as pd
import sys, os

sys.path.insert(0, os.path.abspath('/content/BioinformaticsProject'))

import DataVisualization as dv


def test_pca():
  data = np.array([np.random.randint(0,100, 50) for _ in range(50)])
  data = dv.pca(data, 2)
  assert data.shape == (50,2)
  
# def test_cannylab_tsne():
#   assert False
