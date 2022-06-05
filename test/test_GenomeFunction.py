import pandas as pd
import numpy as np
import pytest

import sys, os

sys.path.insert(0, os.path.abspath('/content/BioinformaticsProject'))

import GenomeFunction as gf

def test_to_bed():
  data = pd.DataFrame([1], columns = ["Test"])
  data.index.name = "TestResult"
  gf.to_bed(data)
  assert "TestResult" in data.index.names
 
  
  
def test_one_hot_encode():
  x = np.array(["A", "C", "G", "T", "N"])
  res = gf.one_hot_encode(x, ws=1)
  check = [[1, 0, 0, 0],
           [0, 1, 0, 0],
           [0, 0, 1, 0],
           [0, 0, 0, 1],
           [0, 0, 0, 0]]
 
  assert all([all(res[i] == check[i]) for i in range(len(check))])

def test_to_dataframe():
  data = [ [1, 0, 0, 0],
           [0, 1, 0, 0],
           [0, 0, 1, 0],
           [0, 0, 0, 1],
           [0, 0, 0, 0]
         ]

  res = gf.to_dataframe(data, 1)
  assert (all(['0A', '0C', '0T', '0G'] == res.columns) and 
          all([all(res.values[i] == data[i]) for i in range(len(data))])
         )
  
