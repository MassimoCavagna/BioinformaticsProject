import numpy as np
import pytest
import pandas as pd
import sys, os

sys.path.insert(0, os.path.abspath('/content/BioinformaticsProject'))

import DataDownload as dd

def test_download_cell_lines():
  line = ["A549"]
  d1, d2 = dd.download_cell_lines(line)
  print(len(d1.keys()))
  assert len(d1.keys()) == 10
