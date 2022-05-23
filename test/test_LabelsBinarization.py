import pytest
import sys, os
import pandas as pd
sys.path.insert(0, os.path.abspath('/content/BioinformaticsProject'))

import LabelsBinarization

def test_binarize_labels():
  # label_df: pd.DataFrame, threshold: tuple, values_to_return: tuple = (-1,1)
  label_df = pd.DataFrame([1,2,3,4,5])
  threshold = (2, 4)
  values_to_return = (-1,1)
  result = binarize_labels(label_df, threshold, values_to_return)
  check = pd.DataFrame([None, -1, None, 1, 1])
  
  assert result.values == check.values
