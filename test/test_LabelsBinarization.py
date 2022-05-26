import pytest
import sys, os
import pandas as pd
import numpy as np
sys.path.insert(0, os.path.abspath('/content/BioinformaticsProject'))

import LabelsBinarization as lb

def test_binarize_labels():
  # label_df: pd.DataFrame, threshold: tuple, values_to_return: tuple = (0,1)
  label_df = pd.DataFrame([1,2,3,4,5])
  threshold = (2, 4)
  values_to_return = (0,1)
  result = lb.binarize_labels(label_df, threshold, values_to_return)
  check = pd.DataFrame([None, 0, None, 1, 1])
  print(check)
  print(result)
  assert result.equals(check)


def test_binarize_and_drop():
  # data: dict, labels: dict, thresholds: list
  data = {"cellLine_promoters_windowsize" : pd.DataFrame([[1,2,3], [1,2,3], [1,2,3], [1,2,3], [1,2,3]], columns = ["a", "b", "c"]),
          "cellLine_enhancers_windowsize" : pd.DataFrame([[1,2,3], [1,2,3], [1,2,3], [1,2,3], [1,2,3]], columns = ["a", "b", "c"])}
  labels = {"cellLine_promoters_windowsize" : pd.DataFrame([0,2,3,4,5], columns = ["label"]),
            "cellLine_enhancers_windowsize" : pd.DataFrame([0,2,3,4,5], columns = ["label"])}
  thresholds = [(0, 1), (0, 5)]

  lb.binarize_and_drop(data, labels, thresholds)

  check_data = {"cellLine_promoters_windowsize" : pd.DataFrame([[1,2,3], [1,2,3]], columns = ["a", "b", "c"]).set_index(pd.Index([0,4])),
                "cellLine_enhancers_windowsize" : pd.DataFrame([[1,2,3], [1,2,3], [1,2,3], [1,2,3], [1,2,3]], columns = ["a", "b", "c"])}
  check_labels = {"cellLine_promoters_windowsize" : pd.DataFrame([0., 1.], columns = ["label"]).set_index(pd.Index([0,4])),
                  "cellLine_enhancers_windowsize" : pd.DataFrame([0,1,1,1,1], columns = ["label"])}
  print(labels["cellLine_promoters_windowsize"])
  assert ( all([data[k].equals(check_data[k]) for k in data.keys()]) 
           and all([labels[k].equals(check_labels[k]) for k in labels.keys()]) )
