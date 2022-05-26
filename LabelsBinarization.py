import pandas as pd

def binarize_labels(label_df: pd.DataFrame, threshold: tuple, values_to_return: tuple = (0,1))->pd.DataFrame:
  """
  This functions binarize the labels in the 'label_df' according to the given threshold.
  
  Parameters:
   - label_df: the dataframe to be binarized
   - threshold: a tuple describing an interval e.g. (0, 5)
   - values_to_return: a tuple describing the outputs (by default is set to (-1, 1)
  
  Output: a new Pandas DataFrame containg:
            - -1 if the value in the row is equal to the first value of the threshold tuple
            -  1 if the value in the row is greater or equal to the second value of the threshold tuple
            The values in between are setted to None
  """
  f = lambda value: values_to_return[0] if value == threshold[0]\
                                        else ( values_to_return[1] if value >= threshold[1] 
                                                                   else None
                                              )     
  return label_df.copy().applymap(f)

def binarize_and_drop(data: dict, labels: dict, thresholds: list = [(0, 1), (0, 5)], values_to_return: tuple = (0,1)):
  """
  This function calls the 'binarize_labels' function and drops (inplace) the None values obtained both on the labels' dataframes
  Parameters:
   - data: is the dictionary containing all the points dataframes
   - labels: is the dictionary containing all the labels of the points in the 'data' dictionary
   - thresholds: the list of 2 tuple to be used in the 'binarize_labels' funciton
                NOTE: the first tuple is for the 'enhancers' data, whilst the second for the 'promoters'
  Output:
    None
  """
  for l in labels:
    th = thresholds[0] if 'enhancers' in l else thresholds[1]
    labels[l] = binarize_labels(labels[l], th)
    to_be_dropped = list(labels[l][labels[l].isna().values].index)
    labels[l].dropna(inplace = True)
    data[l].drop(to_be_dropped, inplace = True)
