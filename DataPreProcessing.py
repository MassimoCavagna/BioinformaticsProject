import pandas as pd
from sklearn.impute import KNNImputer
from typing import Callable

def max_axis_nan(epig: dict, keys: list, columns: list,  axis: int = 0, relative : bool = False) -> pd.DataFrame:
  """
  This function finds the maximum number of Nan values over the given axis
  Params:
    epig: the dict containint the epigenomic data divided into promoters and enhancers.
    keys: a list of the keys used to access the epig dict.
    axis: 0 the maximum number of NaN in a row, 1 same by column
    relative: divides the number of NaN by the number of rows (axis = 0) or columns (axis = 1)
  Returns:
    a pandas DataFrame with two rows (one for the promoters, one for the enhancers) and as much columns as the
    number of window sizes
  """
  df = pd.DataFrame(columns = columns)
  row_max_nan_promoters = []
  row_max_nan_enhancers = []

  for key in keys:
    total_nan = epig[key].isna().sum(1-axis).max() / (epig[key].shape[1-axis] if relative else 1) 
    if 'promoters' in key:
      row_max_nan_promoters.append(round(total_nan, 3))
    else:
      row_max_nan_enhancers.append(round(total_nan, 3))

  df.loc[len(df.index)] = row_max_nan_promoters
  df.loc[len(df.index)] = row_max_nan_enhancers
  df.index = ['promoters', 'enhancers']
  df.index.name = keys[0].split('_')[0] + ('_row' if axis == 0 else '_col')
  return df

################################################################################

#Imputation Functions


def knn_imputer(dataset : pd.DataFrame, n_neighbors : int = 5):
  """
    This function fills the NaN values with the imputation from the neighbours of each NaN
    Params:
      dataset: the dataset that must be filled
      n_neighbors: the number of neighbours considered in the imputation
    Return:
      A pandas DataFrame with the NaN filled with the imputation
  """
  return pd.DataFrame(KNNImputer(n_neighbors = n_neighbors).fit_transform(dataset.values),
                      columns=dataset.columns,
                      index=dataset.index
                      )

def imputation(epig: dict, f: Callable, knn : bool = False, n_neighbors : int = 5 ):
  """
    This function fills the NaN values with the given method
    Params:
      epig: the dict containing the epigenomic data divided into promoters and enhancers.
      f: a function representing the imputation method (mean, median, mode, nearest_neighbor...)
      knn: if it is used the knn_imputer
      n_neighbors: the number of neighbours considered in the imputation
  """
  keys = epig.keys()

  for key in keys:
    if knn:
      epig[key] = f(epig[key], n_neighbors)
    else:
      fill_values = epig[key].apply(f, axis = 0)
      for col in fill_values.index: 
        epig[key][col] = epig[key][col].fillna(fill_values[col])
  return epig

################################################################################

def robust_zscoring(df:pd.DataFrame)->pd.DataFrame:
    return pd.DataFrame(
        RobustScaler().fit_transform(df.values),
        columns=df.columns,
        index=df.index
    )
