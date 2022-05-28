import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import RobustScaler
from typing import Callable
from tqdm import tqdm

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

#robust_zscoring Functions

def robust_zscoring(df:pd.DataFrame)->pd.DataFrame:
    return pd.DataFrame(
        RobustScaler().fit_transform(df.values),
        columns=df.columns,
        index=df.index
    )

################################################################################

# Constant features

def drop_constant_features(df:pd.DataFrame)->pd.DataFrame:
    """
    Retrieve a list of boolean, one for each feature (column) that if it is True the feature is 
    constant among all rows.
    The list is used to filter the features

    Params:
      df: the dataframe what will be cchecked
    
    Return:
      The dataframe without the constante features
    """
    
    return df.loc[:, (df != df.iloc[0]).any()]

def constant_features(epig: dict):
  """
    Check for each dataframe if there are constant features and drop them

    Params:
      epig: the dictionary
  """
  print("Datasets with cosntant features:")
  for key in epig.keys():
    original = epig[key].shape[1]
    dropped = drop_constant_features(epig[key])
    if(original != dropped.shape[1]):
      print(f"{key}: {original}-->{dropped}")
      print("="*50)
    epig[key] = dropped
  
################################################################################
from imblearn.over_sampling import RandomOverSampler
from imblearn.combine import SMOTEENN

def over_sampling(X : np.array, y : np.array, balance_ratio : float = 1/10, method : str = "Random"):
  """
    Perform oversampling and undersampling on the passed dataframes, 
    looking at y in order to check for which class is the minority
    and which is the majorty
    
    Params:
      X: the values of a dataframe that will be resampled
      y: the values of the labels of the relative rows
      balance_ratio: the balance ratio at which the resampling must achieve
      method: which method use (Random, SMOTEENN)
    Return:
      The resampled X and y
  """
  if method == "Random":
    ros = RandomOverSampler(sampling_strategy = balance_ratio, random_state=0)
    X_resampled, y_resampled = ros.fit_resample(X, y)
  else:
    smote_enn = SMOTEENN(sampling_strategy = balance_ratio, random_state=0)
    X_resampled, y_resampled = smote_enn.fit_resample(X, y)
    
  return X_resampled, y_resampled

################################################################################
import scipy.stats as ss
# Outliers drop

def drop_outliers(df: pd.DataFrame, labels: pd.DataFrame , n_std: float = 3.5):
  """
  This function scans all the features of the given DataFrame and drops (inplace) all the rows (of the corresponding feature)
  whose value is an outliers: a value is considered an outlier if exceed the number of standard deviations passed.
  Params:
    - df: the dataframe to be cleaned
    - labels: the dataframe containing the corresponding labels
    - n_std: the number of stanadrd deviations that must be respected as limit
  Return:
    None: the drop is made 'inplace'
  """

  for col in df.columns:
    sigma = ss.median_absolute_deviation(df[col])
    mu = np.median(df[col])
    to_be_dropped = df.index[np.abs( (0.6745*(df[col]-mu)) / sigma) > n_std]
    
    df.drop(to_be_dropped, inplace = True)
    labels.drop(to_be_dropped, inplace = True)
    
################################################################################

# Correlation
def pearson(epig: dict, labels: dict, uncorrelated: dict, p_value_threshold: float = 0.01, correlation_threshold: float = 0.05):

  p_value_threshold = 0.01
  correlation_threshold = 0.05


  for region, x in epig.items():
      for column in tqdm(x.columns, desc=f"Running Pearson test for {region}\n", dynamic_ncols=True, leave=False):
          correlation, p_value = pearsonr(x[column].values.ravel(), labels[region].values.ravel())
          if p_value > p_value_threshold:
              uncorrelated[region].add(column)


def spearman(epig: dict, labels: dict, uncorrelated: dict, p_value_threshold: float = 0.01):
  for region, x in epig.items():
    for column in tqdm(x.columns, desc=f"Running Spearman test for {region}", dynamic_ncols=True, leave=False):
        correlation, p_value = spearmanr(x[column].values.ravel(), labels[region].values.ravel())
        if p_value > p_value_threshold:
            uncorrelated[region].add(column)

def MINE_corr(epig: dict, labels: dict, uncorrelated: dict, correlation_threshold: float = 0.05):
  for region, x in epig.items():
    for column in tqdm(uncorrelated[region], desc=f"Running MINE test for {region}", dynamic_ncols=True, leave=False):
        mine = MINE()
        mine.compute_score(x[column].values.ravel(), labels[region].values.ravel())
        score = mine.mic()
        if score < correlation_threshold:
            #print(region, column, score)
            print()
        else:
            uncorrelated[region].remove(column)
