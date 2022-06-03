import pandas as pd
import numpy as np

def to_bed(data:pd.DataFrame)->pd.DataFrame:
    """Return bed coordinates from given dataset."""
    return data.reset_index()[data.index.names]
  
def one_hot_encode(data : np.array, categories : np.array = np.array(['A', 'C', 'G', 'T']), ws: int = 256):
  """
  This function retrieve the one hot encoding of the dataset passed
  Params:
    - data: an array containing the sequences to be encoded
    - categories: the array of the categories used for the encoding
    - ws: the window size
  Return:
    The array containing the sequence encoded
  """
  result = []
  for d in data:
    tmp = []
    for l in d.upper():
      tmp.append(np.array(categories == l, dtype = int))
    result.append(tmp)
  del data
  return np.array(result).reshape(-1, ws*4).astype(int)

def to_dataframe(x:np.ndarray, window_size:int, nucleotides:str="ACTG")->pd.DataFrame:
  """
  This function transform the encoded array into a dataframe
  Params:
    - x: an array containing the encoded sequence
    - window_size: the window size
    - nucleotides: teh categories
  Return:
    The dataframe containing the sequence encoded
  """
  return pd.DataFrame(
        x,
        columns = [
            f"{i}{nucleotide}"
            for i in range(window_size)
            for nucleotide in nucleotides
        ]
    )
