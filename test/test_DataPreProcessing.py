import numpy as np
import pytest
import pandas as pd
import sys, os

sys.path.insert(0, os.path.abspath('/content/BioinformaticsProject'))

import DataPreProcessing as dp

def test_max_axis_nan():
    df_prom = pd.DataFrame([[1, 2, 3], [None, None, None], [4, None, 6]])
    df_enh =  pd.DataFrame([[1, None, 3], [4, None, None], [5, None, 6]])
    
    dic = {'cellLine_promoters_123': df_prom, 'cellLine_enhancers_123': df_enh}
    keys = list(dic.keys())
    columns = df_prom.columns
    
    check_row = pd.DataFrame([3., 2.], columns = ['123'])
    check_row.index.name = 'cellLine_row'
    check_row.index = ["promoters", "enhancers"]
    
    
    check_col = pd.DataFrame([2., 3.], columns = ['123'])
    check_col.index = ["promoters", "enhancers"]
    check_col.index.name = 'cellLine_col'

    check_row_r = pd.DataFrame([round(1., 3), round(2/3, 3)], columns = ['123'])
    check_row_r.index = ["promoters", "enhancers"]
    check_row_r.index.name = 'cellLine_row'
    
    
    check_col_r = pd.DataFrame([round(2/3, 3), round(1., 3)], columns = ['123'])
    check_col_r.index = ["promoters", "enhancers"]
    check_col_r.index.name = 'cellLine_col'

    print(check_row_r)
    print(dp.max_axis_nan(dic, keys, ['123'], 0, relative = True))
    assert (dp.max_axis_nan(dic, keys, ['123'], 0).equals(check_row) and 
           dp.max_axis_nan(dic, keys, ['123'], 1).equals(check_col) and
           dp.max_axis_nan(dic, keys, ['123'], 0, relative = True).equals(check_row_r) and 
           dp.max_axis_nan(dic, keys, ['123'], 1, relative = True).equals(check_col_r))

def test_imputation():
  # epig: dict, f: Callable, knn : bool = False, n_neighbors : int = 5 
  dic = {'cellLine_promoters_123': pd.DataFrame([[1, 2, 3], [None, None, None], [4, None, 6]]), 
          'cellLine_enhancers_123': pd.DataFrame([[1, 7, 3], [4, None, None], [5, None, 6]])
        }
  dic_knn = {'cellLine_promoters_123': pd.DataFrame([[1, 2, 3], [None, None, None], [4, None, 6]]), 
             'cellLine_enhancers_123': pd.DataFrame([[1, 7, 3], [4, None, None], [5, None, 6]])
            }
  
  result = dp.imputation(dic, np.mean)
  result_knn = dp.imputation(dic_knn, dp.knn_imputer, True, 2)
  
  check_result = {'cellLine_promoters_123': pd.DataFrame([[1, 2, 3], [2.5, 2., 4.5], [4, 2., 6]]), 
                  'cellLine_enhancers_123': pd.DataFrame([[1, 7, 3], [4, 7., 4.5], [5, 7., 6]])
                 }
  check_result_knn = {'cellLine_promoters_123': pd.DataFrame([[1., 2., 3.], [2.5, 2., 4.5], [4., 2., 6.]]), 
                      'cellLine_enhancers_123': pd.DataFrame([[1., 7., 3.], [4., 7., 4.5], [5., 7., 6.]])
                     }
                
  assert (all([result[k].equals(check_result[k]) for k in dic.keys()]) and
          all([result_knn[k].equals(check_result_knn[k]) for k in dic.keys()])
         )


