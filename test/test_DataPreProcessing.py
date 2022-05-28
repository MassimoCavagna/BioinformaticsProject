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

def test_robust_zscoring():
  df = pd.DataFrame([[10,10,10], [30,30,0], [50,10, -10]])

  df = dp.robust_zscoring(df)
  
  check_df = pd.DataFrame([[-1.,0.,1.], [0.,2.,0.], [1.,0., -1.]])

  assert df.equals(check_df)

def test_constant_features():
  d = {
       "promoters" : pd.DataFrame([[10, 15, 10], [10, 30, 0], [10, 10, -10]], columns = ["a", "b", "c"]),
       "enhancers" : pd.DataFrame([[10, 15, 11], [7, 30, 10], [10, 10, 10]], columns = ["a", "b", "c"])
      }
  dp.constant_features(d)

  d_check = d = {
       "promoters" : pd.DataFrame([[15, 10], [30, 0], [10, -10]], columns = ["b", "c"]),
       "enhancers" : pd.DataFrame([[10, 15, 11], [7, 30, 10], [10, 10, 10]], columns = ["a", "b", "c"])
      }
  assert all([d[k].equals(d_check[k]) for k in d.keys()])

def test_over_sampling():
  df_x = pd.DataFrame([[i for _ in range(10)] for i in range(100)])
  df_y = pd.DataFrame([-1 for _ in range(90)] + [1 for _ in range(10)])
  
  _, y = dp.over_sampling(df_x.values, df_y.values, 1/4)
  _, y2 = dp.over_sampling(df_x.values, df_y.values, 1/4, "SMOTEENN")

  v = ( sum( (y2==1) ) / ( sum(y2==-1) ) )
  assert (sum( (y==1) ) == int(1/4 * ( sum(y==-1) )) and
          round(v, 1) == round(1/4, 1)
         )

def test_drop_outliers():
  d = pd.DataFrame([[-10000,-10000,-10000,-10000,5000],[1,1,1,2,-100],[1,1,69, 2, -100],[1,1,69, 2, -100],[69, 2, -100]], columns = ["a", "b", "c","d","e"])
  labels = pd.DataFrame([1,1,1,1,1])

  dp.drop_outliers(d, labels,1)
  print(d)
  assert 0 not in d.index
