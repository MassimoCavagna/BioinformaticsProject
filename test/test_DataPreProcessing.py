import pytest
import pandas as pd
import sys, os

sys.path.insert(0, os.path.abspath('/content/BioinformaticsProject'))

import DataPreProcessing.max_axis_nan as dp_man
import DataPreProcessing.imputation as dp_i

def test_max_axis_nan():
    df_prom = pd.DataFrame([[1, 2, 3], [None, None, None], [4, None, 6]])
    df_enh =  pd.DataFrame([[1, None, 3], [4, None, None], [5, None, 6]])
    
    dic = {'cellLine_promoters_123': df_prom, 'cellLine_enhancers_123': df_enh}
    keys = ['promoters', 'enhancers']
    columns = df_prom.columns
    
    
    check_row = pd.DataFrame([3, 2], columns = ['123'])
    check_row.set_index(keys)
    check_row.index.name = 'cellLine_row'
    
    check_col = pd.DataFrame([2, 3], columns = ['123'])
    check_col.set_index(keys)
    check_col.index.name = 'cellLine_col'
    
    assert dp_man(dic, keys, ['123'], 0).equals(check_row) and dp_man(dic, keys, ['123'], 1).equals(check_col)