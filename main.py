# -*- coding: utf-8 -*-
"""
Created on Thu Mar 29 15:26:55 2018

@author: xiaoziliang_sx
"""

#%%

# execuion part main function
import pandas as pd
from OPT_algo import equal_weights, one_n, mean_variance_hsc,BSP
import os
#%%
#----------------here please select data file you want in the type of csv(pandas dataframe)
#os.chdir('C://Users//Administrator//Desktop//my//OPT')
raw = pd.read_csv('./all_20150104_20170901.csv',index_col = 0)
#raw = raw[(raw.index >= 20080101)  & (raw.index < 20170810)]
dt_idx = pd.to_datetime(raw.index,format = "%Y%m%d",errors='raise')
raw = pd.DataFrame(raw.values,index = dt_idx,columns = raw.columns)



#%%
#--------------------main function, execution part

def main():
    e_w = equal_weights(raw, 1)
    n_1 = one_n(raw, 1)
    mean_var = mean_variance_hsc(raw, 12, 1)
    bsp = BSP(raw,5,12)
    perf1 = bsp.portfolio_perf((0.05,0.15))
    #perf1 = bsp.portfolio_perf((0.,1.))
    perf2 = bsp.portfolio_perf((0.85,0.95))
    
    return e_w,n_1,mean_var,perf1,perf2
    
