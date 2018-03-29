# -*- coding: utf-8 -*-
"""
Created on Thu Mar 29 15:26:58 2018

@author: xiaoziliang_sx
"""

#%%
import matplotlib.pyplot as plt
from main import main
from OPT_algo import sharpe

e_w,n_1,mean_var,perf1,perf2 = main()


plt.plot(e_w,label = "equal weights, sharpe: {0}".format(sharpe(e_w,1)))
plt.plot(n_1,label = "1/N, sharpe: {0}".format(sharpe(n_1,1)))
plt.plot(mean_var,label ="mean variance, sharpe: {0}".format(sharpe(mean_var,1)))
plt.plot(perf1,label ="BSP long_bottom , sharpe:{0}".format(sharpe(perf1,1)))
plt.plot(perf2,label ="BSP long_top , sharpe:{0}".format(sharpe(perf2,1)))
plt.legend()
plt.xlabel("date")
plt.ylabel("PnL")
plt.show()