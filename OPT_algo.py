# -*- coding: utf-8 -*-
"""
Created on Mon Nov 06 15:58:53 2017

@author: xzl
"""

#import glob
import pandas as pd
import numpy as np
from itertools import combinations as comb
import time
import scipy.optimize

#%%
#some basic functions
def assets_historical_returns_and_covariances(prices):
    # create matrix of historical returns
    #rows, cols = ret.T.shape
    # calculate returns
    returns = prices.pct_change().mean(0) 
    expreturns = (1+returns) ** 252 - 1  # Annualize returns
    # calculate covariances 
    covars = prices.pct_change().cov()
    covars = covars * 252  # Annualize covariances
    return expreturns, covars

# Calculates portfolio mean return
def port_mean(W, R):
    return (W*R).sum()

# Calculates portfolio variance of returns
def port_var(W, C):
    return C.dot(W).dot(W)

# Combination of the two functions above - mean and variance of returns calculation
def port_mean_var(W, R, C):
    return port_mean(W, R), port_var(W, C)

def solve_frontier(R, C):
    def fitness(W, R, C, r):
        # For given level of return r, find weights which minimizes portfolio variance.
        mean, var = port_mean_var(W, R, C)
        penalty = 100 * abs(
            mean - r)  # Big penalty for not meeting stated portfolio return effectively serves as optimization constraint
        return var + penalty

    #frontier_mean, frontier_var = [], []
    
    r = round(R.mean(),2)
    
    n = len(R)  # Number of assets in the portfolio
    W = np.ones([n]) / n  # start optimization with equal weights
    b_ = [(0, 1) for i in range(n)]
    c_ = ({'type': 'eq', 'fun': lambda W: sum(W) - 1.})
    
    
    optimized = scipy.optimize.minimize(fitness, W, (R, C, r), method='SLSQP', constraints=c_, bounds=b_)
    if not optimized.success:
        raise BaseException(optimized.message)
        # add point to the efficient frontier [x,y] = [optimized.x, r]
        #frontier_mean.append(r)
        #frontier_var.append(port_var(optimized.x, C))
    #return array(frontier_mean), array(frontier_var)
    return optimized.x

#%%

def equal_weights(data, ndaygap:int):
    print("-----------starting equal weights-----------")

    #first, calculate daily return time series
    rtns = data.pct_change(ndaygap)
    rtns.fillna(0,inplace=True)
    grtns=rtns[::ndaygap]
    #next, compound the returns; this will assign 1 to all strategies on the first day
    netv = np.exp(np.log1p(grtns).cumsum())
    #then, we return the simple average
    print("-----------done equal weights-----------")
    return netv.mean(1)
    
    
    
    #return (data/data.iloc[0]).mean(1)[::ndaygap]

def one_n(data,ndaygap:int):
    print("-----------starting one over n hsc-----------")

    #rebalance frequency in num of trading days
    gap = ndaygap
    
    #then, calculate return time series
    grtns = data.pct_change(gap)
    grtns.fillna(0,inplace=True)
    dgrtns=grtns[::gap]
    
    #next, we calculate average gap-based return
    oonrtn=dgrtns.mean(1) #oon: one over n
    
    #next, compound the returns; this will assign 1 to all strategies on the first day
    netv = np.exp(np.log1p(oonrtn).cumsum())
    #then, we return the simple average

    print("-----------done one over n hsc-----------")
    return netv

def mean_variance(data):
    print("-----------mean variance-----------")
    s_y = data.index.year[0]
    s_m = data.index.month[0]
    e_y = data.index.year[-1]
    e_m = data.index.month[-1]
    p = data.shape[1]
    weights = np.ones([p])/p   
    cash = 1.
    mean_var = pd.Series(index = data.index)
    yr =[n for n in range(s_y,e_y+1)]
    mth = [ n+1 for n in range(12)]
    yy,mm =np.meshgrid(yr,mth)
    yy,mm = yy.T.ravel(),mm.T.ravel()
    for y,m in zip(yy,mm):
        if y>=e_y and m > e_m:
            pass
        elif y <= s_y and m <s_m:
            pass
        else:
            print(y,m)
            #print(weights)
            select = (data.index.year == y)&(data.index.month == m)
            if y == s_y and m == s_m:
                mean_var[select] = (data[select]/data[select].iloc[0]).dot(cash*weights)
            else:
                mean_var[select] = (data[select]/data.shift(1)[select].iloc[0]).dot(cash*weights)
            #update weights and cash at the end of months
            cash = mean_var[select].iloc[-1]
            R,C = assets_historical_returns_and_covariances(data[select])
            weights = solve_frontier(R,C)
    return mean_var

def mean_variance_hsc(data, riskLookBackWinInGaps:int, ndaygap:int):
    print("-----------starting mean variance-----------")

    #check if data size is large enough along time
    if riskLookBackWinInGaps*ndaygap>=data.shape[0]:
        return None
    
    #rebalance frequency in num of trading days
    gap = ndaygap
    
    #then, calculate return time series
    grtns = data.pct_change(gap)
    grtns.fillna(0,inplace=True)
    dgrtns=grtns[::gap]

    p = dgrtns.shape[1]
    weights = np.ones([p])/p   
    cash = 1.
    mean_var_netv = pd.Series(index = dgrtns[riskLookBackWinInGaps:len(dgrtns)].index)

    for dateIdx in range(riskLookBackWinInGaps,len(dgrtns)):
        print("===>date="+str(dgrtns.index[dateIdx]))
        #construct return series and covariance matrix
        returns = dgrtns[dateIdx-riskLookBackWinInGaps:dateIdx].mean(0)
        annualRtns = (1+returns)**(252./20)-1
        #annualRtns = (1+returns)**(250/gap)-1
        covariance = dgrtns[dateIdx-riskLookBackWinInGaps:dateIdx].cov()
        covariance = covariance * (252./20)
        #covariance = covariance * (250/gap)
        
        #put in optimizer
        weights = solve_frontier(annualRtns,covariance)
        mean_var_netv[dgrtns.index[dateIdx]] = cash * (1+dgrtns.iloc[dateIdx].dot(weights))
        cash = mean_var_netv[dgrtns.index[dateIdx]]

    print("-----------done mean variance-----------")
    return mean_var_netv/mean_var_netv.iloc[0]

#%%
class BSP(object):
    def __init__(self,data:pd.DataFrame,ndaygap:int,lookbackwindow:int):
        self.data = data[::ndaygap]
        self.lookbackwindow = lookbackwindow
        
        self.rtns = self.data/self.data.iloc[0]
#        self.opt_weights = {}
#        self.final_weights = {}
#        self.pairs = []
        self.perfs = []
        
        
    
    def solve_opt(self,sel,method = "opt"):
        if method == "opt":        
            def obj_func(W,sel):
                ret = (sel.pct_change()+1).dot(W).prod(0)
                return -ret
                
            n = len(sel.columns)    
            W = np.ones([n])/n
            b_ = [(0,1) for i in range(n)]
            c_ = ({'type':'eq', "fun": lambda W: sum(W) - 1.})
            optimized = scipy.optimize.minimize(obj_func, W,(sel),method = 'SLSQP', constraints = c_, bounds = b_)
            if not optimized.success:
                raise BaseException(optimized.message)
                           
            return optimized.x    
        elif method == "w_allo":
            ret = (sel.pct_change()+1).prod(0)
            max_id = ret.idxmax()
            ret[:] = 0
            ret[max_id] = 1.
            return np.array(ret)
        
               
    def pair_iter(self,pair_size,lookbackwindow):
        self.opt_weights = {}
        self.final_weights = {}
        self.pairs = []
        for pair in comb(self.rtns.columns,pair_size):
            self.pairs.append(pair)    
            self.opt_weights.update({pair:[np.array([0.5,0.5])]*lookbackwindow})
            self.final_weights.update({pair:[np.array([0.5,0.5])]*lookbackwindow})
        
    
    def OptimalWeights(self,pair:tuple,data_sel):
        avg_weights = np.array([0.5,0.5])
        sel = data_sel[list(pair)]
        # here, we simplfy optimal problem, replacing it with "weight allocation”. 
        # Or you could change method ="opt"(default para)
        weights = self.solve_opt(sel,method = "w_allo")   
        rt = sel.iloc[-1]/sel.iloc[-2]
        avg_perf = rt.dot(avg_weights)
        opt_perf = rt.dot(weights)
             
        
        if opt_perf > avg_perf:
            self.opt_weights[pair].append(weights)
        else:
            self.opt_weights[pair].append(avg_weights)
        
    
    
    def ErrorRate(self,pair:tuple,nthTerm):
        numOfError = 0.
        for i in range(nthTerm):
            if (self.final_weights[pair][i] != self.opt_weights[pair][i+1]).any():
                numOfError+=1
                
        rate = numOfError/nthTerm        
        #print("cumulative error ratio: %s" %rate)            
        return rate
    
    def KNN(self,pair:tuple,Kstep,nthTerm,errorcon):
        if nthTerm - Kstep >=0:
            numC1 = 0
            numC2 = 0
            for i in range(nthTerm-Kstep,nthTerm):
                if (self.opt_weights[pair][i] == self.opt_weights[pair][i-1]).all():
                    numC1 += 1
                else:
                    numC2 += 1
            if numC1 > numC2 and self.ErrorRate(pair,i) < errorcon:
                self.final_weights[pair].append(self.opt_weights[pair][nthTerm])
        
            else:
                self.final_weights[pair].append(np.array([0.5,0.5]))
        else:
            print("Wrong input, term has to be larger than Given K-Value")
    
    
 
    def portfolio_perf(self,pct_range:tuple):
        # tuple as input
        # pct_range = (0.05,0.15)  long bottom 5%-15% 
        # pct_range = (0.90,0.95)  long top 5%-10% 
        
        start_time = time.time()
        lookbackwindow = self.lookbackwindow
        bsp_perf = pd.Series(0.0,index = self.rtns.index)
        self.lp = []
        self.pair_iter(2,lookbackwindow)
        length = len(self.pairs[int(len(self.pairs)*pct_range[0]):int(len(self.pairs)*pct_range[1])])
        pct = 1./length
        
        for term in range(lookbackwindow,len(self.rtns)):
            print(self.rtns.index[term],term)
            for pair in self.pairs:
                #-- way 1, use cummulative return as input.
                # sel1 = self.rtns[:term]
                #-- way 2,use term return as input.
                sel1 = self.rtns[term-lookbackwindow:term]
                #-- for some pair, term(time), firstly, we call OptimalWeights function to 
                #-- assign weights, a comparation between average performance and optimal performamce 
                #-- Result is stored in self.opt_weights
                self.OptimalWeights(pair,sel1)
                #-- secondly, we use KNN method to filter the weights that chosen from previous step.
                #-- Result is stored in self.final_weights.
                self.KNN(pair,lookbackwindow,term,0.7)
                W = self.final_weights[pair][term]
                # print("verify length :%s" % len(self.final_weights[pair]))
                sel2 = self.rtns[list(pair)][term-lookbackwindow:term]
                self.perfs.append(sel2.dot(W).prod(0))
            #-- selected pairs' performance            
            perfs_arr =np.array(self.perfs)
            sort_pairs = np.array(self.pairs)[perfs_arr.argsort()]
            long_pair = sort_pairs[int(len(sort_pairs)*pct_range[0]):int(len(sort_pairs)*pct_range[1])]
            
            for pair in long_pair:
                perf = self.rtns.iloc[term][pair].dot(self.final_weights[tuple(pair)][-1])*pct
                bsp_perf[term] += perf 
            self.lp.append(long_pair)
            self.perfs = []    
            
            
        bsp_perf = bsp_perf[bsp_perf.index >= bsp_perf.index[lookbackwindow]]
        self.bsp_perf_ = bsp_perf/bsp_perf.iloc[0]
        
        
        # output bsp performance 
        print("---- %s seconds ---" % (time.time() - start_time))    
        return self.bsp_perf_    
        
        
    def WeightConversion(self):
        weight_df = pd.DataFrame(0.,index = self.data.index,columns = self.data.columns)
        
        
        for term,pairs in enumerate(self.lp):
            print(weight_df.index[term+self.lookbackwindow])
            for pair in pairs:
                for i,unit in enumerate(pair):
                    weight_df.iloc[term+self.lookbackwindow][unit] += self.final_weights[tuple(pair)][term+self.lookbackwindow][i]
        # This equation can only applly to senario pair_size == 2        
        #weight_df = weight_df/(len(weight_df)-1)/len(weight_df)
        weight_df = weight_df.iloc[self.lookbackwindow:]
        weight_normal = (weight_df.T/weight_df.T.sum(0)).T
        return weight_normal    
      
#%%    
        
def sharpe(series, ndaygap:int):
    if ndaygap<=0:
        return 0
    
    r = series.diff().dropna()
    sha = np.sqrt(250/20)*r.mean()/r.std()
    #sha = np.sqrt(250/ndaygap)*r.mean()/r.std()
    return sha



