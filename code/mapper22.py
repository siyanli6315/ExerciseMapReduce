#! /usr/bin/env python2.7
# coding=utf-8
'''
逻辑回归的参数估计
'''

import sys
import numpy as np
from sklearn.linear_model import LogisticRegression

dat=list(sys.stdin.readlines())
n=len(dat)
dat=[s.strip() for s in dat]
dat=[s.split(",") for s in dat]
dat=[[float(a) for a in b] for b in dat]

mod=LogisticRegression(penalty='l2',tol=0.0001,C=0.1,fit_intercept=True)

m=n//10
for i in range(10):
    coef={0:[0 for i in range(784)],
        1:[0 for i in range(784)],
        2:[0 for i in range(784)],
        3:[0 for i in range(784)],
        4:[0 for i in range(784)],
        5:[0 for i in range(784)],
        6:[0 for i in range(784)],
        7:[0 for i in range(784)],
        8:[0 for i in range(784)],
        9:[0 for i in range(784)],
        10:[0 for i in range(10)]}
    train=dat[i*m:(i+1)*m]
    x=np.array([b[1:] for b in train])
    y=np.array([b[0] for b in train])
    mod.fit(x,y)
    for i in range(10):
        for j in range(784):
            coef[i][j] += mod.coef_[i][j]
    for i in range(10):
        coef[10][i] += mod.intercept_[i]
    print(coef)
