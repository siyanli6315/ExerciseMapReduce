#! /usr/bin/env python2.7
# coding=utf-8

import sys
import numpy as np

dat=list(sys.stdin.readlines())
n=len(dat)
dat=[s.strip() for s in dat]
dat=[s.split(",") for s in dat]
dat=[[float(a) for a in b[1:]] for b in dat]
np.random.seed(100)
np.random.shuffle(dat)
n=len(dat)
m=n//5

def OLS(X,Y,lam = 0.1):
    n=len(X)
    for i in range(n):
        X[i].append(1)
    X=np.array(X)
    Y=np.array(Y)
    p=X.shape[1]
    n=X.shape[0]

    tmp=np.dot(np.transpose(X),X)
    for i in range(p):
        tmp[i][i]+=lam
    beta=np.dot(np.dot(np.linalg.inv(tmp),np.transpose(X)),Y)
    return beta

for i in range(5):
    X=[s[:15] for s in dat[i*m:(i+1)*m]]
    Y=[s[15] for s in dat[i*m:(i+1)*m]]
    beta=OLS(X,Y,lam=0.1)
    print(list(beta))
