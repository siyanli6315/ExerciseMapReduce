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
m=n//10

for i in range(10):
    out={"train":{1999:[],2000:[],2001:[],2002:[],2003:[],2004:[],2005:[],2006:[],2007:[],2008:[],2009:[],2010:[],2011:[],2012:[],2013:[],2014:[]},"valid":{1999:[],2000:[],2001:[],2002:[],2003:[],2004:[],2005:[],2006:[],2007:[],2008:[],2009:[],2010:[],2011:[],2012:[],2013:[],2014:[]}}
    dat_valid=[]
    dat_train=[]
    for j in range(i*m,(i+1)*m):
        dat_valid.append(dat[j])
    for j in range(0,i*m):
        dat_train.append(dat[j])
    for j in range((i+1)*m,n):
        dat_train.append(dat[j])
    n_train=len(dat_train)
    n_valid=len(dat_valid)
    for j in range(16):
        for k in range(n_train):
            out["train"][j+1999].append(dat_train[k][j])
    for j in range(16):
        for k in range(n_valid):
            out["valid"][j+1999].append(dat_valid[k][j])
    print(out)
