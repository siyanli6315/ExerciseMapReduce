#! /usr/bin/env python2.7
# coding=utf-8
'''
逻辑回归的结果检验
'''

from __future__ import division
import numpy as np

test_dat=list(open("test").readlines())
test_dat=[s.strip() for s in test_dat]
test_dat=[s.split(",") for s in test_dat]
test_dat=[[float(x) for x in s] for s in test_dat]
test_x=np.array([s[1:] for s in test_dat])
test_y=np.array([s[0] for s in test_dat])

for coef in open("2参数估计.txt"):
    coef=coef.strip()
    coef=eval(coef)

w=[]
for i in range(10):
    w.append(coef[i])

w=np.array(w)
b=np.array(coef[10])

def logit(test_x,w,b):
    z=np.dot(test_x,np.transpose(w))+np.transpose(b)
    y_hat= 1.0/(1.0+np.exp(-z))
    y_hat2=[]
    for i in y_hat:
        y_hat2.append(np.argmax(i))
    return y_hat2

def acc(y_hat,y):
    accuracy=sum([int(y_hat[i]==y[i]) for i in range(len(y))])/len(y)
    return accuracy

y_hat=logit(test_x,w,b)
acc=acc(y_hat,test_y)

print("Accuracy: {0}".format(acc))
