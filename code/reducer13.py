#! /usr/bin/env python2.7
# coding=utf-8

import sys
import numpy as np

def OLS(X,Y,lam = 0.1):
    X=np.array(X)
    Y=np.array(Y)
    p=X.shape[1]
    n=X.shape[0]
    tmp=np.dot(np.transpose(X),X)
    for i in range(p):
        tmp[i][i]+=lam
    beta=np.dot(np.dot(np.linalg.inv(tmp),np.transpose(X)),Y)
    return beta

def regression(X,Y,coef):
    n=len(X)
    p=len(coef)
    X=np.array(X)
    Y=np.array(Y)
    coef=np.array(coef)
    Y_hat=np.dot(X,coef)
    Y_bar=np.average(Y)
    SSE=np.sum([(Y[i]-Y_hat[i])**2 for i in range(n)])
    RMSE=np.sqrt(SSE/n)
    SSR=np.sum([(Y_hat[i]-Y_bar)**2 for i in range(n)])
    SST=np.sum([(Y[i]-Y_bar)**2 for i in range(n)])
    Rsquare=SSR/SST
    return {"RMSE":RMSE,"Rsquare":Rsquare}

count=0
cv=0
for line in sys.stdin:
    count+=1
    tmp=line.strip()
    tmp=eval(tmp)
    train=[]
    valid=[]
    for i in range(1999,2015):
        train.append(tmp["train"][i])
        valid.append(tmp["valid"][i])
    train=list([list(s) for s in np.transpose(train)])
    valid=list([list(s) for s in np.transpose(valid)])
    x_train=[s[:15] for s in train]
    y_train=[s[15] for s in train]
    for i in range(len(x_train)):
        x_train[i].append(1)
    x_valid=[s[:15] for s in valid]
    y_valid=[s[15] for s in valid]
    for i in range(len(x_valid)):
        x_valid[i].append(1)
    coef=OLS(x_train,y_train,lam=0.1)
    inmod=regression(x_train,y_train,coef)
    outmod=regression(x_valid,y_valid,coef)
    print("k: {0}".format(count))
    print("参数: {0}".format(coef))
    print("Rsquare: {0}".format(inmod["Rsquare"]))
    print("模型内预测误差: {0}".format(inmod["RMSE"]))
    print("模型外预测误差: {0}".format(outmod["RMSE"]))
    cv+=outmod["RMSE"]

print("{0} 折交叉验证误差是：{1}".format(count,cv/count))
