#! /usr/bin/env python2.7
# coding=utf-8

from __future__ import division
import sys

count=0
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

b=[0 for i in range(10)]
w=[[0 for i in range(784)] for j in range(10)]

for line in sys.stdin:
    count += 1
    line=line.strip()
    line=eval(line)
    for i in range(10):
        coef[10][i] += line[10][i]
    for i in range(10):
        for j in range(784):
            coef[i][j] += line[i][j]

for i in range(10):
    coef[10][i]=coef[10][i]/count

for i in range(10):
    for j in range(784):
        coef[i][j]=coef[i][j]/count

print(coef)
