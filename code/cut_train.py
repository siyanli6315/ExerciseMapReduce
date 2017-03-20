#! /usr/bin/env python2.7
# coding=utf-8
'''
分割训练集和测试集
输入数据，打乱之后输出训练集
'''

import sys
import numpy as np

dat=[s.strip() for s in sys.stdin.readlines()]
np.random.seed(10)
np.random.shuffle(dat)

n=len(dat)
train=dat[:n/4*3]

for s in train:
    print(s)
