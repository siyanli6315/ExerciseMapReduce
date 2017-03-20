#! /usr/bin/env python2.7
# coding=utf-8
'''
统计训练集合的样本个数
'''
import sys

dat=[s.strip() for s in sys.stdin.readlines()]
dat=[s.split(",") for s in dat]
y=[s[0] for s in dat]

for i in y:
    print(i)
