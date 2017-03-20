#! /usr/bin/env python2.7
# coding=utf-8

from __future__ import division
import sys

coef={1999:0,2000:0,2001:0,2002:0,2003:0,2004:0,2005:0,2006:0,2007:0,2008:0,2009:0,2010:0,2011:0,2012:0,2013:0,"intercept":0}
count=0

for line in sys.stdin:
    count+=1
    line=line.strip()
    line=eval(line)
    for i in range(15):
        coef[i+1999]+=line[i]
    coef["intercept"]+=line[15]

for i in range(15):
    coef[i+1999]=coef[i+1999]/count
coef["intercept"]=coef["intercept"]/count

print(coef)
