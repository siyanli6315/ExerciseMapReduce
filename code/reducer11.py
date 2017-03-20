#! /usr/bin/env python2.7
# coding=utf-8

import sys
import numpy as np

for line in sys.stdin:
    line=line.strip()
    year,dat=line.split("\t",1)
    dat=eval(dat)

    result={"mean":[],"sigma":[],"median":[],"max":[],"min":[],"1Q":[],"3Q":[]}
    result["mean"]=np.average(dat)
    result["sigma"]=np.std(dat)
    result["median"]=np.median(dat)
    result["max"]=np.max(dat)
    result["min"]=np.min(dat)
    result["1Q"]=np.percentile(dat,25)
    result["3Q"]=np.percentile(dat,75)

    print('%s\t%s' % (year, result))
