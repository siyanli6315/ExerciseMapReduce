#! /usr/bin/env python2.7
# coding=utf-8

import sys
import numpy as np

result={0:0,1:0,2:0,3:0,4:0,5:0,6:0,7:0,8:0,9:0}
for line in sys.stdin:
    line=line.strip()
    line=int(line)
    result[line]+=1

print(result)
