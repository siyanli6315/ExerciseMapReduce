#! /usr/bin/env python2.7
# coding=utf-8

import sys

dat=list(sys.stdin.readlines())
n=len(dat)
dat=[s.strip() for s in dat]
dat=[s.split(",") for s in dat]
dat=[[float(a) for a in b[1:]] for b in dat]

result={1999:[],2000:[],2001:[],2002:[],2003:[],2004:[],2005:[],2006:[],2007:[],2008:[],2009:[],2010:[],2011:[],2012:[],2013:[],2014:[]}

for j in range(16):
    for i in range(n):
        result[j+1999].append(dat[i][j])

for i in range(16):
    print('%s\t%s' % (i+1999, result[i+1999]))
