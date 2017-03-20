#!/bin/sh

MY_LOCAL=/home/lifeng/exam/cufe_lisiyan

#第一题：回归分析
cat $MY_LOCAL/data/gdp.csv | \
$MY_LOCAL/code/mapper11.py | \
$MY_LOCAL/code/reducer11.py > \
$MY_LOCAL/data/1描述统计.txt

cat $MY_LOCAL/data/gdp.csv | \
$MY_LOCAL/code/mapper12.py | \
$MY_LOCAL/code/reducer12.py > \
$MY_LOCAL/data/1参数估计.txt

cat $MY_LOCAL/data/gdp.csv | \
$MY_LOCAL/code/mapper13.py | \
$MY_LOCAL/code/reducer13.py > \
$MY_LOCAL/data/1交叉验证.txt


#第二题：逻辑回归
cat $MY_LOCAL/data/mnist | \
$MY_LOCAL/code/cut_train.py > \
$MY_LOCAL/data/train

cat $MY_LOCAL/data/mnist | \
$MY_LOCAL/code/cut_test.py > \
$MY_LOCAL/data/test

cat $MY_LOCAL/data/train | \
$MY_LOCAL/code/mapper21.py | \
$MY_LOCAL/code/reducer21.py > \
$MY_LOCAL/data/2描述统计.txt

cd $MY_LOCAL/data
cat 2参数估计.txt | ./test2.py
