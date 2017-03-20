#!/bin/sh

#环境变量
HADOOP_STREAMING=/home/dmc/hadoop/share/hadoop/tools/lib/hadoop-streaming-2.6.0.jar
MY_HADOOP=/user/lifeng/cufe_lisiyan
MY_LOCAL=/home/lifeng/exam/cufe_lisiyan

#创建环境并上传代码与数据
hadoop fs -rm R $MY_HADOOP
hadoop fs -mkdir $MY_HADOOP

hadoop fs -put $MY_LOCAL/data/gdp.csv $MY_HADOOP/
hadoop fs -put $MY_LOCAL/data/mnist $MY_HADOOP/
hadoop fs -put $MY_LOCAL/data/train $MY_HADOOP/
hadoop fs -put $MY_LOCAL/data/test $MY_HADOOP/

hadoop fs -put $MY_LOCAL/code/mapper11.py $MY_HADOOP/
hadoop fs -put $MY_LOCAL/code/mapper12.py $MY_HADOOP/
hadoop fs -put $MY_LOCAL/code/mapper13.py $MY_HADOOP/
hadoop fs -put $MY_LOCAL/code/mapper21.py $MY_HADOOP/
hadoop fs -put $MY_LOCAL/code/mapper21.py $MY_HADOOP/
hadoop fs -put $MY_LOCAL/code/reducer11.py $MY_HADOOP/
hadoop fs -put $MY_LOCAL/code/reducer12.py $MY_HADOOP/
hadoop fs -put $MY_LOCAL/code/reducer13.py $MY_HADOOP/
hadoop fs -put $MY_LOCAL/code/reducer21.py $MY_HADOOP/
hadoop fs -put $MY_LOCAL/code/reducer22.py $MY_HADOOP/
hadoop fs -put $MY_LOCAL/code/cut_test.py $MY_HADOOP/
hadoop fs -put $MY_LOCAL/code/cut_train.py $MY_HADOOP/

#第一题：回归分析
hadoop fs -rmr $MY_HADOOP/output11
hadoop jar $HADOOP_STREAMING \
    -input $MY_HADOOP/gdp.csv \
    -output $MY_HADOOP/output11 \
    -mapper "python mapper11.py" \
    -reducer "python reducer11.py" \
    -file $MY_LOCAL/code/mapper11.py $MY_LOCAL/code/reducer11.py
hadoop fs -get $MY_HADOOP/output11/part-00000 $MY_LOCAL/data/1描述统计hadoop.txt

hadoop fs -rmr $MY_HADOOP/output12
hadoop jar $HADOOP_STREAMING \
    -input $MY_HADOOP/gdp.csv \
    -output $MY_HADOOP/output12 \
    -mapper "python mapper12.py" \
    -reducer "python reducer12.py" \
    -file $MY_LOCAL/code/mapper12.py $MY_LOCAL/code/reducer12.py
hadoop fs -get $MY_HADOOP/output12/part-00000 $MY_LOCAL/data/1参数估计hadoop.txt

hadoop fs -rmr $MY_HADOOP/output13
hadoop jar $HADOOP_STREAMING \
    -input $MY_HADOOP/gdp.csv \
    -output $MY_HADOOP/output13 \
    -mapper "python mapper13.py" \
    -reducer "python reducer13.py" \
    -file $MY_LOCAL/code/mapper13.py $MY_LOCAL/code/reducer13.py
hadoop fs -get $MY_HADOOP/output13/part-00000 $MY_LOCAL/data/1交叉验证hadoop.txt

#第二题：逻辑回归
hadoop fs -rmr $MY_HADOOP/output_train
hadoop jar $HADOOP_STREAMING \
    -input $MY_HADOOP/mnist \
    -output $MY_HADOOP/output_train \
    -mapper "/bin/cat" \
    -reducer "python cut_train.py" \
    -file $MY_LOCAL/code/cut_train.py
hadoop fs -get $MY_HADOOP/output_train/part-00000 $MY_LOCAL/data/train_hadoop

hadoop fs -rmr $MY_HADOOP/output_test
hadoop jar $HADOOP_STREAMING \
    -input $MY_HADOOP/mnist \
    -output $MY_HADOOP/output_test \
    -mapper "/bin/cat" \
    -reducer "python cut_test.py" \
    -file $MY_LOCAL/code/cut_test.py
hadoop fs -get $MY_HADOOP/output_test/part-00000 $MY_LOCAL/data/test_hadoop

hadoop fs -rmr $MY_HADOOP/output21
hadoop jar $HADOOP_STREAMING \
    -input $MY_HADOOP/train \
    -output $MY_HADOOP/output21 \
    -mapper "python mapper21.py" \
    -reducer "python reducer21.py" \
    -file $MY_LOCAL/code/mapper21.py $MY_LOCAL/code/reducer21.py
hadoop fs -get $MY_HADOOP/output21/part-00000 $MY_LOCAL/data/2描述统计hadoop.txt

hadoop fs -rmr $MY_HADOOP/output22
hadoop jar $HADOOP_STREAMING \
    -input $MY_HADOOP/train \
    -output $MY_HADOOP/output22 \
    -mapper "python mapper22.py" \
    -reducer "python reducer22.py" \
    -file $MY_LOCAL/code/mapper22.py $MY_LOCAL/code/reducer22.py
hadoop fs -get $MY_HADOOP/output22/part-00000 $MY_LOCAL/data/2参数估计hadoop.txt
