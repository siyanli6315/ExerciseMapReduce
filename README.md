# 分布式计算

## 1.基于mapreduce的OLS参数估计

我的数据集来自于中经网县域年度库: http://db.cei.gov.cn/page/Default.aspx

### 1.1 数据的描述

我从中经网上收集了1999年到2014年中部五省（河南省、安徽省、湖北省、湖南省和江西省）的所有县区的国内生产总值数据。

### 1.2 描述性统计

数据中包含的字段是中部五省从1999年到2014年来的国民生产总值数据，在这个数据中，我关心的内容是：所有县每年的平均国民生产总值、标准差、最大值、最小值、中位数、四分位数。上述过程都可以通过mapreduce实现。

#### 1.2.1 mapper.py

```python
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
      for i in range(345):
          result[j+1999].append(dat[i][j])

  for i in range(16):
      print('%s\t%s' % (i+1999, result[i+1999]))
```

mapper 函数将输入按照1990到2014年的顺序进行切割，将每一个切片数据按照 <key,value> 的形式传递给 reducer 函数。

#### 1.2.2 reducer.py

```python
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
```

reducer 对传递进来的每一个行数据统计均值、标准差、中位数、最大值、最小值、上下四分位数。然后将结果以字典的形式输出。

#### 1.2.3 main.sh

```
	hadoop fs -rmr $MY_HADOOP/output11
	hadoop jar $HADOOP_STREAMING \
	    -input $MY_HADOOP/gdp.csv \
	    -output $MY_HADOOP/output11 \
	    -mapper "python mapper11.py" \
	    -reducer "python reducer11.py" \
	    -file $MY_LOCAL/code/mapper11.py $MY_LOCAL/code/reducer11.py
	hadoop fs -get $MY_HADOOP/output11/part-00000 $MY_LOCAL/data/1描述统计hadoop.txt
```

main 函数定义了向 hadoop 文件系统中上传数据和代码，运行结果，然后将 hadoop 运算的结果下载下来的过程。

#### 1.2.3 结果

```
  1999	{'1Q': 200056.0, 'min': 1914.7217675976001, 'max': 1035800.0, '3Q': 467354.30499439, 'median': 314081.60320932302, 'sigma': 196871.42057119965, 'mean': 336919.82597281743}
  2000	{'1Q': 174466.64939007501, 'min': 2665.5039088818598, 'max': 982400.0, '3Q': 397113.0, 'median': 269595.0, 'sigma': 182104.99155378749, 'mean': 301849.26243577019}
  2001	{'1Q': 174349.02571065599, 'min': 10326.2884031019, 'max': 1107800.0, '3Q': 435118.0, 'median': 291800.0, 'sigma': 203634.07708821647, 'mean': 324024.09397263388}
```

部分结果如上所示，结果显示，使用Hadoop平台计算出了所有年份的均值、标准差、中位数、最大值、最小值和上下四分位数。

### 1.3 参数估计

我建立的回归方程为，自变量为1999年到2013年的GDP总量，因变量变量2014年的GDP总量。该回归方程一共有15个自变量和1个因变量。具体的计算过程是，首先将数据分成k个不同的块，然后对每一块数据都使用OLS估计回归参数，最后将所有的参数进行平均。这种方法估计出的回归参数和使用全样本估计的回归参数具有一致性。

#### 1.3.1 mapper.py

```python
  #! /usr/bin/env python2.7
  # coding=utf-8

  import sys
  import numpy as np

  dat=list(sys.stdin.readlines())
  n=len(dat)
  dat=[s.strip() for s in dat]
  dat=[s.split(",") for s in dat]
  dat=[[float(a) for a in b[1:]] for b in dat]
  np.random.seed(100)
  np.random.shuffle(dat)
  n=len(dat)
  m=n//5

  def OLS(X,Y,lam = 0.1):
      n=len(X)
      for i in range(n):
          X[i].append(1)
      X=np.array(X)
      Y=np.array(Y)
      p=X.shape[1]
      n=X.shape[0]

      tmp=np.dot(np.transpose(X),X)
      for i in range(p):
          tmp[i][i]+=lam
      beta=np.dot(np.dot(np.linalg.inv(tmp),np.transpose(X)),Y)
      return beta

  for i in range(5):
      X=[s[:15] for s in dat[i*m:(i+1)*m]]
      Y=[s[15] for s in dat[i*m:(i+1)*m]]
      beta=OLS(X,Y,lam=0.1)
      print(list(beta))
```

将读入的数据分块，并对每一块进行OLS参数估计，然后将参数以 <key,value> 的形式输出。

#### 1.3.2 reducer.py

```python
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
```

对传递进来的参数做平均。然后将最终的参数以 <key,value> 的形式输出。

#### 1.3.3 main.sh

```
	hadoop fs -rmr $MY_HADOOP/output12
	hadoop jar $HADOOP_STREAMING \
	    -input $MY_HADOOP/gdp.csv \
	    -output $MY_HADOOP/output12 \
	    -mapper "python mapper12.py" \
	    -reducer "python reducer12.py" \
	    -file $MY_LOCAL/code/mapper12.py $MY_LOCAL/code/reducer12.py
	hadoop fs -get $MY_HADOOP/output12/part-00000 $MY_LOCAL/data/1参数估计hadoop.txt
```

向 hadoop 传递函数和数据，运行回归分析并将输出结果抓回本地的命令如上所示。

#### 1.3.4 结果

```
  {'intercept': 8773.931545354391, 1999: -0.11473278180367327, 2000: 0.11392886873411885, 2001: 0.029933583172624078, 2002: -0.022508264416987656, 2003: -0.0016896370512311076, 2004: 0.014796881447506304, 2005: 0.02028500160172405, 2006: -0.00536205905069328, 2007: 0.004754991404023352, 2008: -0.013335121270181305, 2009: -0.014990601315835827, 2010: 0.006550816845005419, 2011: -0.1959058911996795, 2012: -0.3740375734731825, 2013: 1.5844698887798239}
```

参数结果显示，2011年之后的国民生产总值的参数比较大，说明这些年份的生产总值可以很好的解释2014年的国民生产总值，2011年之前的国民生产总值的参数比较小，说明2011年之前的国民生产总值对2014年的国民生产总值解释性不强。

### 1.4 交叉验证

交叉验证的思想是，将数据分割成k块，使用其中的k-1块作为训练集，拟合模型，然后用第k块评价模型的好坏。从以上的过程可以看出，交叉验证非常适合用mapreduce计算系统计算。

#### 1.4.1 mapper.py

```
  #! /usr/bin/env python2.7
  # coding=utf-8

  import sys
  import numpy as np

  dat=list(sys.stdin.readlines())
  n=len(dat)
  dat=[s.strip() for s in dat]
  dat=[s.split(",") for s in dat]
  dat=[[float(a) for a in b[1:]] for b in dat]
  np.random.seed(100)
  np.random.shuffle(dat)
  n=len(dat)
  m=n//10

  for i in range(10):
      out={"train":{1999:[],2000:[],2001:[],2002:[],2003:[],2004:[],2005:[],2006:[],2007:[],2008:[],2009:[],2010:[],2011:[],2012:[],2013:[],2014:[]},"valid":{1999:[],2000:[],2001:[],2002:[],2003:[],2004:[],2005:[],2006:[],2007:[],2008:[],2009:[],2010:[],2011:[],2012:[],2013:[],2014:[]}}
      dat_valid=[]
      dat_train=[]
      for j in range(i*m,(i+1)*m):
          dat_valid.append(dat[j])
      for j in range(0,i*m):
          dat_train.append(dat[j])
      for j in range((i+1)*m,n):
          dat_train.append(dat[j])
      n_train=len(dat_train)
      n_valid=len(dat_valid)
      for j in range(16):
          for k in range(n_train):
              out["train"][j+1999].append(dat_train[k][j])
      for j in range(16):
          for k in range(n_valid):
              out["valid"][j+1999].append(dat_valid[k][j])
      print(out)
```

在 mapper 函数中，我将原始数据分成了10份，用其中的9份作为训练集，最后一份作为测试集。每一行输出代表着一次 train valid 分割。

#### 1.4.2 reducer.py

```python
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
```

reducer 函数估计了参数并使用参数和 valid 数据计算了R方、模型内预测误差和模型外预测误差。

#### 1.4.3 main.sh

```
	hadoop fs -rmr $MY_HADOOP/output13
	hadoop jar $HADOOP_STREAMING \
	    -input $MY_HADOOP/gdp.csv \
	    -output $MY_HADOOP/output13 \
	    -mapper "python mapper13.py" \
	    -reducer "python reducer13.py" \
	    -file $MY_LOCAL/code/mapper13.py $MY_LOCAL/code/reducer13.py
	hadoop fs -get $MY_HADOOP/output13/part-00000 $MY_LOCAL/data/1交叉验证hadoop.txt
```

#### 1.4.4 结果

部分输出结果如下所示：

```
  k: 10
  参数: [ -5.79213515e-02   4.21734265e-03   3.41991250e-02   4.97352801e-02
    -3.61054772e-02   2.01640084e-02   2.46440058e-03  -2.44458818e-03
    -2.85474250e-02  -4.92792607e-03   6.26018969e-03   5.73374150e-04
    -1.27315568e-01  -1.87136933e-01   1.36834684e+00   5.09063418e+03]
  Rsquare: 0.996569075473
  模型内预测误差: 81732.734033
  模型外预测误差: 202964.136701
  10 折交叉验证误差是：98402.4485774
```

部分结果如上所示。我们的模型R方为99.6\%。模型外预测均方根误差为 202964.14 万元。2014年中部所有县区的平均国民生产总值是 1871907.64 万元。均方根误差约等于均值的十分之一，上述模型的预测准确率较高。

## 2.使用逻辑回归识别手写数字

在本节，我使用的数据是MNIST手写数字识别数据集。数据集网站是：http://yann.lecun.com/exdb/mnist/。原网站中的数据集包含60000个训练样本和10000个测试样本。在本例中，我用到了从原始数据集合中抽取的10000个样本进行实验。数据包含10000行，785列，每一列的第一个数字表示该手写数字的标签（0到9），每一列的后784个数字表示数字的的像素点（$28 \times 28$）。

### 2.1 切割训练集和测试集

```python
  #! /usr/bin/env python2.7
  # coding=utf-8
  '''
  分割训练集和测试集
  输入数据，打乱之后输出测试集
  '''

  import sys
  import numpy as np

  dat=[s.strip() for s in sys.stdin.readlines()]
  np.random.seed(10)
  np.random.shuffle(dat)

  n=len(dat)
  test=dat[n/4*3:]

  for s in test:
      print(s)
```

```python
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
```

我使用了四分之三的数据作为训练集，四分之一的数据作为测试集。main.sh 代码如下所示：

```
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
```

### 2.2 描述统计

#### 2.2.1 mapper.py

```python
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
```

mapper 函数将训练集数据的第一列数据按格式输出。第一列数据表示该行数据表示的图像是哪一个数字。

#### 2.2.2 reducer.py

```python
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
```

reducer 函数定义了一个字典，字典的 key 表示因变量的值，value 表示该数字在训练集中出现的次数。

#### 2.2.3 main.sh

```
	hadoop fs -rmr $MY_HADOOP/output21
	hadoop jar $HADOOP_STREAMING \
	    -input $MY_HADOOP/train \
	    -output $MY_HADOOP/output21 \
	    -mapper "python mapper21.py" \
	    -reducer "python reducer21.py" \
	    -file $MY_LOCAL/code/mapper21.py $MY_LOCAL/code/reducer21.py
	hadoop fs -get $MY_HADOOP/output21/part-00000 $MY_LOCAL/data/2描述统计hadoop.txt
```

#### 2.2.4 结果

```
  {0: 735, 1: 834, 2: 781, 3: 755, 4: 740, 5: 686, 6: 702, 7: 769, 8: 736, 9: 762}
```

结果显示，在训练集中，10个数字都有出现，而且出现的次数大致相同。

### 2.3 参数估计

参数估计的过程和OLS估计的过程类似，首先将数据分成k个不同的块，然后对每一块数据都使用 sklearn 估计回归参数，最后将所有的参数进行平均。

#### 2.3.1 mapper.py

```python
  #! /usr/bin/env python2.7
  # coding=utf-8
  '''
  逻辑回归的参数估计
  '''

  import sys
  import numpy as np
  from sklearn.linear_model import LogisticRegression

  dat=list(sys.stdin.readlines())
  n=len(dat)
  dat=[s.strip() for s in dat]
  dat=[s.split(",") for s in dat]
  dat=[[float(a) for a in b] for b in dat]

  mod=LogisticRegression(penalty='l2',tol=0.0001,C=0.1,fit_intercept=True)

  m=n//10
  for i in range(10):
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
      train=dat[i*m:(i+1)*m]
      x=np.array([b[1:] for b in train])
      y=np.array([b[0] for b in train])
      mod.fit(x,y)
      for i in range(10):
          for j in range(784):
              coef[i][j] += mod.coef_[i][j]
      for i in range(10):
          coef[10][i] += mod.intercept_[i]
      print(coef)
```

#### 2.3.2 reducer.py

```python
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
```

#### 2.3.3 main.sh

```
	hadoop fs -rmr $MY_HADOOP/output22
	hadoop jar $HADOOP_STREAMING \
	    -input $MY_HADOOP/train \
	    -output $MY_HADOOP/output22 \
	    -mapper "python mapper22.py" \
	    -reducer "python reducer22.py" \
	    -file $MY_LOCAL/code/mapper22.py $MY_LOCAL/code/reducer22.py
	hadoop fs -get $MY_HADOOP/output22/part-00000 $MY_LOCAL/data/2参数估计hadoop.txt
```

#### 2.3.4 结果

```
  {0: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -7.545928952223733e-06, -2.1170522893738815e-05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
```

部分结果如上所示，我们的模型一共包含7850($784\times 10+10$)个参数（预测每一个数字需要784个w和1个b）。所以我们的输出的结果是 如上的 <key,value> 形式，11 个 <key> 分别对应了10个数字和1个截距。value 对应所有的参数值。

#### 2.3.5 预测效果检验

```python
   #! /usr/bin/env python2.7
   # coding=utf-8
   '''
   逻辑回归的结果检验
   '''

   from __future__ import division
   import numpy as np
   import sys

   test_dat=list(open("./逻辑回归/中间数据/test").readlines())
   test_dat=[s.strip() for s in test_dat]
   test_dat=[s.split(",") for s in test_dat]
   test_dat=[[float(x) for x in s] for s in test_dat]
   test_x=np.array([s[1:] for s in test_dat])
   test_y=np.array([s[0] for s in test_dat])

   for coef in sys.stdin:
       coef=coef.strip()
       coef=eval(coef)
   w=[]
   for i in range(10):
       w.append(coef[i])
   w=np.array(w)
   b=np.array(coef[10])

   def logit(test_x,w,b):
       z=np.dot(test_x,np.transpose(w))+np.transpose(b)
       y_hat= 1.0/(1.0+np.exp(-z))
       y_hat2=[]
       for i in y_hat:
           y_hat2.append(np.argmax(i))
       return y_hat2

   def acc(y_hat,y):
       accuracy=sum([int(y_hat[i]==y[i]) for i in range(len(y))])/len(y)
       return accuracy

   y_hat=logit(test_x,w,b)
   acc=acc(y_hat,test_y)

   print("Accuracy: {0}".format(acc))
```

使用测试集预测的代码如上所示，由于该代码需要两个输入，所以目前该代码只能在本地运行，不能使用 Hadoop.

```
lifeng@master3:~/exam/cufe_lisiyan/data$ cat 2参数估计.txt | ./test2.py
Accuracy: 0.8368
lifeng@master3:~/exam/cufe_lisiyan/data$ cat 2参数估计hadoop.txt | ./test2.py
Accuracy: 0.8584
```

如上所示，在本机上预测准确率是83.68\%。在hadoop集群上运行的预测准确率是85.8\%。在kaggle数据网上，使用logistic预测mnist的最高预测准确率是89\%左右。
