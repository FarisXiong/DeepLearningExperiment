# 实验四

## 实验内容
利用Pytorch自己实现RNN、GRU、LSTM和Bi-LSTM不可直接调用nn.RNN(), nn.GRU(), nn.LSTM()。  
利用上述四种结构进行文本多分类，计算测试结果的准确率、召回率和F1值，对比分析四种结构的实验结果。  
任选上述一种结构进行温度预测，使用五天的温度值预测出未来两天的温度值；给出与真实值的平均误差和中位误差。



## 数据来源
将以下数据下载后放入Dataset文件夹下对应名字目录下。  
[online_shopping_10_cats](https://github.com/SophonPlus/ChineseNlpCorpus/blob/master/datasets/online_shopping_10_cats/intro.ipynb)  
[jena_climate_2009_2016](https://www.kaggle.com/datasets/stytch16/jena-climate-2009-2016)  

## 目录结构
```
│  README.md
│  
├─Code
│  │  argparser.py
│  │  config.py
│  │  datapreprocess.py
│  │  dataset.py
│  │  main.py
│  │  model.py
│  │  pretrained.py
│  │  trainer.py
│  │  utils.py
│  │  
│  └─__pycache__
│          *
│          
├─Dataset
│  ├─jena_climate_2009_2016
│  │      jena_climate_2009_2016.csv
│  │      test.csv
│  │      train.csv
│  │      
│  └─online_shopping_10_cats
│          online_shopping_10_cats.csv
│          
├─Models
│  ├─Images
│  │      confusion_Text_BiLSTM.png
│  │      confusion_Text_GRU.png
│  │      confusion_Text_LSTM.png
│  │      confusion_Text_RNN.png
│  │      
│  ├─Logs
│  │  ├─Text_BiLSTM
│  │  │      *
│  │  │      
│  │  ├─Text_GRU
│  │  │      *
│  │  │      
│  │  ├─Text_LSTM
│  │  │      *
│  │  │      
│  │  ├─Text_RNN
│  │  │      *
│  │  │      
│  │  └─Weather_LSTM
│  │          *
│  │          
│  ├─Model
│  │      Text_BiLSTM.pkl
│  │      Text_GRU.pkl
│  │      Text_LSTM.pkl
│  │      Text_RNN.pkl
│  │      Weather_LSTM.pkl
│  │      
│  ├─pretrained
│  │      FastText.bin
│  │      FastText.bin.syn1neg.npy
│  │      FastText.bin.wv.vectors_ngrams.npy
│  │      FastText.bin.wv.vectors_vocab.npy
│  │      
│  └─TemperatureOutput
│          01.03.2016 00_10_00.png
│          01.09.2015 00_10_00.png
│          01.12.2015 00_10_00.png
│          02.02.2016 00_10_00.png
│          02.06.2015 00_10_00.png
│          02.08.2016 00_10_00.png
│          03.02.2015 00_10_00.png
│          03.03.2015 00_10_00.png
│          03.05.2016 00_10_00.png
│          03.11.2015 00_10_00.png
│          04.08.2015 00_10_00.png
│          04.10.2016 00_10_00.png
│          05.01.2016 00_10_00.png
│          05.04.2016 00_10_00.png
│          05.05.2015 00_10_00.png
│          05.07.2016 00_10_00.png
│          06.01.2015 00_10_00.png
│          06.09.2016 00_10_00.png
│          06.10.2015 00_10_00.png
│          06.12.2016 00_10_00.png
│          07.04.2015 00_10_00.png
│          07.06.2016 00_10_00.png
│          07.07.2015 00_10_00.png
│          08.03.2016 00_10_00.png
│          08.09.2015 00_10_00.png
│          08.11.2016 00_10_00.png
│          08.12.2015 00_10_00.png
│          09.02.2016 00_10_00.png
│          09.06.2015 00_10_00.png
│          09.08.2016 00_10_00.png
│          10.02.2015 00_10_00.png
│          10.03.2015 00_10_00.png
│          10.05.2016 00_10_00.png
│          10.11.2015 00_10_00.png
│          11.08.2015 00_10_00.png
│          11.10.2016 00_10_00.png
│          12.01.2016 00_10_00.png
│          12.04.2016 00_10_00.png
│          12.05.2015 00_10_00.png
│          12.07.2016 00_10_00.png
│          13.01.2015 00_10_00.png
│          13.09.2016 00_10_00.png
│          13.10.2015 00_10_00.png
│          13.12.2016 00_10_00.png
│          14.04.2015 00_10_00.png
│          14.06.2016 00_10_00.png
│          14.07.2015 00_10_00.png
│          15.03.2016 00_10_00.png
│          15.09.2015 00_10_00.png
│          15.11.2016 00_10_00.png
│          15.12.2015 00_10_00.png
│          16.02.2016 00_10_00.png
│          16.06.2015 00_10_00.png
│          16.08.2016 00_10_00.png
│          17.02.2015 00_10_00.png
│          17.03.2015 00_10_00.png
│          17.05.2016 00_10_00.png
│          17.11.2015 00_10_00.png
│          18.08.2015 00_10_00.png
│          18.10.2016 00_10_00.png
│          19.01.2016 00_10_00.png
│          19.04.2016 00_10_00.png
│          19.05.2015 00_10_00.png
│          19.07.2016 00_10_00.png
│          20.01.2015 00_10_00.png
│          20.09.2016 00_10_00.png
│          20.10.2015 00_10_00.png
│          20.12.2016 00_10_00.png
│          21.04.2015 00_10_00.png
│          21.06.2016 00_10_00.png
│          21.07.2015 00_10_00.png
│          22.03.2016 00_10_00.png
│          22.09.2015 00_10_00.png
│          22.11.2016 00_10_00.png
│          22.12.2015 00_10_00.png
│          23.02.2016 00_10_00.png
│          23.06.2015 00_10_00.png
│          23.08.2016 00_10_00.png
│          24.02.2015 00_10_00.png
│          24.03.2015 00_10_00.png
│          24.05.2016 00_10_00.png
│          24.11.2015 00_10_00.png
│          25.08.2015 00_10_00.png
│          26.01.2016 00_10_00.png
│          26.04.2016 00_10_00.png
│          26.05.2015 00_10_00.png
│          26.07.2016 00_10_00.png
│          27.01.2015 00_10_00.png
│          27.09.2016 00_10_00.png
│          27.10.2015 00_10_00.png
│          27.12.2016 00_10_00.png
│          28.04.2015 00_10_00.png
│          28.06.2016 00_10_00.png
│          28.07.2015 00_10_00.png
│          29.03.2016 00_10_00.png
│          29.09.2015 00_10_00.png
│          29.11.2016 00_10_00.png
│          29.12.2015 00_10_00.png
│          30.06.2015 00_10_00.png
│          30.08.2016 00_10_00.png
│          31.03.2015 00_10_00.png
│          31.05.2016 00_10_00.png
│          error.png
│          output.csv
│          
└─Report
        BiLSTMTrainF1.png
        BiLSTMTrainloss.png
        confusion_Text_BiLSTM.png
        confusion_Text_GRU.png
        confusion_Text_LSTM.png
        confusion_Text_RNN.png
        error.png
        GRU.png
        GRUTrainF1.png
        GRUTrainloss.png
        LSTM.png
        LSTMTrainF1.png
        LSTMTrainloss.png
        refs.bib
        Report.aux
        Report.bbl
        Report.blg
        Report.log
        Report.pdf
        Report.synctex.gz
        Report.tex
        Report.toc
        RNN.png
        RNNtrainF1.png
        RNNtrainloss.png
        weather1.png
        weather2.png
        weather3.png
        weather4.png
        weather5.png
        weather6.png
        weather7.png
        weather8.png
        weather9.png
```


## 训练
### 参数
|params|options|
|:--:|:--:|
|data|Text/Weather|
|model|RNN/GRU/LSTM/BiLSTM|
|opt|SGD/Adam|
|lr|learning_rate:float|
|bs|batch_size:int|
|epoch|epoches:int|
### 文本分类训练
训练FastText词向量模型。  
```shell
python3 ./Code/pretrained.py
```  
数据预处理后即可训练。例如:  
```shell
python3 ./Code/main.py --data Text --model RNN --bs 512 
```

### 天气预测训练
```shell
python3 ./Code/main.py --data Weather --model LSTM
```



## 文本分类训练效果
在该数据集上训练各种网络的效果
### RNN
测试集上各个类别的结果  

![RNN Confusion Matrix](./Models/Images/confusion_Text_RNN.png)    

| class | precision | recall | f1-score | support |
|--:|:--:|:--:|:--:|:--:|
| Book |0.69|0.76|0.72|770|
| Pad |0.60|0.67|0.63|2000|
| Phone |0.00|0.00|0.00|464|
| Fruit |0.55|0.61|0.58|2000|
| Shampoo |0.49|0.50|0.49|2000|
| ElectricWaterHeaters |0.00|0.00|0.00|115|
| Monmilk |0.52|0.60|0.56|407|
| Clothe |0.75|0.63|0.69|2000|
| Computer |0.58|0.76|0.66|798|
| Hotel |0.92|0.94|0.93|2000|
| <b>Accuracy</b> | ||0.65|12554|
| <b>macro avg</b> |0.51|0.55|0.53|12554|
| <b>weighted avg</b> |0.62|0.65|0.63|12554|

### GRU
测试集上各个类别的结果，对应的confusion matrix如下图所示。  

![GRU Confusion Matrix](./Models/Images/confusion_Text_GRU.png)    

| class | precision | recall | f1-score | support |
|--:|:--:|:--:|:--:|:--:|
| Book | 0.79 | 0.95 | 0.86 | 770 |
| Pad | 0.53 | 0.82 | 0.64 | 2000 |
| Phone |0.00|0.00|0.00|464|
| Fruit |0.85|0.86|0.85|2000|
| Shampoo |0.75|0.84|0.79|2000|
| ElectricWaterHeaters |0.00|0.00|0.00|115|
| Monmilk |0.00|0.00|0.00|407|
| Clothe |0.81|0.89|0.85|2000|
| Computer |0.00|0.00|0.00|798|
| Hotel |0.93|0.97|0.95|2000|
| <b>Accuracy</b> | ||0.76|12554|
| <b>macro avg</b> |0.47 |0.53|0.49|12554|
| <b>weighted avg</b> | 0.67|0.76|0.70|12554|


### LSTM
测试集上各个类别的结果，对应的confusion matrix如下图所示。  

![LSTM Confusion Matrix](./Models/Images/confusion_Text_LSTM.png)    

| class | precision | recall | f1-score | support |
|--:|:--:|:--:|:--:|:--:|
| Book |0.59 |0.93|0.72|770|
| Pad |0.56|0.75|0.64|2000|
| Phone |0.00|0.00|0.00|464|
| Fruit |0.91|0.84|0.88|2000|
| Shampoo |0.65|0.86|0.74|2000|
| ElectricWaterHeaters |0.00|0.00|0.00|115|
| Monmilk |0.00|0.00|0.00|407|
| Clothe |0.86|0.86|0.86|2000|
| Computer |0.00|0.00|0.00|798|
| Hotel |0.90|0.97|0.93|2000|
| <b>Accuracy</b> | ||0.74|12554|
| <b>macro avg</b> | 0.45|0.52|0.48|12554|
| <b>weighted avg</b> |0.65 |0.74|0.69|12554|


### BiLSTM
测试集上各个类别的结果，对应的confusion matrix如下图所示。  

![BiLSTM Confusion Matrix](./Models/Images/confusion_Text_BiLSTM.png)    

| class | precision | recall | f1-score | support |
|--:|:--:|:--:|:--:|:--:|
| Book | 0.94|0.90|0.92|770|
| Pad |0.78|0.79|0.78|2000|
| Phone |0.76|0.82|0.79|464|
| Fruit |0.90|0.90|0.90|2000|
| Shampoo |0.80|0.83|0.81|2000|
| ElectricWaterHeaters |0.00|0.00|0.00|115|
| Monmilk |0.96|0.95|0.96|407|
| Clothe |0.87|0.88|0.88|2000|
| Computer |0.84|0.85|0.84|798|
| Hotel |0.97|0.96|0.97|2000|
| <b>Accuracy</b> | ||0.87|12554|
| <b>macro avg</b> | 0.78|0.79|0.79|12554|
| <b>weighted avg</b> | 0.86|0.87|0.86|12554|

