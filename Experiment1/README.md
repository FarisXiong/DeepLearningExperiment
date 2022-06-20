# 实验一

## 实验要求
使用PyTorch实现MLP，并在MNIST数据集上验证。  
- 环境配置
- 代码编写
- 实验验证，在MNIST数据集上进行实验验证

## 目录结构
```
│  README.md
│
├─Code
│  │  confusion.png
│  │  dataprocessed.py
│  │  model.py
│  │  train.py
│  │  train_eval.py
│  │
│  ├─dataset
│  │  └─MNIST
│  │      └─raw
│  │              t10k-images-idx3-ubyte
│  │              t10k-images-idx3-ubyte.gz
│  │              t10k-labels-idx1-ubyte
│  │              t10k-labels-idx1-ubyte.gz
│  │              train-images-idx3-ubyte
│  │              train-images-idx3-ubyte.gz
│  │              train-labels-idx1-ubyte
│  │              train-labels-idx1-ubyte.gz
│  │
│  └─__pycache__
│          *
│
├─Logs
│  ├─04-28_20.17
│  │      *
│  │
│  └─04-28_21.22
│         *
│
├─Models
│      CNN.ckpt
│      MLP.ckpt
│
└─Report
        CnnAcc.png
        CnnLoss.png
        confusionCNN.png
        confusionMLP.png
        MlpAcc.png
        MlpLoss.png
        report.aux
        report.log
        report.pdf
        report.synctex.gz
        report.tex
        report.toc
```


## 训练
```shell
# 预处理数据
python3 ./Code/dataprocessed.py
# 训练模型
python3 ./Code/train.py
```


## 实验效果
### MLP
ACC:  
![MLP_ACC](Report/MlpAcc.png)  

Loss:  
![MLP_LOSS](Report/MlpLoss.png)  

confusion matrix:
![MLP_CONFUSION_MATRIX](Report/confusionMLP.png)  

### CNN
ACC:  
![CNN_ACC](Report/CnnAcc.png)  

Loss:  
![CNN_LOSS](Report/CnnLoss.png)  

confusion matrix:
![CNN_CONFUSION_MATRIX](Report/confusionCNN.png)  




