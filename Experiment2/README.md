# 实验二

## 实验要求
- 基于PyTorch实现AlexNet结构
- 在Caltech101数据集上进行验证
- 使用tensorboard进行训练数据可视化，Loss曲线
- 如有条件，尝试不同参数的影响，尝试其他网络结构
- 请勿使用torchvision.models.AlexNet

## 实验实现
- AlexNet
- VGG16
- ResNet50(torchvision.models)
- InceptionV3(torchvision.models)
- DenseNet121(torchvision.models)

## 目录结构
```
│  README.md
│  
├─Caltech101
│  └─ *
│          
├─Code
│  │  getdata.py
│  │  model.py
│  │  train.py
│  │  trainDenseNet.py
│  │  trainInception.py
│  │  trainResNet.py
│  │  trainVGG.py
│  │  train_eval.py
│  │  
│  └─__pycache__
│          *
│          
├─Logs
│  ├─AlexNet
│  │      *
│  │      
│  ├─DenseNet121
│  │      *
│  │      
│  ├─InceptionV3
│  │      *
│  │      
│  ├─ResNet50
│  │      *
│  │      
│  └─VGG16
│         *
│          
├─Models
│      AlexNet.ckpt
│      DenseNet121.ckpt
│      InceptionV3.ckpt
│      ResNet50.ckpt
│      VGG16.ckpt
│      
└─Report
        AlexNetModel.jpg
        AlexNetModel1.png
        AlexNetTensorboard.png
        DenseNetTensorboard.png
        InceptionTensorboard.png
        report.aux
        report.log
        report.pdf
        report.synctex.gz
        report.tex
        report.toc
        ResNetTensorboard.png
        VGG16Model.jpg
        VGGTensorboard.png
```

## 训练
```shell
# 训练AlexNet
python3 ./Code/train.py
# 训练VGG16
python3 ./Code/trainVGG.py
# 训练ResNet50
python3 ./Code/trainResNet.py
# 训练InceptionV3
python3 ./Code/trainInception.py
# 训练DenseNet121
python3 ./Code/trainDenseNet.py

```

## 实验效果

### AlexNet
![AlexNet](Report/AlexNetTensorboard.png)  

### VGG16
![VGG16](Report/VGGTensorboard.png)  

### ResNet50
![ResNet50](Report/ResNetTensorboard.png)  

### InceptionV3
![InceptionV3](Report/InceptionTensorboard.png)  

### DenseNet121
![DenseNet121](Report/DenseNetTensorboard.png)  




