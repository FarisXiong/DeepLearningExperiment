# 实验三

## 实验要求

### 基础部分
基于PyTorch实现VGG/ResNet/SENet等结构:  
- 自己实现VGG(11)：要求Conv部分参照论文，可动态调整结构；
- 自己实现ResNet(18)：要求基于残差块，参照论文，可动态调整；
- 在ResNet基础上，添加SE block，对比其性能表现。
- 要求基于CUDA实现，可设定是否使用GPU，通过argparse包实现，默认参数设定为GPU
### 性能调优  
- 公共部分（选用一个上述最优的模型）
> 1.进行优化器(SGD与Adam)对比；
> 2.进行data augmentation（翻转、旋转、移位等操作）对比。
- 自选部分
> 引入新的模块/换用更好的模型，内容不限，可有效提升性能即可


## 实现
### 基础部分

- 实现VGG16
- 实现ResNet18、34、50、101、152
- 实现SE-ResNet18、34、50、101、152

### 性能调优部分
使用Swin Transformer模型，基于预训练模型训练。

## 数据来源
[Plant Seedlings Classification](https://www.kaggle.com/competitions/plant-seedlings-classification)

## 目录结构
```
│  README.md
│  
├─Code
│  │  argparser.py
│  │  config.py
│  │  dataset.py
│  │  main.py
│  │  models.py
│  │  trainer.py
│  │  
│  └─__pycache__
│          *
│          
├─Dataset
│  ├─test
│  │      *
│  │      
│  └─train
│      ├─ *
│              
├─out
│      ResNetAdamNoAugsubmission.csv
│      ResNet_all_1_submission.csv
│      ResNet_submission.csv
│      SENet_all_1_submission.csv
│      SENet_none_submission.csv
│      SENet_rot_flp_pst_slr_submission.csv
│      SENet_submission.csv
│      SwinTransformer_none_submission.csv
│      SwinTransformer_submission.csv
│      VGG_all_1_submission.csv
│      VGG_none_1_submission.csv
│      VGG_submission.csv
│      
├─Report
│  │  refs.bib
│  │  Report.aux
│  │  Report.bbl
│  │  Report.blg
│  │  Report.brf
│  │  Report.log
│  │  Report.out
│  │  Report.pdf
│  │  Report.synctex.gz
│  │  Report.tex
│  │  Report.toc
│  │  
│  └─figures
│          ResNetBlock.png
│          resnet_aug_macrof1score_dev.png
│          resnet_aug_macrof1score_dev.svg
│          resnet_aug_microf1score_dev.png
│          resnet_aug_microf1score_dev.svg
│          resnet_noaug_macrof1score_dev.png
│          resnet_noaug_macrof1score_dev.svg
│          resnet_noaug_microf1score_dev.png
│          resnet_noaug_microf1score_dev.svg
│          senet.png
│          senetwithresnet.png
│          senet_aug_macrof1score_dev.png
│          senet_aug_macrof1score_dev.svg
│          senet_aug_microf1score_dev.png
│          senet_aug_microf1score_dev.svg
│          senet_noaug_macrof1score_dev.png
│          senet_noaug_macrof1score_dev.svg
│          senet_noaug_microf1score_dev.png
│          senet_noaug_microf1score_dev.svg
│          swintrans_noaug_macrof1score_dev.png
│          swintrans_noaug_macrof1score_dev.svg
│          swintrans_noaug_microf1score_dev.png
│          swintrans_noaug_microf1score_dev.svg
│          VGG16Model.jpg
│          vgg_aug_macrof1score_dev.png
│          vgg_aug_macrof1score_dev.svg
│          vgg_aug_microf1score_dev.png
│          vgg_aug_microf1score_dev.svg
│          vgg_noaug_macrof1score_dev.png
│          vgg_noaug_macrof1score_dev.svg
│          vgg_noaug_microf1score_dev.png
│          vgg_noaug_microf1score_dev.svg
│          
└─Trains
    ├─logs
    │  └─*
    │          
    └─models
            ResNet_all_1_1.pkl
            ResNet_none_1_1.pkl
            SENet_all_1_1.pkl
            SENet_none_1_1.pkl
            SwinTransformer.pkl
            SwinTransformer_none_1_5.pkl
            SwinTransformer_none_2_5.pkl
            SwinTransformer_none_3_5.pkl
            SwinTransformer_none_4_5.pkl
            SwinTransformer_none_5_5.pkl
            VGG_all_1_1.pkl
            VGG_none_1_1.pkl
```

## 训练
### 训练参数
|params|options|
|:--:|:--:|
|aug|all/none|
|model|VGG/ResNet/SENet/SwinTransformer|
|mode|train/test/both|
|opt|SGD/Adam|
|dev|cpu/cuda|
|lr|learning_rate:float|
|bs|batch_size:int|
|epoch|epoches:int|
|kf|KFold:int|


### 相关命令
```shell
python3 ./Code/main.py --mode train --model VGG --opt Adam --aug rot --epoch 100 --bs 128 --lr 1e-5 --dev cuda 
```

## 测试
测试阶段会使用训练阶段训练好的模型测试。  
```shell
python3 ./Code/main.py --mode test --model VGG
```


## 结果

|模型|数据增强|优化器| MicroF1Score | MacroF1Score |KaggleScore|  
|:---:|:------:|:--:|:------------:|:------------:|:---:|
|VGG16|无|Adam|   0.8505    |   0.8276    |0.85390|
|VGG16|数据增强|Adam|0.8421|0.8208|0.8576|
|ResNet18|无|Adam|   0.9284    |   0.9175    |0.92191|
|ResNet18|数据增强|Adam|0.8244|0.8244|0.85894|
|SENet18|无|Adam|   0.9221    |   0.9095    |0.91939|
|SENet18|数据增强|Adam|   0.7273    |   0.6161    |0.84382|
|SwinTransformer|预训练的feature_extractor+Kfold|Adam|0.9728|0.9706|   0.97481    |

