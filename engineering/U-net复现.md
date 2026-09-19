# U-net复现实验报告

### 一、实验概述

​	本实验基于U-Net模型，使用FIVES眼底图像血管分割数据集，在AutoDL云GPU环境中完成训练流程复现。

### 二、数据集

数据集链接：https://figshare.com/articles/figure/FIVES_A_Fundus_Image_Dataset_for_AI-based_Vessel_Segmentation/19688169

​	FIVES 数据集包含 800 张带标注的眼底图像，其中训练部分 600 张，测试部分 200 张。原始图像尺寸为 2048×2048，训练时使用 0.25 的缩放比例，输入尺寸为 512×512。



### 三、训练过程

实验原计划训练 20 轮，并在每轮结束后计算验证集 Dice。![训练结果](screenshots/train.png)



四、训练与测试结果

本次实验完成了数据读取、预处理、模型训练、验证集模型选择、独立测试评估和结果导出的完整流程。![训练曲线](screenshots/curves.png)



#### 分割示例

从左到右：原图、真实标注、预测![临时结果](screenshots/temp.png)

