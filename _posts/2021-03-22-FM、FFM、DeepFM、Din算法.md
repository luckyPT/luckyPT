---
date: 2021-03-22 13:24:49
layout: post
title: FM、FFM、DeepFM、Din算法
description: FM、FFM、DeepFM、Din算法
image: /post_images/ml/FM-FFM-Din封面.png
optimized_image: /post_images/ml/FM-FFM-Din封面.png
category: 机器学习
tags:
  - 机器学习
  - machine learning
  - FM
  - FFM
  - DeepFM
  - Din
author: 沙中世界
---

## FM算法
### 特点
- 稀疏特征中表现比较好
- 计算效率高，线性时间复杂度
- 可直接利用梯度下降的方式进行参数训练（不需要像SVM那种，转为对偶问题）；

![FM优化逻辑](/my_docs/ml/images/28-1.png)

## FFM算法

## DeepFM算法
通常FM算法只用于二阶特征组合（理论上可以进行高阶特征组合，但是复杂度极高）；<br>
对于高阶特征组合之间的关系，很自然想到用DNN去拟合

## Din算法

## AFM算法

## 双塔模型
