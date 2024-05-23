---
date: 2020-03-21 19:24:49
layout: post
title: page rank & text rank
description: page rank & text rank
image: /post_images/ml/page rank & text rank封面.png
optimized_image: /post_images/ml/page rank & text rank封面.png
category: 机器学习
tags:
  - 机器学习
  - machine learning
  - page rank
  - text rank
author: 沙中世界
---

## page rank
主要用于计算网页权重
### 基本思想
将所有的网站看成一张有向图（网站与网站之间存在链接关系）；
每一个网站的权重计算方式如下：<br>
![PageRank 权重计算](/my_docs/ml/images/24-1.jpg)<br>
计算举例：<br>
![PageRank 权重计算举例](/my_docs/ml/images/24-2.jpg)

## text rank
常常用于关键词提取与摘要生成
