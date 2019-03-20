A Bi-directional Message Passing Model for Salient Object Detection

标签（空格分隔）： 论文笔记

---

笔记
TITLE: A Bi-directional Message Passing Model for Salient Object Detection 双向消息传递模型
AUTHER: Lu Zhang, Ju Dai, Huchuan Lu, You He, Gang Wang
FROM: CVPR2018

## 1. CONTRIBUTIONS

作者提出一种网络结构上的改变能够提高Salient Detection的MAE值

提出了两个模块的组合：

* MCFEM : multi-scale context-aware feature extraction 
* GBMPM : gated bi-directional meassage passing 

结构清晰：
1. five level each side
2. capture side information by feature extraction(MCFEM)
3. integrate each side's feature information by low2high and high2low(GBMPM)
4. fusion high2low
    * why from high2low instead of low2high？ Answer： coarse to fine means resolution 
    
实验部分：

* 评价标准: F-measure MAE 
* 数据集：ECSSD PASCAL-S SOD HKU-IS DUTS-test
* 未进行详细的实验分析和叙述，对比论文 Deeply Supervised Salient Object Detection with Short Connections。

## 2. SOME DETAILS

![Bi-directional Message Passing Model][1]

### 2.1. 结构

修改了原VGG-16:
1. 去掉全连接层: 每个level得到的是pixel-wise的信息，不需要从最后的全连接处得到最终的salient map
2. 去掉pooling层: retain detials of last convolutional layer 保留最后一层的细节

### 2.2. MCFEM

* 功能：capture 各个level 的Context
* 3.2 节
* input : Image[WxH] 5 level 's feature map $f_i$
* output : $f_i => f_i^c$ 
* process : 
    1. four dilated convolutional layer(same kernal size 3*3 with different dilation rate 1,3,5,7) , 
    2. combine 4 layers' feature map by cross-channel concatenation

使用扩张卷积的好处：reduce redundant computation

### 2.3. GBMPM

* 3.3 节
* 功能 : integrate high-lelel sematic concept and low-level spatial detail
* input :  $f_i^c$
* output : $h_i^0(f_i^c)$ =>  $h_i^1$ & $h_i^2$ => $h_i^3$
* process : 
    1. $h_i^0 = f_i^c$
    2. $h_i^1$ : 从low到high level之间是通过卷积+Gate+downsample 操作
    3. $h_i^2$ : 从high到low level之间是通过卷积+Gate+upsample 操作
    4. $h_i^3$ : 合并 $h_i^1$ 和 $h_i^2$ 是通过卷积+对应channel的concatenation操作

* Gate : sigmod function to 3*3 convolutional layer 
    * 功能: adaptively pass , 决定当前的level的feature是否对下一个level的feature有用
    *  Gate的范围: passing rate : [0, 1], [没用一点不传递到下层，完全传递]


### 2.4. Fusion

也要经过一个1*1 converlutional layer 来确定融合过程中的参数

### 2.5. Loss
用最终的salient detection map 和 ground truth 使用corss-entropy loss

## 3. ADVANTAGES
## 4. DISADVANTAGES

- 从Limitation或者Future Work中找

## 5. OTHER

- [相关网站][2] 
- 笔记 
- [代码][3] 
- [文章][4]
- 笔记时间
    * 2018年5月3日20:21:42


  [1]: http://p6uxtqgn2.bkt.clouddn.com/Bi-directional%20Message%20Passing%20Model.png
  [2]: http://ice.dlut.edu.cn/lu/publications.html
  [3]: https://github.com/luzhang111/A-Bi-directional-message-passing-model-for-salient-object-detection
  [4]: https://drive.google.com/open?id=1sD2hXJpnOJqCOn2MA1L7kVwop7tSdhMD