# 车道线检测实现示例

参考论文：Towards End-to-End Lane Detection: an Instance Segmentation Approach   
参考代码：https://github.com/klintan/pytorch-lanenet


## 环境安装

### 安装pytorch（推荐使用anaconda安装）

1. anaconda创建名称为pytorch1.7.0的python3.7环境（环境名称可修改）   
`conda create -n pytorch1.7.0 python=3.7`
2. 激活虚拟环境，环境名称和上面一致   
`conda activate pytorch1.7.0`
3. 安装pytorch（根据服务器配置选择合适的pytorch版本，建议使用官网https://pytorch.org/ 命令安装）  
`conda install pytorch==1.7.0 torchvision==0.8.0 torchaudio==0.7.0 cudatoolkit=11.0 -c pytorch`


### 安装库文件
安装库，可在requirements.txt中添加或修改需要的库文件  
`pip install -r requirements.txt`



## 车道线检测实现方法


### 使用示例图片进行模型训练

主文件train.py位于lanenet文件夹下，使用本仓库的示例图片进行训练可用如下命令：   
`python3 lanenet/train.py --dataset ./data/training_data_example`   
注意，示例训练图片样本较少，实际训练时需使用公开数据集或者自己添加更多数据进行训练。   
代码中重要模块介绍如下，其他信息可见代码注释：


#### 1. 设置参数

可指定的参数见utils下cli_helper.py文件 ，训练时可在命令行中指定这些参数

#### 2. 读取数据


①指定数据集路径  
`train_dataset_file = os.path.join(args.dataset, 'train.txt')`   
② 读取数据  
`train_dataset = LaneDataSet(train_dataset_file, transform=transforms.Compose([Rescale((512, 256))]))`  
`train_loader = DataLoader(train_dataset, batch_size=args.bs, shuffle=True)`  


在dataloader文件夹下data_loaders.py中定义自己的数据集。  
示例代码定义LaneDataSet类，继承 Dataset 方法，并重写__getitem__()和__len__()方法。


示例训练数据包括三部分：原始图片数据，二值化标签，实例标签，位于data.training_data_example中。  
若使用公开数据集或自己采集的数据，可以运行scripts文件夹下tusimple_transform.py程序获得这三种类型数据。  
若使用其他方法进行车道线检测，可能需要重新编写tusimple_transform.py文件以获得其他类型的数据标签。


#### 3. 加载模型和损失函数


`model = LaneNet()`  
在model文件夹下model.py中搭建自己的车道线检测模型并定义损失函数。示例代码定义一个LaneNet类，继承 nn.Module，并重写初始化的__init__函数和forward函数  
LaneNet将车道线检测视为实例分割问题，每个车道线形成独立的实例，但都属于车道线这个类别。网络框架包含分割和聚类两个部分，详细结构可见论文和代码。


示例代码中损失函数定义为compute_loss，需要根据自己的网络框架设计对应的损失函数。

#### 4. 指定优化器和学习策略


这里使用adam优化器  
`optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)`

#### 5. 模型训练

①前向传播    
`net_output = model(image_data)`  
②计算loss   
`total_loss, binary_loss, instance_loss, out, train_iou = compute_loss(net_output, binary_label, instance_label)`  
③梯度清零   
`optimizer.zero_grad()`  
④反向传播   
`total_loss.backward()`  
⑤更新参数   
`optimizer.step()`

#### 6. 测试指标


示例方法使用的评价指标为准确率acc。 除了准确率，还可计算FP,FN分数。   
在其他方法中评价指标还包括fps、F1分数等（代码需要自己实现）。

   
①准确率Accuracy  
准确率Accuracy定义为每张图像的平均正确点数。当基准点和预测值之间的差值小于某个阈值时，则这个点是正确的。   
$$Accuracy =\sum_{im} \frac{c_{im}}{s_{im}}$$    
$c_{im}$是正确预测的车道点数量，$s_{im}$是真实标签中的车道点数量。
  
②误检率FP 漏检率FN  F1分数   
$$FP=\frac{F_{pred}}{N_{pred}}$$   
$$FN=\frac{M_{pred}}{N_{gt}}$$  
$F_{pred}$为预测错误的车道线数量, $N_{pred}$为预测的车道线总数量, $M_{pred}$为没有预测到但真实存在的车道线数量,$N_{gt}$为标签中所有的车道线数量。   


F1分数    
$$\text{precision}=\frac{TP}{TP+FP}$$   
$$\text{recall}=\frac{TP}{TP+FN}$$  
$$F_1=2\cdot \frac{\text{precision} \cdot \text{recall}}{\text{precision}+\text{recall}}$$

F1分数同时兼顾了分类模型的精确率和召回率，可以看作是模型精确率和召回率的一种调和平均，它的最大值是1，最小值是0。



③FPS每秒传输帧数  
效率指标，计算每秒能检测的图片数量，衡量检测速度。

### 使用TuSimple数据集进行模型训练


### 1.下载TuSimple数据集 
https://github.com/TuSimple/tusimple-benchmark/issues/3


### 2.运行`scripts`文件夹下tusimple_transform.py文件  
`python tusimple_transform.py --src_dir <directory of downloaded tusimple>`


### 3.使用如下命令训练模型   
`python3 lanenet/train.py --dataset <tusimple_transform script output folder>`



## 资源

### 相关仓库

https://github.com/MaybeShewill-CV/lanenet-lane-detection


### 参考文献

Towards End-to-End Lane Detection: an Instance Segmentation Approach  
https://arxiv.org/pdf/1802.05591.pdf


ESPNet: Efficient Spatial Pyramid of Dilated Convolutions for Semantic Segmentation  
https://arxiv.org/abs/1803.06815


ENet: A Deep Neural Network Architecture for Real-Time Semantic Segmentation  
https://arxiv.org/abs/1606.02147  
https://maybeshewill-cv.github.io/lanenet-lane-detection/
