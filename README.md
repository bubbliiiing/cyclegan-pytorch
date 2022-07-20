## CycleGan：Cycle-Consistent Adversarial Networks模型在pytorch当中的实现
---

### 目录
1. [所需环境 Environment](#所需环境)
3. [文件下载 Download](#文件下载)
4. [预测步骤 How2predict](#预测步骤)
5. [训练步骤 How2train](#训练步骤)
6. [参考资料 Reference](#Reference)

## 所需环境
pytorch==1.2.0    

## 文件下载
本仓库以horse2zebra（马与斑马）数据集为示例，训练了转换的例子，训练好的生成器与判别器模型如下：   
[Generator_A2B_horse2zebra.pth](https://github.com/bubbliiiing/cyclegan-pytorch/releases/download/v1.0/Generator_A2B_horse2zebra.pth)；   
[Generator_B2A_horse2zebra.pth](https://github.com/bubbliiiing/cyclegan-pytorch/releases/download/v1.0/Generator_B2A_horse2zebra.pth)；   
[Discriminator_A_horse2zebra.pth](https://github.com/bubbliiiing/cyclegan-pytorch/releases/download/v1.0/Discriminator_A_horse2zebra.pth)；   
[Discriminator_B_horse2zebra.pth](https://github.com/bubbliiiing/cyclegan-pytorch/releases/download/v1.0/Discriminator_B_horse2zebra.pth)；   
可以通过网盘下载或者通过GITHUB下载。   

权值的网盘地址如下：    
链接: https://pan.baidu.com/s/1mbg-nNX0BuXWff3J4rde3Q 提取码: ykdc    

常用的数据集地址如下：   
链接: https://pan.baidu.com/s/1xng_uQjyG-8CFMktEXRdEg 提取码: grtm     

## 预测步骤
### a、使用预训练权重
1. 下载完库后解压，下载对应权值文件存放到model_data中。
2. 运行predict.py文件。
3. 输入需要预测的图片路径，获得预测结果。
### b、使用自己训练的权重 
1. 按照训练步骤训练。    
2. 在cyclegan.py文件里面，在如下部分修改model_path使其对应训练好的文件；**model_path对应logs文件夹下面的权值文件**。    
```python
_defaults = {
    #-----------------------------------------------#
    #   model_path指向logs文件夹下的权值文件
    #-----------------------------------------------#
    "model_path"        : 'model_data/Generator_A2B_horse2zebra.pth',
    #-----------------------------------------------#
    #   输入图像大小的设置
    #-----------------------------------------------#
    "input_shape"       : [128, 128],
    #-------------------------------#
    #   是否进行不失真的resize
    #-------------------------------#
    "letterbox_image"   : True,
    #-------------------------------#
    #   是否使用Cuda
    #   没有GPU可以设置成False
    #-------------------------------#
    "cuda"              : True,
}
```
3. 运行predict.py文件。
4. 输入需要预测的图片路径，获得预测结果。

## 训练步骤
1. 训练前将期望转换的图片文件放在datasets文件夹下，一共两类，训练目的是让A类与B类互相转换。
2. 运行根目录下面的txt_annotation.py，生成train_lines.txt，保证train_lines.txt内部是有文件路径内容的。  
3. 运行train.py文件进行训练，训练过程中生成的图片可查看results/train_out文件夹下的图片。  
