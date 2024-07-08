# Lecture 9: CNN Architecture

## ILSVRC

![linear](./images/Lec09/1%20(12).png){: width="600px" .center}

> We focus on AlexNet, VGGNet, GooleNet and ResNet.

## AlexNet

![linear](./images/Lec09/1%20(1).png){: width="600px" .center}

+ AlexNet看似将整个网络分成了上下两部分, 这是因为在2012年左右, 作者使用的GPU的内存不能存下整个网络的参数, 所以需要分成两部分. 我们可以看到CONV1之后的depth应当是96(96个filter), 所以两部分分别存了64层. 
  <br>Historical note: Trained on GTX 580 GPU with only 3 GB of memory. Network spread across 2 GPUs, half the neurons (feature maps) on each GPU.
  
+ CONV1, CONV2, CONV4, CONV5: Connections only with feature maps on same GPU.
+ CONV3, FC6, FC7, FC8: Connections with all feature maps in preceding layer, communication across GPUs.

## VGGNet

**Small filters, deeper networks.**

![linear](./images/Lec09/1%20(2).png){: width="600px" .center}

!!!question "Why use smaller fliters?(3*3 conv)"
    ![linear](./images/Lec09/1%20(1).jpg){: width="400px" .center}
    Stack of three 3\*3 conv(stride 1) layers has the **same effective receptive field** as one 7\*7 conv layer. But **deeper**, and **more non-linearities**. 三层3\*3的卷积核实际上形成了一个金字塔形的结构, 能够采集到原始图像7\*7的信息
    
    Also, **fewer parameters**: $3*(3^2 C^2)$ vs. $7^2 C^2$ for C channels per layer. 每个卷积核大小为3*3*C, 共有C个, 然后总共3层.

## GoogleNet

**Deeper networks, with computational efficinecy.**

![linear](./images/Lec09/1%20(3).png){: width="600px" .center}

### Inception Module

Design a good local network topology(network within network) and then stack modules on top of each other.

![linear](./images/Lec09/1%20(4).png){: width="600px" .center}

注意, 上图中的第2, 3, 4列的1\*1 convolution layer, 被称为bottleneck layer, 作用是减少计算量, 也可以理解为降维. 使用一定量的1\*1卷积核, 可以在保留原图尺寸不变的情况下减少层数(depth), 防止computationally expensive.

???failure "Without bottleneck layers"
    ![linear](./images/Lec09/1%20(5).png){: width="600px" .center}
    加上bottleneck之后, 计算量会减小到1/3左右.
    ![linear](./images/Lec09/1%20(6).png){: width="600px" .center}

### Overall Hierachy

![linear](./images/Lec09/1%20(7).png){: width="600px" .center}

可以看到, 大致结构为:

+ 最左端的stem network, 由传统的卷积层和池化层线性组成.
+ 中间的stacked inception modules, 即很多inception module堆叠在一起.
+ 最右端的classifier output, 这里去除了传统CNN在末尾的多个全连接层, 只使用了一个全连接层用来输出score.
+ 很关键的一个点在于中间的**auxiliary classification output**, 这是为了防止梯度消失, 在训练过程中, 会对这个辅助分类器进行反向传播, 但是在测试过程中, 这个辅助分类器会被忽略.

## ResNet

**Very deep networks using residual connections.**

![linear](./images/Lec09/1%20(8).png){: width="600px" .center}

!!!success "Hypothesis"
    + The problem is an optimization problem, deeper models are harder to optimize.
    + The deeper model should be able to perform at least as well as the shallower model.
    + A solution by construction is copying the learned layers from the shallower model and setting additional layers to identity mapping.

Solution to train a deeper model: Use network layers to fit a residual mapping instead of directly trying to fit a desired underlying mapping.

![linear](./images/Lec09/1%20(9).png){: width="600px" .center}

常规的CNN都是希望一个layer的输出 $H(x)$ 能够逼近真实值 $F(x)$, 而ResNet则是希望layer的输出 $H(x)$ 能够逼近残差 $F(x) - x$, 即 $H(x) = F(x) - x$, 这样的话, 我们只需要训练 $F(x)$ 就可以了, 而不需要训练 $H(x)$, 因为 $H(x)$ 可以直接由 $F(x)$ 和 $x$ 相加得到. 这里 $H(x)$ 和 $F(x)$ 分别指的是layer实际的输出关于输入的函数, 和理想的输出关于输入的函数.

一点intuition, 这样学习的话就可以符合hypothesis, 即deeper model should be able to perform at least as well as the shallower model, 因为如果shalow layer已经学得很好了那么后面的layer可以学习出$F(x) = 0$, 即
$H(x) = x$, 这样的话, 整个网络就相当于一个浅层网络. 这说明了经过某层layer之后, 结果一定不会比之前坏.

![linear](./images/Lec09/1%20(10).png){: width="600px" .center}

For deeper networks, use "bottleneck" layer to improve efficiency, very similar to GooleNet.

![linear](./images/Lec09/1%20(11).png){: width="400px" .center}

> 注意, 因为每个残差结构的输出需要和输入相加, 所以应当保证二者尺寸相当.








