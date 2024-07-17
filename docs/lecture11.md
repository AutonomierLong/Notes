# Lecture 11: Detection and Segmentation

## Semantic Segmentation

![linear](./images/Lec11/1%20(1).png){: width="400px" .center}

+ Label each pixel in the image with a category label.
+ Don't differentiate instances only care about pixels. 就比如上面右侧的图片中, 可以划分出奶牛在哪里, 但是明确地区分两头之间的界限.

### Idea: Fully Convolutional

![linear](./images/Lec11/1%20(2).png){: width="600px" .center}

+ Design network as a bunch of convolutional layers, with downsampling and upsampling inside the network.
+ We can perform downsampling by Pooling or strided convolution, but what about upsampling?

### Upsampling

#### Unpooling

![linear](./images/Lec11/1%20(3).png){: width="600px" .center}

> Nearest Neighbour 可以理解为average pooling的逆操作, 而Bed of Nails可以理解为max pooling的逆操作.

![linear](./images/Lec11/1%20(4).png){: width="600px" .center}

我们可以将整个网络设计为一个对称的结构, downsampling时的每个max pooling对应upsampling 时的一个max unpooling, 在unpooling的时候我们根据max pooling时选取的最大值的位置进行填充.

但是上述这三个upsampling的算法都和pooling一样, 没有wight parameter, 所以不是learnable的.

![linear](./images/Lec11/1%20(5).png){: width="600px" .center}

以上面这张图为例, 我们有一个3\*3的参数filter, 在2\*2的分辨率上对于每个元素乘以该filter, 作为upsampling结果的一部分. 这一过程可以理解为卷积的逆过程.

???Notes "1D Example"
    ![linear](./images/Lec11/1%20(6).png){: width="600px" .center}


## Classification+ Localization

Localization 其实就是在图片中对**单个**物体画出一个矩形边框.

![linear](./images/Lec11/1%20(7).png){: width="600px" .center}

fc layer之前的convolutional layer一般是在imageNet上pretrain出来的.

这一我们需要将两个不同的loss加和成一个总loss, 需要对两个loss分配不同的权重, 这两个权重也是hyperparameter, 需要不断调整.

通过预测点坐标和边框宽度 高度, 我们实际上是把这个问题处理成为了一个regression problem.

### Aside: Human Pose Estimation

![linear](./images/Lec11/1%20(8).png){: width="600px" .center}

同样用regression的思路处理, 我们将人的姿态抽象成14个点坐标, 然后在卷积网络后面加上fc layer, 预测14个坐标, Loss是14个点左边的L2 Loss之和.

!!!info Human Pose Estimation
    ![linear](./images/Lec11/1%20(9).png){: width="600px" .center}

## Object Detection

### Region Proposals

+ Find "blooby" image regions that are likely to contain objects.
+ Relatively fast to run, r.g. Selective Search gives 1000 region proposals in a few seconds on CPU.

使用特定的算法可以在较为快速的时间内找到很多Region proposals, 这些regions都是较为可能包含某个object的region, 所以我们在这些region proposal上再去操作.

### R-CNN

![linear](./images/Lec11/1%20(10).png){: width="600px" .center}

这样的R-CNN需要对每个region proposal进行foward pass和backward pass. 对于每个region proposal, 我们预测一个offset vector, 表示当前这个region proposal和一个真实的bounding box的差异有多少, 然后我们对这个预测结果使用SVM Loss.

!!!Failure "Problems"
    + Training is slow, takes a lot disk space, about 84 hours.
    + Inference(detection) is slow, about 47s/image with VGG 16.

### Fast R-CNN

![linear](./images/Lec11/1%20(11).png){: width="600px" .center}

我们先将整张图片在ConvNet中进行forward pass, 生成一张feature map, 然后在这张feature map上面再根据region proposals选取一些区域, 再通过Pooling 和 FC layer预测类别(softamx)和bounding box, 最后在训练时还是将两个Loss加权求和.

???info "Faster R-CNN: ROI Pooling"
    ![linear](./images/Lec11/1%20(12).png){: width="600px" .center}

> Fast R-CNN速度非常快, 快到了运行Region Proposals的时间成为了整个算法的bottleneck.

### Faster R-CNN

> Make CNN do proposals.

Insert Region Proposal Network(RPN) to predict proposals from features.

![linear](./images/Lec11/1%20(13).png){: width="600px" .center}

We need jointly train four losses:

1. RPN classify object/not object.

2. RPN regress box coordinates.

3. Final classification score(object calsses).

4. Final box coordinates.



???info "Test-time speed"
    ![linear](./images/Lec11/1%20(14).png){: width="600px" .center}

### YOLO/SSD

![linear](./images/Lec11/1%20(15).png){: width="600px" .center}

### Aside: Object Detection + Captioning = Dense Captioning

![linear](./images/Lec11/1%20(16).png){: width="600px" .center}

## Instance Segmentation

### Mask R-CNN

![linear](./images/Lec11/1%20(17).png){: width="600px" .center}

在Detection的同时(预测类别和bounding box), 我们同时继续为每个类别预测出一个mask, 用于认定哪一部分pixels是该类别.

!!!info 
    Mask R-CNN can also does pose.
    我们只需要在预测bounding box的同时将Joint coordinates也预测出来.
    ![linear](./images/Lec11/1%20(18).png){: width="600px" .center}




