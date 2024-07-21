# Lecture 12: Visualizing and Understanding

> An neural network, especially a convolutional neural network, is just like a blackbox to us, where we input an image and get some results. But how does it actually work, what's all those intermediare feature looking for? These questions are what we are going to answer in this lecture.

## Visualizing Filters

### First Layer

![linear](./images/Lec12/1%20(1).png){: width="600px" .center}

可以看到第一层卷积核往往是在寻找一些较为简单的特征, 比如一些edge, blob之类.

!!!question
    为什么卷积核长什么样子代表了他寻找什么样的特征?
    <br>卷积核与我们的图像进行的是点积操作, 在二者相同的情况下响应值最大.

### Other Layers

!!!info
    我们也可以可视化一下除了first layer之外的其他layer, 但是由于这些layer作用的对象并不是图像本身, 而是经过一些filter处理的activation map, 所以他们的可视化结果并不是很直观.

### Last Layer

Suppose our last layer is mapping a 4096 feature vector to 10 categories. We run the network on many images, then collect the feature vectors for all these images.

#### Nearest Neighbour

For each of the image, we take its fearure vector, and find the nearest neighbour of its feature vactor produced by other images, then compare these images. The results are as follows:

![linear](./images/Lec12/1%20(2).png){: width="300px" .center}

> 不同于直接对图像求nearest neighbour, 我们对特征向量求nearest neighbour, 可以大大增加图像匹配的准确率, 比如第二行中一只头朝左的大象可以成功匹配到一只头朝右的大象, 这说明只要是大象这一类, 提取出来的feature vector都很相近.

#### Dinmensionality Reduction

Visualize the "space" of feature vectos by reducing dimensionality of vectors from 4096 to 2 dimensions.

Algorithms: Principle Component Analysis(PCA), t-SNE...

![linear](./images/Lec12/1%20(3).png){: width="300px" .center}

> 可以看到有明显的聚类效果, 不同的类别被分成了一个个cluster.

![linear](./images/Lec12/1%20(4).png){: width="600px" .center}

> 可以看到, 在被reduced之后的二维空间中, 不同类别的图像大致被分在了不同区域, 如右上角有一些都有着蓝天背景的图片, 左下角则是一些植物. See high-resolution version at [http://cs.stanford.edu/people/karpathy/cnnembed/](http://cs.stanford.edu/people/karpathy/cnnembed/).

## Visualizing Activations

### Maximally Activating Patches

+ Pick a layer and a channel, e.g. conv5 is 128\*13\*13, pick channel 17/128.

+ Run many images through the network, record values of chosen channel.

+ Visualize image patches taht correspond to maximal actiavtions.

因为一个channel是由同一个卷积核生成的, 所以这一层上不管什么位置的高响应值都对应着在某张图片的某个位置出现了特定特征. 我们将所有在这一层的这一channel有着高响应值的图片提取出来, 然后截取出高响应值对应的reception field, 就得到了这些图片上出现特定特征的位置.

![linear](./images/Lec12/1%20(5).png){: width="600px" .center}

> 可以看到, 每个channel所寻找的特征不尽相同, 比如第一行就是在寻找类似一个"洞"的特征, 第二行则是在寻找一些弧形边缘.

### Occlusion Experiments

Mask part of the image before feeding to CNN, draw hetmap of probability at each mask location.

![linear](./images/Lec12/1%20(6).png){: width="600px" .center}

这一做法可以帮助我们了解到, 图像的那个部位对图像识别的结果影响最大, 红色区域是影响最大的, 黄色区域则影响较小.

### Saliency Map

+ How to tell which pixel matter for classification?
+ Compute gradient of (unnormalized) class score with respect to image pixels, take absolute value and max over RGB channels.

![linear](./images/Lec12/1%20(7).png){: width="600px" .center}

这样我们就可以看出图像的哪一部分对识别的贡献较大.

### Imtermediate Features via (guided) backprop

+ Pick a single intermediate neruro, e.g. one value in 128\*13\*13 conv5 feature map.
+ Compute gradient of neuron value with respect to image pixels.

+ Images come out nicer if you only backprop positive gradients through each ReLU (guided backprop).

![linear](./images/Lec12/1%20(8).png){: width="600px" .center}

## Visualizing CNN Features: Gradient Ascent

**Guided Backprop:** Find the part of an image that a neuron responds to.
**Gradient Ascent:** Generate a synthetic iamge that maximally activates a neuron.

$$
I^* = argmax_I f(I) + R(I)
$$

where $f(I)$ is the neuron value, and $R(I)$ is the iamge regularizer.

+ Initialize image to zeros.

Repeat:

+ Froward image to compute current scores.
+ Backprop to get gradient of neuron value with respect to image pixels.
+ Make a small update to the image.

现在假如我们试图通过最大化某个类别的分数, 来生成该类别的一张图片:

$$
argmax_I S_c(I) - \lambda||I||^2_2
$$

Simple Regularizer: Penalize L2 norm of generated image.

Better Regularizer: Penalize L2 norm of iamge; also during optimaization periodically:
1. Gaussian blur image
2. Clip pixels with small values to 0.
3. Clip pixels with small gradients to 0.

| ![linear](./images/Lec12/1%20(9).png){: width="600px" .center}    | ![linear](./images/Lec12/1%20(10).png){: width="600px" .center} |
| --------- | ----------- |

> 可以看到生成的图片都较为合理的, 有些达到了不错的效果, 如flamingo, billiard table.

!!!info
    We can use the same approach to visualize intermediate features:
    ![linear](./images/Lec12/1%20(11).png){: width="600px" .center}

???tip "mutil-faceted"
    ![linear](./images/Lec12/1%20(12).png){: width="600px" .center}


## Deep Dream: Amplify existing features

| ![linear](./images/Lec12/1%20(13).png){: width="650px" .center}    | ![linear](./images/Lec12/1%20(14).png){: width="600px" .center} |
| --------- | ----------- |

## Feature Inversion

Given a CNN feature vector for an image, find a new image that:

+ Matches the given feature vector.
+ Looks natural.
  
$$
x^* = \mathop{\arg\min}_{x \in \mathbb{R}^{H \times W \times C}} L(\Phi(x), \Phi_0) + \lambda R(x)
$$

where
$$
L(\Phi(x), \Phi_0) = ||\Phi(x) - \Phi_0||^2
$$

$$
R(x) = \sum_{i, j}((x_{i, j+1}-x_{i, j})^2+(x_{i+1, j}-x_{i, j})^2)^{\frac{\beta}{2}}
$$
This regularizer is called Total Variation Regularizer, which encourages spatial smmothness.

![linear](./images/Lec12/1%20(15).png){: width="600px" .center}

> 可以看到, 使用越低维的特征, 生成的图片与原图越吻合, 这说明了高维特征信息会丢弃掉一些不那么重要的信息, 只保留图像一些比较显著的特征.

## Texture synthesis

Given a simple patch of some texture can we generate a bigger image of the same texture.

![linear](./images/Lec12/1%20(16).png){: width="300px" .center}

| ![linear](./images/Lec12/1%20(17).png){: width="600px" .center}    | ![linear](./images/Lec12/1%20(18).png){: width="600px" .center} |
| --------- | ----------- |

## Neural Style Transfer

![linear](./images/Lec12/1%20(19).png){: width="600px" .center}

We combine feature reconstruction and texture synthesis together to do the neural style transfer.
 