# Lecture 13: Generative Models

## Unsupervised Learning

+ Just data, no labels, which means training is cheap.
+ Learn some underlying hidden structure of the data.

## Generative Model

Given training data, generate new samples from same distribution.

![linear](./images/Lec13/1%20(13).png){: width="600px" .center}

!!!info "Taxonomy of Generative Models"
    ![linear](./images/Lec13/1%20(14).png){: width="600px" .center}

### PixelRNN and PixelCNN

Explicit density model, which use chain rule to decompose likelihood of an image x into product of 1-d distributions:

$$
p(x) = \prod_{i=1}^{n}p(x_i|x_1, ..., x_{i-1})
$$

+ On the left is the likelihood of image x, while on the right is the probabilities of ith pixel value given all previous pixels.

+ What we want to do is to maximize the likelihood of the training data.

+ But how do we know about the complex distribution over pixel values? Express them using a neural network.

+ Also, we need to define ordering of "previous pixels".

#### PixelRNN

![linear](./images/Lec13/1%20(15).png){: width="600px" .center}

+ Generate image pixels starting from corner.
+ Dependency on previous pixels modeled using an RNN(LSTM).
+ Drawback: Sequential generation is slow.

#### PixelCNN

![linear](./images/Lec13/1%20(16).png){: width="600px" .center}

+ Still generate image pixels starting from corner.
+ Dependency on previous pixels now modeled using a CNN over context region.
+ Training: Maximize likelihood of training images.
+ Training is faster than PixelRNN (can parallelize convolutions since context region values known from training images).
+ Generation must still proceed sequentially => still slow.

???tips "Pros and Cons"
    ![linear](./images/Lec13/1%20(17).png){: width="600px" .center}

### Variational Autoencoder(VAE)

PixelCNNs define tractable density function, optimize likelihood of training data:

$$
p(x) = \prod_{i=1}^{n}p(x_i|x_1, ..., x_{i-1})
$$

VAEs define intractable density function with latent z:

$$
p_\theta (x) = \int p_\theta(z)p(x|z)dz
$$

$z$可以认为是一些提取出来的feature.

Cannot optimize directly, derive and optimize lower bound on likelihood instead, which will be discussed later.

#### Autoencoders

Unsupervised approach for learning a lower-dimensional feature representation from unlabeled training data.

![linear](./images/Lec13/1%20(18).png){: width="600px" .center}

我们使用诸如CNN的手段提取出feature Z, 用于捕捉图像中的meaningful factor.

![linear](./images/Lec13/1%20(19).png){: width="600px" .center}
通过训练, 我们希望feature Z能重构出原始图像, 使用L2 loss:

$$
||x - \hat{x}||_2^2
$$

训练之后, 我们throw away decoder部分, 只保留encoder部分用于提取图像的特征Z.

#### Variational Autoencoders

![linear](./images/Lec13/1%20(20).png){: width="600px" .center}

使用encoder生成z的分布, 然后根据均值和方差采样生层z, 再通过decoder生成重构图像的均值和方差, 采样生成重构图像.

![linear](./images/Lec13/1%20(21).png){: width="600px" .center}

这里是$p_\theta(x^{(i)})$表达式的推导过程. 最后一行中,前两项可以认为是tractable的lower bound, which we can take gradient of and optimize. 第三项是intractable的, 但是其$\geq 0$. 所以我们在训练时优化前两项即可, 相当于最大化训练数据集上概率的lower bound.

!!!info
    ![linear](./images/Lec13/1%20(22).png){: width="600px" .center}

???example
    |   ![linear](./images/Lec13/1%20(1).png){: width="300px" .center}  | ![linear](./images/Lec13/1%20(2).png){: width="600px" .center} |
    | --------- | ----------- |

???tips "Pros and Cons"
    ![linear](./images/Lec13/1%20(3).png){: width="600px" .center}

#### Generative Adversarial Networks(GAN)

What if we give ip on explicitly modeling density, and just want ability to sample?

GANS: don't work with any explicitly density function, instead, take game theoretic approach: learn to generate from training distribution through 2-player game.

![linear](./images/Lec13/1%20(4).png){: width="600px" .center}

我们需要训练两个网络, 一个generator, 一个discriminator, 二者同时训练.

![linear](./images/Lec13/1%20(5).png){: width="600px" .center}

discriminator希望最大化这个函数, 即当输入为真实数据时, 输出为1, 当输入为生成数据时, 输出为0. generator希望最小化这个函数, 即能够成功骗过discriminator.

| ![linear](./images/Lec13/1%20(6).png){: width="600px" .center}    | ![linear](./images/Lec13/1%20(7).png){: width="600px" .center} |
| --------- | ----------- |

!!!note "GAN training algorithm"
    ![linear](./images/Lec13/1%20(8).png){: width="600px" .center}

After training, we only use the generator network to generate new images.

!!!info "Results of generation"
    ![linear](./images/Lec13/1%20(9).png){: width="600px" .center}
    我们可以看出generator不会直接生成一个完全和训练集某个数据相同的图像, 而是在学习training set的基础上生成新图像.

???tips "Generative Adversarial Nets: Convolutional Architectures"
    ![linear](./images/Lec13/1%20(10).png){: width="600px" .center}
    ![linear](./images/Lec13/1%20(11).png){: width="600px" .center}
    ![linear](./images/Lec13/1%20(12).png){: width="600px" .center}






