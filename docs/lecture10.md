# Lecture 10: Recurrent Neural Networks

???notes "somthing about lecture 9"
    ![linear](./images/Lec10/1%20(8).png){: width="600px" .center}

    在上面这张图中的DenseNet, FarctalNet或者之前学习过的ResNet, 他们都有一些跨越性的连接, 即将一层的输出越过下下面几层连接到后面的层去. 这样做的一点intuition就是相当于在反向传播时为梯度流动建立的高速公路, 这样梯度就可以快速地流动, 而不需要经过很多层, 可以避免梯度在每一层流动造成的过分衰减.

!!!info
    + [An amazing blog about RNN](https://towardsdatascience.com/recurrent-neural-networks-rnns-3f06d7653a85).
    + [An RNN network constructed with 112 lines of python and numpy](https://gist.github.com/karpathy/d4dee566867f8291f086).

## Recurrent Neural Networks: Process Sequences

![linear](./images/Lec10/1%20(9).png){: width="600px" .center}

最左侧的是我们之前讨论过的神经网络, 他们都是一个one to one 的函数映射, 而RNN可以处理输入数量不固定, 且输出数量不固定的问题.

+ one to many: Image Captioning, image -> sequence of words.
+ many to one: Sentimental Classification, sequence of words -> sentiment.
+ many to many: Machine Translation, sequence of words -> sequence of words.
+ many to many: Video Classification, sequence of frames -> category.

We can process a sequence of vectors $x$ by applying a recurence formula at every time step:

$$
h_t = f_W(h_{t-1}, x_t)
$$

where $h_t$ and $h_{t-1}$ are hidden states at time $t$ and $t-1$ respectively, $x_t$ is the input at time $t$, $f_W$ is some combination of a weight matrix and a non-linearility. And usually, we want to predict a vector at some time steps.

![linear](./images/Lec10/1%20(10).png){: width="600px" .center}

!!!warning 
    Notice: the same function and the same set of parameters are used at every time step.

## Vanilla Recurrent Neural Network

> The state consist a single hidden vector $h$.

![linear](./images/Lec10/1%20(11).png){: width="600px" .center}

## RNN: Computational Graph

> Re-use the same weight matrix at every time step.

![linear](./images/Lec10/1%20(12).png){: width="600px" .center}

我们在做反向传播的时候, 要对$W$求梯度, 需要在每个time step求$W$的梯度然后相加. 即先求$L$对每个loss $L_i$的梯度, 在求出每个$L_i$对$W$的梯度, 最后对于每个time step的梯度求和.

!!!info "Encoder and Decoder"
    ![linear](./images/Lec10/1%20(13).png){: width="600px" .center}
    Encoder将一些列的$x_i$输入编码成为一个向量, 交给Decoder进行解码, 然后Decoder将这个向量解码成为一系列的$y_i$.

!!!notes "Example"
    ![linear](./images/Lec10/1%20(14).png){: width="600px" .center}
    **At test time sample characters one at a time, feed back to model.** 这句话很重要, 我们不是选择argmax(probability)作为下一层输入的$x$, 而是依照概率分布来取样作为下一个time step的输入. 这样做的好处是一定程度上能够增大生成文本的diversity.

### Backpropagation Through Time

#### Naive Implementation

![linear](./images/Lec10/1%20(15).png){: width="600px" .center}

The naive idea is to just froward through the entire sequence to compute the loss, and then backward through entire sequence to compute gradient.

但仔细想想这样肯定不行, 假如我们的训练数据是整个wikipedia的文本, 这无疑会使得我们的计算过程非常复杂且缓慢, 内存开销也很大.

#### Truncated Backpropagation Through Time 

![linear](./images/Lec10/1%20(1).png){: width="600px" .center}

+ Run forward and backwrad through chunks of the sequence instead of whole sequence.
+ Carry hidden states forward in time forever, but only backpropagate for some smaller number of steps.

> 我们每次forward pass 和 backprop的时候, 都只用当前一段RNN来来进行求loss和梯度, 解决了计算量过大的问题.

## Image Captioning

> Combined both CNN and RNN.

![linear](./images/Lec10/1%20(2).png){: width="600px" .center}

!!!info
    ![linear](./images/Lec10/1%20(3).png){: width="600px" .center}
    我们将CNN提取出的图像特征输入到RNN中, 利用RNN生成$y_0$, 而后在$y_0$中sample出一个word, 作为下一层的输入, 如此迭代训练RNN网络, 直到最后从某个$y_i$中sample的word是$<End>$ token, 我们就结束caption.
    另外, 现在的一个recurrent module多了一个图像特征的输入, 所以需要加入一个新的weight matrix $W_{ih}$.

## Long Short Term Memory(LSTM)

### Vanilla RNN Gradient Flow

![linear](./images/Lec10/1%20(4).png){: width="600px" .center}

在一个单一的module中, 通过对$h_t$的梯度求解对$h_{t-1}$的梯度, 实际上是乘以了$W_{hh}^T$.

![linear](./images/Lec10/1%20(5).png){: width="600px" .center}

但是在这样一个很长的RNN架构下, 需要多次乘以$W_{hh}^T$, 可能会导致exploding gradients 或者 vanishing gradients.

Exploding gradients 相对好解决一些, 我们可以使用gradien clipping, 即如果梯度太大我们就把他放缩到某一个threshold. 但是梯度趋于零这件事不是很好解决, 需要我们修改一下RNN的架构, 这也就引入了下面的LSTM.

### LSTM

![linear](./images/Lec10/1%20(6).png){: width="600px" .center}

+ LSTM的引入了除$h_i$之外的另一个状态$c_i$.
+ f: Forget Gate, whther to erase cell.
+ i: Input Gate, whether to write to cell.
+ g: G Gate, how much to write to cell.
+ o: Output Gate, how much to reveal cell.

![linear](./images/Lec10/1%20(7).png){: width="600px" .center}

我们可以看到, 如果对$C_i$求梯度, 那么整个backprop的过程是不存在矩阵乘法的, 只有向量点乘. 而且, 由于每一个module生成的forget gate大概率不完全一样, 所以不会存在连续点乘一个向量的情况. 这样以来, 相当于我们为$C_i$的gradient flow 建立了一条"高速公路", 可以便捷且准确地求出其梯度. 这一思想和resnet, densenet都很相似.

但是W的梯度其实才是我们真正care的. 由于$W$的梯度一方面来自$C$, 另一方面来自$h$, 我们优化了$C$的gradient flow, 也就相应优化了$W$的gradient flow.

## Other RNN Variants

### GRU

Learning phrase representations using RNN encoder-decoder for statistical machine translation, 2014.