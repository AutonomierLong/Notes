# Transformer

## Self-Attention

### Input

输入的vector不止一个, 而且向量个数也不确定, 所以我们需要对向量进行编码.

![linear](./images/Trans/1%20(1).png){: width="600px" .center}

One-hot encoding 编码在vocabulary库很大的情况下需要极大的存储空间, 性能较差, 所以我们使用word embedding进行编码.

假设我们的vocabulary总共有$N$个word, 需要把每个word编码成$M$维的向量, 那么我们需要听过某些机器学习的手段学习一个$W$矩阵, 灭一行对应一个word的编码:

$$
W = \begin{pmatrix}
w_{11} & w_{12} & \cdots & w_{1M} \\
w_{21} & w_{22} & \cdots & w_{2M} \\
\vdots & \vdots & \ddots & \vdots \\
w_{N1} & w_{N2} & \cdots & w_{NM}
\end{pmatrix}
$$

所以在使用每个词的时候我们只需要在这个矩阵中找到该word对应的编码即可, 使用这个编码作为self-attention的输入.

另外, 我们使用PCA将word embedding之后的向量降维到二维平面上, 发现词义相近的word靠的更近, 有一定的聚类效果.

### Output

![linear](./images/Trans/1%20(2).png){: width="600px" .center}

大致就是三种可能性, 分别为对每个输入都输出一个标签, 对整个输入序列输出一个标签, 和由网络自行决定生成几个标签.

### Architecture

#### Overall Principle

![linear](./images/Trans/1%20(3).png){: width="600px" .center}

上面这个网络展示了只有一层self-attention的架构, 我们可以看到, self-attention层综合衡量了四个输入的vector, 然后产生了四个输出vector, 每个输出的vector都既包含了当前位置的输入信息, 又包含了整个序列的信息. 然后对于self-attention的输出, 我们使用一个FC layer来预测类别.

![linear](./images/Trans/1%20(4).png){: width="600px" .center}

+ 可以把多层self-attention叠起来, 中间加上FC layers. 所以一个self-attention的输入不一定就是原始输入, 也可能是hidden layer产生的输出.
+ 在通过$a^i$计算$b^i$的时候, 其实我们考虑了其他$a^j(i\neq j)$与$a^i$之间的相关性, 具体地计算细节在下文展开叙述.

#### Detailed Architecture

![linear](./images/Trans/1%20(5).png){: width="600px" .center}

一般self-attention layer的细节架构有两种, 一种是左边的dot-product, 一种是右边的additive, 但是左侧的dot-product更为主流, 也是我们主要讨论的.

具体来讲, 计算过程分为三步, 我们以通过$a^1$计算$b^1$为例:

![linear](./images/Trans/1%20(6).png){: width="600px" .center}

我们实现通过$W^q$矩阵计算出$q^1$, 再通过$W^k$矩阵分别计算出$a^2$到$a^4$, 然后计算出他们之间的dot-product.

![linear](./images/Trans/1%20(7).png){: width="600px" .center}

注意我们也需要计算$q^1$和$k^1$的点积, 即自己的query和自己的key之间的点积. 计算出这些点积之后, 我们将其通过某个activation function, 如这里的softmax.

![linear](./images/Trans/1%20(8).png){: width="600px" .center}

我们还需要通过$W^v$矩阵计算出$v^i$向量, 然后extract  information based on attention scores. 即下式:

$$
b^i = \sum_j \alpha^{'}_{i, j} v^j
$$

这样我们就得到了self-attention的输出.

#### Parallel

Self-Attention layer每个输出的计算都是可以并行的, 所以效率较高.

$$
q^i = W^q a^i,\ \ (q^1, q^2, q^3, q^4) = W^q (a^1, a^2, a^3, a^4)
$$

$$
k^i = W^k a^i,\ \ (k^1, k^2, k^3, k^4) = W^k (a^1, a^2, a^3, a^4)
$$

$$
v^i = W^v a^i,\ \ (v^1, v^2, v^3, v^4) = W^v (a^1, a^2, a^3, a^4)
$$

where $a, q, k, v$ are all column vactors.

![linear](./images/Trans/1%20(9).png){: width="600px" .center}

![linear](./images/Trans/1%20(10).png){: width="600px" .center}

$$
(b^1, b^2, b^3, b^4) = (v^1, v^2, v^3, v^4) \cdot A^{'}
$$

![linear](./images/Trans/1%20(11).png){: width="600px" .center}

Summary:

![linear](./images/Trans/1%20(12).png){: width="600px" .center}

### Muti-head Self-Attention

In vanilla self-attention, we only considered one type of relevance, therefore there's only one $q^i, k^i, v^i$. Now in the muti-head version, we need to consider different types of relevance. Take, 2-head for example.

![linear](./images/Trans/1%20(13).png){: width="600px" .center}

### Positional Encoding

+ No position information in self-attention.
+ Each position has unique positional vactor $e^i$.
+ hand-crafted.
+ learned from data.

![linear](./images/Trans/1%20(14).png){: width="200px" .center}

???info "Some methods to perform positional encoding"
    ![linear](./images/Trans/1%20(15).png){: width="600px" .center}

### Application

#### Self-attention for Speech

![linear](./images/Trans/1%20(16).png){: width="600px" .center}

因为speech的信息每10ms就会生成一个vector, 所以通常输入向量都很多, 如果计算$A^{'}$矩阵的话就会过于庞大, 消耗太多内存, 所以我们使用Truncated Self-attention. 即预测每个输出的时候只看周围的一部分.

#### Self-attention for Image

![linear](./images/Trans/1%20(17).png){: width="600px" .center}

+ CNN: self-attention that can only attends in a receptive field.
+ CNN is asimlified self-attention.
+ Self-attention: CNN with learnable receptive field.
+ Self-attention is the complex version of CNN.

![linear](./images/Trans/1%20(18).png){: width="600px" .center}

因为CNN可以看做self-attention的一个子集, 所以self-attention的可扩展性和解决问题的广度更好, 这也意味着对于CNN擅长的工作(如图像识别), CNN会在更小的数据集上表现更好, 而当数据集非常的时候, self-attention会表现更好.

#### Self-attention v.s. RNN

![linear](./images/Trans/1%20(19).png){: width="600px" .center}

RNN是不可并行的, 所以self-attention基本上是完胜的.

## Transformer

Transformer解决的事sequence-to-sequence(seq2seq)的问题.

![linear](./images/Trans/1%20(20).png){: width="600px" .center}

### Encoder

![linear](./images/Trans/1%20(21).png){: width="600px" .center}

|   ![linear](./images/Trans/1%20(22).png){: width="600px" .center}  | ![linear](./images/Trans/1%20(23).png){: width="600px" .center} |
| --------- | ----------- |

一个encoder由多个相同的block组成, 每个block内部包含了self-attention, residual和layer normalization三种结构. 具体来讲, 输入的向量先经过self-attention产生输出, 然后将输出通过残差结构加上输入, 最后做一个layer normalization,

### Decoder

#### Autoregressive(AT)

![linear](./images/Trans/1%20(24).png){: width="600px" .center}

Encoder的输出通过某种方式(后面会讲)给到Decoder, 然后decoder接受一个$<start>$ token开始输出预测结果, 下一步的输入设置为上一步预测结果中概率最高的编码.

Decoder中的attention module和encoder不太一样, 多了一个mask, 称为Masked Self-attention.

|   ![linear](./images/Trans/1%20(25).png){: width="600px" .center}  | ![linear](./images/Trans/1%20(26).png){: width="700px" .center} |
| --------- | ----------- |

现在每一个输出都只考虑在该当前输出之前的序列信息与当前输入的relevance. 这其实非常符合我们的直觉.

我们通常还会有个$<end$ token, 作为stop token, 在decoder输出这个token之后, 我们就结束生成.

#### Non-autoregressive(NAT)

![linear](./images/Trans/1%20(27).png){: width="600px" .center}

+ How to decide the output length for NAT decoder?
    + Another predictor for output length.
    + Output a very long sequence, ignore tokens after END.
+ Advantage: parallel, more stable generation.
+ NAT is usually worse than AT.

### Encoder - Decoder

![linear](./images/Trans/1%20(28).png){: width="600px" .center}

Decoder中有个部分(Cross Attention)用于接受Encoder传来的信息.

#### Cross Attention

![linear](./images/Trans/1%20(29).png){: width="600px" .center}

我们对masked self-Attention的输出求出query, 然后对encoder的输出vector计算出key, 再分别计算query与每个key的dot product, 最后合成$v$向量作为FC layers的输入.

### Training

Teacher Forcing: using the ground truth as input.

![linear](./images/Trans/1%20(30).png){: width="600px" .center}

但是在测试的时候我们会使用前一步输出的结果作为下一步输出.

但是测试时可能会出错, 怎么尽量避免前一步错导致后面步步错呢?

#### Scheduled Sampling

Scheduled Sampling 是一种技术，用于在训练序列生成模型（例如 RNNs 和 Transformer）时解决暴露偏差（exposure bias）的问题。暴露偏差指的是模型在训练过程中总是接收真实的数据作为输入，而在测试时只能接收模型自身生成的数据作为输入，这种差异会导致模型在实际应用中的表现不佳。

Scheduled Sampling 通过逐步改变模型在训练过程中使用的输入数据来源来缓解这一问题。具体做法如下：

1. **初始阶段**：在训练的初期，模型完全使用真实的数据作为输入，就像传统的教师强制（teacher forcing）方法一样。教师强制法是指每一步生成时，模型使用上一步的真实输出作为输入。

2. **逐渐过渡**：随着训练的进行，模型逐渐增加使用自身生成的数据作为输入的概率。这是通过引入一个概率 \( \epsilon \) 来实现的。每一步生成时，模型有概率 \( 1 - \epsilon \) 使用真实的输入，有概率 \( \epsilon \) 使用模型自己生成的输入。这个概率 \( \epsilon \) 随着训练的进行逐渐增加，从而使得模型逐渐适应在测试时的输入情况。

3. **最终阶段**：在训练的后期，模型完全使用自身生成的数据作为输入，模拟实际应用中的情况。

Scheduled Sampling 的优点是可以使模型逐步适应生成过程中使用自身输出作为输入的情况，从而减少暴露偏差，提高模型在实际应用中的表现。


