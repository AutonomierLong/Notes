# Lecture 7: Training Neural Networks, Part 2

## Fancier Optimization

### Problem with SGD Optimization

+ What if loss changes quickly in one direction and slowly in another? What does gradient descent do?
    This means loss function has high **condition number:** ratio of largest to smallestsingular value of the Hessian matrix is large. 其实就是梯度最大/最小方向的梯度之比.

    ![linear](./images/Lec07/1%20(2).png){: width="300px" .center}

    可以看到, 对于SGD来讲, 一开始在左右两边向上的梯度较大, 前后两边向下的梯度较小, 那么前后方向往下走的分量会比较小, 导致最开始SGD会zig-zag振荡, 显然这样不如直接走直线来得有效率.

    > Very slow progress along shallow dimension, jitter along steep direction.

+ What if the loss function has a local minima or saddle point? 
    ![linear](./images/Lec07/1%20(3).png){: width="600px" .center}
    
    Saddle points much more common in high dimension. 因为在高维空间内, local minima需要所有参数都在此处达到局部最小, 即两边的导数都小于零, 这样的概率很低, 但是saddle points只需要一边小于零, 另一边大于零, 因此saddle points更多.

    Zero gradient, gradient descent gets stuck. 在这两种位置, 梯度都比较小, gradient descent进行地很慢! 

+ Our gradients come from minibatches so they can be noisy. 从mini-batch中获得的梯度不一定能真正代表整个训练集的梯度, 因此在进行梯度下降时可能存在扰动, 不能完全按照正确路径行进.

### SGD + Momentum

$$
v_{t+1} = \rho v_t + \nabla f(x_t)
$$

$$
x_{t+1} = x_t - \alpha v_{t+1}
$$

```python
vx = 0 # initialize the velocity zero.
while True:
    dx = compute_gradient(x)
    vx = rho * vx + dx
    x += learning_rate * vx
```

+ <u>Build up "velocity"</u> as a running mean of gradients. 其实靠前的gradient权重小于后面的gradient.
+ Rho gives "friction", typically rho = 0.9 or 0.99. 

> SGD + Momentum addresses all the problem of SGD !

+ Poor conditioning的情况下, 原来我们会在一开始在梯度较大的方向振荡, 但是加上momentum之后, 两次震荡的梯度会一定程度上相互抵消, 减少振荡, 加快gradient descent.
+ 在local minima 和 saddle point处, 由于我们现在存在一个momentum项, 积累了先前的梯度, 所以可以较快经过梯度平缓的区域.
+ Momentum term sort of average the random gradient noise.

### Nesterov Momentum

![linear](./images/Lec07/1%20(4).png){: width="600px" .center}

区别在于, 区别于之前的SGD + Momentum, 这里我们先沿着velocity走一步, 再计算那里的梯度, 最后形成actual step.

$$
v_{t+1} = \rho v_t + \alpha \nabla f(x_t + \rho v_t)
$$

$$
x_{t+1} = x_t + v_{t+1}
$$

变量替换一下$\hat{x_t} = x_t + \rho v_t$:
$$
v_{t+1} = \rho v_t - \alpha \nabla f(\hat{x_t})
$$

$$
\hat{x_{t+1}} = \hat{x_t} + v_{t+1} + \rho (v_{t+1} - v_t)
$$

这样变换之后的主要目的是方便我们计算梯度, 直接计算$f(\hat{x_t})$的梯度.

```python
dx = compute_gradient(x)
old_v = v
v = rho * v - laerning_rate * dx
x += v + rhp(v - old_v)
```
!!!notes
    其实Nesterov Momentum相比最基本的Momentum, 就是加上了$\rho (v_{t+1}-v_t)$这一项, 这一项的作用可以使得我们当前走出的一步更多地参考了一点之前的速度, 从而防止太激进的走法. 但需要注意的是, 原始的Momentum已经在走当前步骤时参考了之前步骤, 而现在我们相当于是给之前的速度加了一些权重.
    ![linear](./images/Lec07/1%20(5).png){: width="400px" .center}
    由这张对比图可以看出, 添加了Momentum的optimization算法效率显著高于普通SGD, 但他们都存在一个走过了最优点之后再纠正放回走的过程, 但是Nesterov Momentum没有SGD + Momentum那么激进.

### AdaGrad

Added element-wise scaling of the gradient based on the historical sum of squares in each dimension.

```python
while True:
    dx = compute_gradient(x)
    grad_squared += dx * dx
    x -= learning_rate * dx / (np.sqrt(grad_squared) + 1e-7)
```

+ Intuitely, this algorithm makes the step size in every dimension closer, since all of the gradients are divided by $dx*dx$.
+ But the step size of each dimension will be smaller and smaller during the training time, since `grad_squared` is monotonically increasing.
+ AdaGrad is not very popular in practice, because it makes the step size too small. But we have a modified version.

### RMSProp

```python
grad_squared = 0
while True:
    dx = compute_gradient(x)
    grad_squared = decay_rate * grad_squared + (1 - decay_rate) * dx * dx
    x -= learning_rate * dx / (np.sqrt(grad_squared) + 1e-7)
```

+ RMSProp is a modification of AdaGrad, which makes the step size decay slower.

!!!notes
    RMSProp在训练过程中使得每个维度的步长更加均匀, 是与区别于momentum的另一种加速思路.
    ![linear](./images/Lec07/1%20(6).png){: width="400px" .center}
    可以看到, RMS不会像momentum那样先越过再纠正.

### Adam
> Why don't we just combine the momentum and RMSProp?

#### Vinilla Version

```python
first_momentum = 0
second_momentum = 0
while True:
    dx = compute_gradient(x)
    first_momentum = beta1 * first_momentum + (1 - beta1) * dx
    second_momentum = beta2 * second_momentum + (1 - beta2) * dx * dx
    x -= learning_rate * first_momentum / (np.sqrt(second_momentum) + 1e-7)
```

However, if we implement just like this , over initial steps will be gigantic, since at that time the second momentum is relatively small.

#### Unbiased Version

```python
first_momentum = 0
second_momentum = 0
while True:
    dx = compute_gradient(x)
    first_momentum = beta1 * first_momentum + (1 - beta1) * dx
    second_momentum = beta2 * second_momentum + (1 - beta2) * dx * dx
    first_unbias = first_momentum / (1 - beta1 ** t)
    second_unbias = second_momentum / (1 - beta2 ** t)
    x -= learning_rate * first_unbias / (np.sqrt(second_unbias) + 1e-7)
```

!!!info 
    Adam with bata1 = 0.9, beta2 = 0.999 and learning_rate = 1e-3 or 5e-4 is a great starting point for many models.

### Learning Rate Decay

#### Exponential Decay

$$
\alpha = \alpha_0 e^{-kt}
$$

#### 1/t Decay

$$
\alpha = \frac{\alpha_0}{1 + kt}
$$

!!!info 
    ![linear](./images/Lec07/1%20(7).png){: width="400px" .center}
    如果Loss与时间的关系长这个样子, 说明在某些节点进行了learning rate decay. Intuitely, 在某些时候loss下降到一定程度, 当下的learning rate可能会导致loss在最优解附近震荡, 所以此时需要decay一下learning rate, 逼近最优.
    <br>不过对于Adam这种step size本身就会逐渐降低的算法, 不是那么关键, 但是对于momentum算法还是十分重要的.

### Second Order Optimization

What we've talked about is all first order optimization, which is based on gradient. But there are second order optimization algorithms, which is based on Hessian matrix.

![linear](./images/Lec07/1%20(8).png){: width="600px" .center}

!!!failure
    但是这种算法一般不会在deep learning中使用, 因为计算Hessian matrix的复杂度太高, 而且对于如果参数规模达到million级别, 内存也存不下整个Hessian Matrix.

## Model Ensembles

1. Train multiple independence models.
2. At test time average their results.

This will enjoy about 2% extra performance.

!!!info "Model Ensembles: Tips and Tricks"
    ![linear](./images/Lec07/1%20(9).png){: width="600px" .center}
    <br>
    ![linear](./images/Lec07/1%20(10).png){: width="600px" .center}

## Regularization

> How to improve a single-model performance? -- Regularization!

We've already learnt that we can add term to loss, say L2 regularization, L1, etc.

### Dropout

+ In each forward pass, randomly set some neurons to zero
+ Probability of dropping is a hyperparameter, 0.5 is common

![linear](./images/Lec07/1%20(11).png){: width="600px" .center}

```python
p = 0.5

def train_step(X):
    """ X contains the data """
    H1 = np.maximum(0, np.dot(W1, X))
    U2 = np.random.randn(*H1.shape) > p # first dropout mask
    H1 *= U2 # drop
    U2 = np.maximum(0, np.dot(W2, H1)) # second dropout mask
    H2 *= U2 # drop
    out = np.dot(W3, H2) + b3

    # backprop and parameter update not showm.
```

!!!info "Intuition"
    How can this possibly be a good idea?
    ![linear](./images/Lec07/1%20(12).png){: width="600px" .center}
    这样做可以使得我们不要过分地依赖一个特征来估计结果, 而是将权重平均分配到各个特征上去. 即当某个特征被drop之后, 我们仍能较好地预测结果.

!!!info "Another Interpretation"
    + Dropout is training **a large ensemble of models** that share parameters. 
    + Each binary mask is one model. 
    + So for a FC layer with 4096 units, it has $2^{4096} = 10^{1233}$ possible masks.

!!!notes "during test time"
    ![linear](./images/Lec07/1%20(13).png){: width="600px" .center}
    <br>
    ![linear](./images/Lec07/1%20(14).png){: width="600px" .center}

```python
def predict(X):
    #ensembled forward pass
    H1 = np.maximum(0, np.dot(W1, X) + b1) * p # Note: Scale the activations
    H2 = np.maximum(0, np.dot(W2, H1) + b2) * p # Note: Scale the activations
    out = np.dot(W3, H2) + b3
```

**At test time all neurons are active always**, so we must scale the activations so that for each neuron: output at test time = expected output at training time.

But a more common case is that we do a "inverted dropout", which means we divid the each layer's activation by p dduring training, leaving testing part unchanged.

### Data Augmentation

![linear](./images/Lec07/1%20(15).png){: width="600px" .center}

+ Horizontal Flips
+ Random Crops and Scales
    Training: sample random crops / scales.
    Testing: average a fixed set of crops.

+ Color Jitter
    ![linear](./images/Lec07/1%20(16).png){: width="600px" .center}

## Transfer Learning

> You need a lot of a data if you want totrain/use CNNs.

Transfer learning busted this sort of idea.

![linear](./images/Lec07/1%20(17).png){: width="600px" .center}

!!!info "Transfer learning with CNNs is pervasive"
    ![linear](./images/Lec07/1%20(1).png){: width="600px" .center}
    当我们拿到手一个任务时, 一般可以先找找有没有在类似任务上训练过的神经网络, 有的话我们其实只需要拿过来将最后几层重新训练一下即可.

!!!info
    Deep learning frameworks(Pytorch/TensorFlow/Caffe) provide a “Model Zoo” of pretrained models so you don’t need to train your own.


