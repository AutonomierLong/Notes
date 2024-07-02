# Lecture 3: Loss Functions and Optimization

## Loss Functions

> A loss function tells how good our current classifier is.

Given a dataset of examples $\{(x_i, y_i)\}_{i=1}^{N}$, where $x_i$ is image and $y_i$ is (integer) label.

Loss over the dataset is a sum of loss over examples:
$$
L = \frac{1}{N} \sum_{i}L_i(f(x_i, W), y_i)
$$

### Multiclass SVM Loss

$$
L_i = \sum_{j \neq y_i}
\begin{cases} 
    0 & \text{if } s_{y_i} \geq s_j + \Delta \\
    s_j - s_{y_i} + \Delta & \text{otherwise} \\
\end{cases}
$$

Or:
$$
L_i = \sum_{j \neq y_i}max(0, s_j - s_{y_i} + \Delta)
$$

```python
def L_i_vectorized(x, y, W):
    scores = W.dot(x)
    margins = np.maximum(0, socres - scores[y] + Delta)
    margins[y] = 0
    loss_i = np.sum(margins)
    return loss_i
```

The final Loss will be:
$$
L = \frac{1}{N} \sum_{i=1}^{N}L_i
$$

!!!note "Example"
    ![linear](./images/Lec03/lec03%20(1).png){: width="300px" .center}
    我们用第一行举例计算Loss(取$\Delta = 1$), $L = (5.1-3.2+1) + 0 = 2.9$.

!!!info
    ![linear](./images/Lec03/lec03%20(8).png){: width="600px" .center}
    The Multiclass Support Vector Machine "wants" the score of the correct class to be higher than all other scores by at least a margin of delta. If any class has a score inside the red region (or higher), then there will be accumulated loss. Otherwise the loss will be zero. Our objective will be to find the weights that will simultaneously satisfy this constraint for all examples in the training data and give a total loss that is as low as possible.

对于$max(0, -)$这种形式的函数会被称为hinge loss, 有些人还会使用 square hinge loss, 即$max^2 (0, -)$. The unsquared version is more standard, but in some datasets the squared hinge loss can work better. This can be determined during cross-validation.

### Softmax Loss

scores = unnormalized log probabilities of the classes.

$$
P(Y = k|X = x_i) = \frac{e^{s_k}}{\sum_j e^{s_j}} 
$$
where $s = f(x_i, W)$.

Want to maximize the log likelihood, or (for a loss function) to minimize the negative log likelihood of the correct class:

$$
L_k = -logP(Y = k|X = x_i)
$$

in summary:
$$
L_k = -log(\frac{e^{s_k}}{\sum_j e^{s_j}})
$$

!!!info "Comparison between softmax and SVM"
    ![linear](./images/Lec03/lec03%20(6).png){: width="600px" .center}
    Suppose I take a datapoint and I jiggle a bit (changing its score slightly). What happens to the loss in both cases?
    <br>对于SVM来讲, 只要正确类的分数比别的类至少大一个$\Delta$的margin, 那么这一项$L_i$就会变为0, 不在对loss产生贡献. 但是对于softmax来讲, $L_i$始终不可能衰减为零. <br>In other words, the Softmax classifier is never fully happy with the scores it produces: the correct class could always have a higher probability and the incorrect classes always a lower probability and the loss would always get better. However, the SVM is happy once the margins are satisfied and it does not micromanage the exact scores beyond this constraint.

!!!question 
    Suppose that we found a W such that L = 0. Is this W unique?
    <br>No! 2W also has $L=0$.

## Regularization

### Intro 

> Suppose that we have a dataset and a set of parameters W that correctly classify every example (i.e. all scores are so that all the margins are met, and $L_i=0$
 for all i). The issue is that this set of W is not necessarily unique: there might be many similar W that correctly classify the examples. One easy way to see this is that if some parameters W correctly classify all examples (so loss is zero for each example), then any multiple of these parameters $\lambda W$
 where $\lambda >1$
 will also give zero loss because this transformation uniformly stretches all score magnitudes and hence also their absolute differences. For example, if the difference in scores between a correct class and a nearest incorrect class was 15, then multiplying all elements of W by 2 would make the new difference 30.

 其实上面这段引入就是想说明我们需要在存在多个W矩阵的时候采用某种方法避免过拟合.

![linear](./images/Lec03/lec03%20(7).png){: width="300px" .center}

### Definition
Regularization即为给损失函数增加一项regularization penalty来防止过拟合:

 $$
 L = \frac{1}{N}\sum_{i=1}^N L_i(f(x_i, W), y_i) + \lambda R(W)
 $$

 + Data loss: Model predictions should match training data
 + Regularization: Model should be “simple”, so it works on test data
 + $\lambda$ is a hyperparameter

### Types

+ L2 regularization: $R(W) = \sum_k \sum_l W_{k,l}^2$
+ L1 regularization: $R(W) = \sum_k \sum_l |W_{k,l}|$
+ Elastic net: $\lambda R(W) = \lambda \alpha R_1(W) + (1-\lambda) R_2(W)$

> 直觉上来讲, L2 regularization倾向于让数值在W矩阵内分布更均匀, 而不是某几个元素特别大.

## Optimization

### Strategy: Follow the slope

+ In multiple dimensions, the gradient is the vector of (partial derivatives) along each dimension.
+ In multiple dimensions, the gradient is the vector of (partial derivatives) along
each dimension.
+ The direction of steepest descent is the negative gradient.

### Calculation of the Gradient

Note that in the mathematical formulation the gradient is defined in the limit as h goes towards zero, but in practice it is often sufficient to use a very small value (such as 1e-5 as seen in the example). Ideally, you want to use the smallest step size that does not lead to numerical issues. Additionally, in practice it often works better to compute the numeric gradient using the **centered difference formula**: $[f(x+h)−f(x−h)]/2h$. 

![linear](./images/Lec03/lec03%20(5).png){: width="600px" .center}

This, however, has a problem of efficiency. The loss is just a function of W, so we use calculus to compute an analytic gradient, and plug the current W in to get a numeric gradient.

In summary:

+ Numerical gradient: approximate, slow, easy to write
+ Analytic gradient: exact, fast, error-prone(容易出错)

> In practice: Always use analytic gradient, but check implementation with numerical gradient. This is called a gradient check.

### Gradient Descent

```python
while True:
    weights_grad = evaluate_gradient(loss_fun, data, weights)
    weights += -step_size * weights_grad
```

#### Stochastic Gradient Descent (SGD)
$$
    L(W) = \frac{1}{N}\sum_{i=1}^N L_i(f(x_i, W), y_i) + \lambda R(W) \\
$$

$$
    \nabla_W L(W) = \frac{1}{N}\sum_{i=1}^N \nabla_W L_i(f(x_i, W), y_i) + \lambda \nabla_W R(W) \\
$$

这两个式子是梯度下降的表达式, 但是当N很大的时候, 计算量会非常大. 所以我们可以使用大小为32/64/128的**mini-batch**来代替N, 一定程度上优化计算量.

```python
while True:
    data_batch = sample_training_data(data, 256)
    weights_grad = evaluate_gradient(loss_fun, data_batch, weights)
    weights += -step_size * weights_grad
```

#### Step Size(Learning Rate)

The gradient tells us the direction in which the function has the steepest rate of increase, but it does not tell us how far along this direction we should step. Choosing the step size (also called the learning rate) will become one of the most important (and most headache-inducing) hyperparameter settings in training a neural network.

!!!info 
    A [website](http://vision.stanford.edu/teaching/cs231n-demos/linear-classify/) to visualize the Linear Classification Loss.

## Image Features

### Color Histogram

![linear](./images/Lec03/lec03%20(4).png){: width="600px" .center}

### Histogram of Oriented Gradients (HoG)

![linear](./images/Lec03/lec03%20(3).png){: width="600px" .center}

### Bag of Words

![linear](./images/Lec03/lec03%20(2).png){: width="600px" .center}