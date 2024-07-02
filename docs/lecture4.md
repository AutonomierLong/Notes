# Lecture 4: Backpropagation and Neural Networks

## Chain Rule

假设我们有两个函数 \( f \) 和 \( g \)，其中 \( f \) 是 \( g \) 的函数，即 \( f(g(x)) \)。链式求导法则告诉我们，复合函数 \( f(g(x)) \) 的导数可以表示为：

$$
\frac{d}{dx} f(g(x)) = f'(g(x)) \cdot g'(x) 
$$

其中，\( f'(g(x)) \) 是函数 \( f \) 对 \( g(x) \) 的导数，\( g'(x) \) 是函数 \( g \) 对 \( x \) 的导数。

也即:
$$
\frac{df}{dx} = \frac{df}{dg} \cdot \frac{dg}{dx}
$$

## Computational Graphs

把一个复杂的代数表达式简化成一个图结构, 每个node为一个operator, 接受一个或多个operand进行运算. 当每个variable有具体值的时候, 整个图中拓扑序最靠后的node的输出为该表达式的值.

!!!info "An Simple Example"
    ![linear](./images/Lec04/lec04%20(3).png){: width="600px" .center}

### Forward Pass

在真正利用computation graph时, 我们首先需要使用forward pass来计算出在每个variable给定时最终的Loss值. 我们只需要按照拓扑序对整张图进行依次计算即可. 对于某一个node, 接受下游的一个或多个数值, 对其进行该node对应的操作, 然后再将计算结果传递给上游.

### Backpropagation

在利用forward pass计算完各个node的对应值之后, 我们就可以按照整张图的逆拓扑序利用chain rule进行求导.

具体来讲, 因为每个node都是较为简单的函数(如+, -, *, /, sigmoid, max等), 我们可以预先确定这些node的关于输入变量的导数表达式, backpropagation时只需要将某个变量(a)的具体值代入对应偏导表达式, 乘上上游传来的导数值, 就能求得$\frac{\partial L}{\partial a}$.

!!!info "A more complicated example"
    ![linear](./images/Lec04/lec04%20(2).png){: width="600px" .center}
    可以看到, 我们只需要依次进行刚刚所叙述的forward pass和propagation即可完成Loss关于各个变量偏导数具体值的计算.
    ![linear](./images/Lec04/lec04%20(1).png){: width="600px" .center}
    另外, 我们可以选择不同granularity(粒度)的node, 比如将第一张图的几个node合并成一个sigmoid node, 这样可以通过构建稍微复杂一些的node来简化computation graph.

### Some Gates in Particular

> Note that here we use gate and node interchangeably.

![linear](./images/Lec04/lec04%20(8).png){: width="400px" .center}

+ add gate: gradient distributor
+ max gate: gradient router
+ mul gate: gradient switcher

### Gradients for Vectorized Code

**Jacobian Matrix**:

假设我们有一个从 $\mathbb{R}^n$ 映射到 $\mathbb{R}^m$ 的函数 $\mathbf{f} : \mathbb{R}^n \to \mathbb{R}^m$，其形式为：

\[
\mathbf{f}(\mathbf{x}) = \begin{bmatrix}
f_1(x_1, x_2, \ldots, x_n) \\
f_2(x_1, x_2, \ldots, x_n) \\
\vdots \\
f_m(x_1, x_2, \ldots, x_n)
\end{bmatrix}
\]

雅可比矩阵 $\mathbf{J}$ 是 $\mathbf{f}$ 对 $\mathbf{x}$ 的偏导数组成的矩阵，定义为：

\[
\mathbf{J}(\mathbf{x}) = \begin{bmatrix}
\frac{\partial f_1}{\partial x_1} & \frac{\partial f_1}{\partial x_2} & \cdots & \frac{\partial f_1}{\partial x_n} \\
\frac{\partial f_2}{\partial x_1} & \frac{\partial f_2}{\partial x_2} & \cdots & \frac{\partial f_2}{\partial x_n} \\
\vdots & \vdots & \ddots & \vdots \\
\frac{\partial f_m}{\partial x_1} & \frac{\partial f_m}{\partial x_2} & \cdots & \frac{\partial f_m}{\partial x_n}
\end{bmatrix}
\]

这里，$\frac{\partial f_i}{\partial x_j}$ 表示 $f_i$ 对 $x_j$ 的偏导数。

![linear](./images/Lec04/lec04%20(7).png){: width="600px" .center}


!!!success "A useful conclusion"
    如果我们有:
    $$
    y = f(AB) = f(c)
    $$
    其中 $A, B, C$ 都是矩阵，那么有如下结论:
    $$
    \frac{\partial f}{\partial A} = \frac{\partial f}{\partial C} \cdot B^T
    $$
    同理:
    $$
    \frac{\partial f}{\partial B} = A^T \cdot \frac{\partial f}{\partial C}
    $$

结合上面的结论我们看一个例子:
![linear](./images/Lec04/lec04%20(6).png){: width="600px" .center}

可以看到:
$$
    \frac{\partial f}{\partial q} = 2q
$$
$$
    \frac{\partial f}{\partial W} = \frac{\partial f}{\partial q} \cdot x^T
$$
$$
    \frac{\partial f}{\partial x} = W^T \cdot \frac{\partial f}{\partial q}\\
$$

!!!warning
    Always check: The gradient with respect to a variable should have the same shape as the variable.

### Modularized Implementation

#### Graph Object
```python
class ComputationalGraph
    # ...
    def forward(inputs):
        # 1. pass inputs to input gates
        # 2. forward the computational graph
        for gate in self.graph.nodes_topologically_sorted():
            gate.forward()
        return loss
    def backward():
        for gate in reversed(self.graph.nodes_topologically_sorted()):
            gate.backward()
        return inputs_gradients
```

#### Gate Object
```python
class MultiplyGate(object)
    def forward(x, y)
        z = x*y
        self.x = x
        self.y = y
        return z
    def backward(dz)
        dx = self.y * dz
        dy = self.x * dz
        return [dx, dy]
```

## Neural Networks

+ (Before) Linear score function: $f = Wx$
+ (Now) 2-layer Neural Network: $f = W_2max(0, W_1x)$
+ (Further) 3-layer Neural Network: $f = W_3max(0, W_2max(0, W_1x))$

![linear](./images/Lec04/lec04%20(5).png){: width="600px" .center}

在Linear Classifier里, 我们说W矩阵的每一行会学习到每种类别的模板, 最终生成10个类别相应的分数. 但是因为每种类别可能会出现多种情况, 所以单一模板会将这些情况平均地学习进去, 如有着两个头的马. 

在两层神经网络中, 我们将h层(hidden layer)设置成100个节点, 这样W1矩阵会相应增大, 那么每一行会学习到更加具体的特征. 比如, 新的W1的一行可能学习到了一个头偏向左边的马, 而梁另外有一行学习到的是头偏向右的马. 这些更为具体的特征, 再通过W2的加权求和, 最终才生成了10个类别的分数. 那么假如现在我们有三匹马, 头的朝向分别为左, 右和中间, 那么第一匹马在"头朝左"类别上分数很高, 第二匹在"头朝右"类别分数很高, 第三匹在二者都有一定的分数, 那么经过训练后的W2的加权求和, 三匹马都能在horse这一类上获得较高的score.

这里的$max$函数被称为non-linear 的 activation function, 对于神经网络来讲有多种选择:
![linear](./images/Lec04/lec04%20(4).png){: width="600px" .center}

```python title="Full implementation of training a 2-layer Neural Network"
import numpy as np
from numpy.random import randn

N, D_in, H, D_out = 64, 1000, 100, 10
x, y = randn(N, D_in), randn(N, D_out)
w1, w2 = randn(D_in, H), randn(H, D_out)
learning_rate = 1e-4

for t in range(2000):
    h = 1 / (1 + np.exp(-x.dot(w1))) # sigmoid
    y_pred = h.dot(w2)
    loss = np.square(y_pred - y).sum()
    print(t, loss)

    grad_y_pred = 2.0 * (y_pred - y)
    grad_w2  = h.T.dot(frad_y_pred)
    grad_h = grad_y_pred.dot(w2.T)
    grad_w1 = x.T.dot(grad_h * h * (1-h))
    # 上面这几行应当认为的是 S = hW2 = sigmoid(xW1)W2, 如果顺序反过来的话上面dot括号内外的顺序需要要反一下.

    w1 -= learning_rate * grad_w1
    w2 -= learning_rate * grad_w2
```


