# Lecture 3 & 4: Transformation

## Homogeneous Coordinates

> Translation cannot be represented by linear transformation.

$$
\begin{pmatrix}
x^{'} \\
y^{'}
\end{pmatrix}
= 
\begin{pmatrix}
a & b \\
c & d
\end{pmatrix}
\begin{pmatrix}
x \\
y
\end{pmatrix}
+
\begin{pmatrix}
t_x \\
t_y
\end{pmatrix}
$$

So we add a third coordinate, which is w-coordinate.

### Matrix Representation

+ 2D point: $(x, y, 1)^T$

$$
\begin{pmatrix}
x^{'} \\
y^{'} \\
w^{'} \\
\end{pmatrix}
=
\begin{pmatrix}
1 & 0 & t_x \\
0 & 1 & t_y \\
0 & 0 & 1
\end{pmatrix}
\begin{pmatrix}
x \\
y \\
1
\end{pmatrix}
$$

+ 2D vector: $(x, y, 0)^T$

$$
\begin{pmatrix}
x \\
y \\
0 \\
\end{pmatrix}
=
\begin{pmatrix}
1 & 0 & t_x \\
0 & 1 & t_y \\
0 & 0 & 1
\end{pmatrix}
\begin{pmatrix}
x \\
y \\
0
\end{pmatrix}
$$

所以我们可以看到, 之所以将vector的w-coordinate设为0, 是因为vector具有平移不变性.


### Valid Operation

+ point + vector = point
+ vector + vector = vector
+ point - point = vector
+ point + point = undefined(invalid)

!!!note
    In homogeneous coordinates:
    $$
    (x, y, w)^T
    \text{is the same 2D point as}
    (x/w, y/w, 1)^T
    \text{, }
    w \neq 0
    $$

### Affine map

> Affine map = linear map + translation

Using homogeneous coordinates, we can represent affine map as a matrix multiplication:

$$
\begin{pmatrix}
x^{'} \\
y^{'} \\
1
\end{pmatrix}
=
\begin{pmatrix}
a & b & t_x \\
c & d & t_y \\
0 & 0 & 1
\end{pmatrix}
\begin{pmatrix}
x \\
y \\
1
\end{pmatrix}
$$

### Different 2D Transformations

![linear](../images/games101_1/1%20(6).png){: width="400px" .center}

## 3D Transformations

Same as 2D case:

![linear](../images/games101_1/1%20(5).png){: width="600px" .center}

![linear](../images/games101_1/1%20(4).png){: width="600px" .center}

> 这种矩阵表示的affine map是先进行linear map, 再进行translation的.

Scale and translation are basically the same as in 2D case, but rotation is much complicated:

![linear](../images/games101_1/1%20(3).png){: width="600px" .center}

![linear](../images/games101_1/1%20(2).png){: width="600px" .center}

!!!success
    Actually, we can represent a random 3D representation by composing $R_x, R_y, R_z$:

    $$
    R(\vec{n}, \alpha) = \cos(\alpha)I + (1-\cos(\alpha))\vec{n}\vec{n}^T + \sin(\alpha)N
    $$

    where 

    $$
    N = \begin{pmatrix}
        0 & -n_z & n_y \\
        n_z & 0 & -n_x \\
        -n_y & n_x & 0 \\
    \end{pmatrix}
    $$

    This formula is called Rodrigues' rotation formula.



## Viewing Transformations

Think about how to take a photo:

+ Find a good place to arrange people (model transformation)
+ Find a good place to put the camera (view transformation)
+ Cheese! (projection transformation)

### View/Camera Transformation

![linear](../images/games101_1/1%20(17).png){: width="600px" .center}

If both the camera and the object move together, the photo will be the same. We assume that the camera is fixed and all the objects can move. So we need to transform our camera to :

+ The origin, up at Y, look at -Z.
+ And transform the objects along with the camera.

#### Matrix Representation

![linear](../images/games101_1/1%20(8).png){: width="600px" .center}

![linear](../images/games101_1/1%20(9).png){: width="600px" .center}

> 因为旋转矩阵是正交矩阵, 所以其逆矩阵即为其转置.

### Projection Transformation

![linear](../images/games101_1/1%20(11).png){: width="600px" .center}

Projection in Computer Graphics

+ 3D to 2D
+ Orthographic projection
+ Perspective projection

#### Orthographic Projection

In general, we want to map a cuboid $[l, r] \times [b, t] \times [f, n]$ to the canonical cube $[-1, 1]^3$.

![linear](../images/games101_1/1%20(13).png){: width="600px" .center}

+ Center cuboid by translating
+ Scale into canonical cube

#### Perspective Projection

+ More common in Computer Graphics, art, visual systems.
+ Further objects are smaller.
+ Parallel lines not parallel, but converge to vanishing point.

![linear](../images/games101_1/1%20(14).png){: width="300px" .center}

How do we do perspective projection?

+ First squish the frustum into a cuboid(n->n, f->f)
+ Do orthographic projection

![linear](../images/games101_1/1%20(15).png){: width="600px" .center}

> 第一步对应一个$M_1$矩阵, 表示从frustum到cuboid的变换. 第二部对应前面Orthographic Projection中的矩阵.

![linear](../images/games101_1/1%20(16).png){: width="600px" .center}

对于任意一点$(x, y, z)$, 变换后的x, y坐标都可以通过相似三角形的得到, 但是变换后的z坐标未知, 所以我们可以得到:

$$
\begin{pmatrix}
x \\
y \\
z \\
1 \\
\end{pmatrix}
=>
\begin{pmatrix}
nx/z \\
ny/z \\
unknown \\
1 \\
\end{pmatrix}
=>
\begin{pmatrix}
nx \\
ny \\
still\ unknown \\
z \\
\end{pmatrix}
$$

有了这个关系, 其实我们已经可以填出$M_1$的部分元素:

$$
M_1 =
\begin{pmatrix}
n & 0 & 0 & 0 \\
0 & n & 0 & 0 \\
? & ? & ? & ? \\
0 & 0 & 1 & 0 \\
\end{pmatrix}
$$

为了得到第三行的各个元素, 我们需要考虑还有没有别的限制条件:

+ Any point on the near plane will not change.

$$
\begin{pmatrix}
x \\
y \\
n \\
1 \\
\end{pmatrix}
=>
\begin{pmatrix}
nx/n \\
ny/n \\
n \\
1 \\
\end{pmatrix}
=>
\begin{pmatrix}
nx \\
ny \\
n^2 \\
n \\
\end{pmatrix}
$$

单独看第三行:

$$
(M_{31}, M_{32}, M_{33}, M_{34})
\begin{pmatrix}
x \\
y \\
n \\
1 \\
\end{pmatrix}
= n^2
$$

$n$ has nothing to do with $x$ and $y$, so $M_{31}$ and $M_{32}$ should be zero.


+ Any point on the far plane will not change.

$$
(M_{31}, M_{32}, M_{33}, M_{34})
\begin{pmatrix}
x \\
y \\
n \\
1 \\
\end{pmatrix}
= f^2
$$

Solve these two equations:

$$
M_{33}n + M_{34} = n^2 \\
$$

$$
M_{33}f + M_{34} = f^2
$$

We have:

$$
M_{33} = n+f
$$

$$
M_{34} = -nf
$$

Finally, we get our $M_1$ matrix:

$$
M_1 =
\begin{pmatrix}
n & 0 & 0 & 0 \\
0 & n & 0 & 0 \\
0 & 0 & n+f & -nf \\
0 & 0 & 1 & 0 \\
\end{pmatrix}
$$

What's next?

+ Do orthographic projection($M_2$) to finish.
+ $M_{persp}$ = $M_2 \cdot M_1$

!!!question
    对于一个点$(x, y, z)$, 经过透视投影后, 变换后的坐标为$(nx/z, ny/z, z, 1)$, 那么变换后的z坐标是什么? 相比于原来的z坐标有什么变化?

    $$
    \begin{pmatrix}
    n & 0 & 0 & 0 \\
    0 & n & 0 & 0 \\
    0 & 0 & n+f & -nf \\
    0 & 0 & 1 & 0 \\
    \end{pmatrix}
    \begin{pmatrix}
    x \\
    y \\
    z \\
    1 \\
    \end{pmatrix}
    =
    \begin{pmatrix}
    nx \\
    ny \\
    (n+f)z - nf \\
    z \\
    \end{pmatrix}
    $$

    我们只需要比较$(n+f)z - nf$与$z^2$的关系即可, 移项并分解因式可得:

    $$
    (n+f)z - nf > z^2
    $$
    但是因为$0>n>z>f$, 所以变换后的z坐标绝对值变小, 数值变大.

Now, what's near plane's l, r, b, t then?

Sometimes people prefer: vertical **field-of-view (fovY)** and aspect ratio(assume symmetry i. e. I = -r, b = -t)

![linear](../images/games101_2/1%20(1).png){: width="500px" .center}

![linear](../images/games101_2/1%20(2).png){: width="500px" .center}

#### MVP

+ Model transformation (placing objects)
+ View transformation (placing camera)
+ Projection transformation
    + Orthographic projection (cuboid to "canonical" cube $[-1, 1]^3$)
    - Perspective projection (frustum to "canonical" cube)


