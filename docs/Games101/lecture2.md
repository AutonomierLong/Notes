# Lecture 2: Review of Linear Algebra

> 这节课很简单, 简单记录几点.

## Cross Product

$$
|| \vec{a} \times \vec{b} || = || \vec{a} || \cdot || \vec{b} || \cdot \sin \theta
$$

$$
\vec{a} \times \vec{b} = \vec{b} \times \vec{a}
$$

+ Cross product is orthogonal to two initial vectors.

+ Direction determined by right-hand rule.

+ Useful in constructing coordinate system(默认右手系)

$$
\vec{x} \times \vec{y} = \vec{z}
$$

$$
\vec{y} \times \vec{z} = \vec{x}
$$

$$
\vec{z} \times \vec{x} = \vec{y}
$$

+ 矩阵形式:

![linear](../images/games101_1/1%20(1).png){: width="600px" .center}

+ Determin left/right, inside/outside.

![linear](../images/games101_1/1%20(7).png){: width="200px" .center}

> 比如对于这个三角形, 我们想判断点P是否在三角形内, 我们可以分别计算AP & AB, BP & BC 和 CA & CP的叉积, 如果结果都大于0, 则P在三角形内.



