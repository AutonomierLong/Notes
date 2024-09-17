# Lecture 7 & 8 & 9: Shading

What we've covered so far:

![linear](../images/games101_3/1%20(4).png){: width="600px" .center}

## Definition

The process of applying a material to an object.

## A simple Shading Model: Blinn-Phong Reflective Model

### Perceptual Observations

![linear](../images/games101_3/1%20(5).png){: width="500px" .center}

在我们观察一个场景的时候, 有些地方会产生高光的效果, 即光线基本呈现镜面反射的效果; 有些地方十分平滑, 即光线基本呈现漫反射的效果; 有些地方较暗, 不能被光源直接照射到, 是依靠别的地方的反射光照亮的.

> Shading 对应的是着色的过程, 不会考虑影子的问题.

![linear](../images/games101_3/1%20(6).png){: width="500px" .center}

### Model Basics

![linear](../images/games101_3/1%20(7).png){: width="600px" .center}

模型的输入有观测点相对于着色点的方向, 光线相对于着色点的方向, 着色点处平面的法线, 着色点处平面的一些其他信息(颜色, 亮度).

#### Diffuse Reflection

> Light is uniformly scattered in all directions, and surface color is the same for all viewing directions.

![linear](../images/games101_3/1%20(8).png){: width="300px" .center}

##### Lambert's Law

![linear](../images/games101_3/1%20(9).png){: width="600px" .center}

surface 吸收的光照强度与入射光线的角度有关, 与观测点的角度无关(因为漫反射各个方向观测color都一样).

##### Light Falloff

![linear](../images/games101_3/1%20(10).png){: width="300px" .center}

我们假设光沿着四面八方均匀传播, 由于能量守恒, 每一时刻的球壳上带有的能量都是一样的, 那么单位面积的光照强度就与球壳半径的平方(球壳面积)成反比.

综合Lambert's Law和Light Falloff, 我们可以得到漫反射下的着色公式:

$$
L_d = k_d(I/r^2)max(0,\hat{n}\cdot \hat{l})
$$

![linear](../images/games101_3/1%20(11).png){: width="600px" .center}

> $k_d$的存在是因为我们假设表面会吸收一些能量, 假如$k_d$为1, 代表表面完全没有吸收能量, 为0则代表吸收了所有能量.

!!!note "The impact of $k_d$"
    ![linear](../images/games101_3/1%20(12).png){: width="600px" .center}
    可以看到, $k_d$越大, 反射的光强越大, 即表面越亮.


