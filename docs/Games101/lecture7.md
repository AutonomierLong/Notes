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

综合Lambert's Law和Light Falloff, 我们可以得到漫反射下的着色公式(Diffusion Term):

$$
L_d = k_d(I/r^2)max(0,\hat{n}\cdot \hat{l})
$$

![linear](../images/games101_3/1%20(11).png){: width="600px" .center}

> $k_d$的存在是因为我们假设表面会吸收一些能量, 假如$k_d$为1, 代表表面完全没有吸收能量, 为0则代表吸收了所有能量.

!!!note "The impact of $k_d$"
    ![linear](../images/games101_3/1%20(12).png){: width="600px" .center}
    可以看到, $k_d$越大, 反射的光强越大, 即表面越亮.

#### Specular Reflection

> Intensity depends on view direction.

Bright near mirror reflection direction. 镜面反射的特点是存在着一个反射方向, 只有在该反射方向附近的观测方向才能看到高光.

![linear](../images/games101_4/1%20(1).png){: width="600px" .center}

我们一般使用half vector来计算观测方向与入射方向的角平分线方向向量.

> 计算half vector与法线的点积而不是计算反射光线和观测方向的点积, 这是因为前者计算上更加简便.

得到Specular Term如下:

$$
L_s = k_s(I/r^2)max(0,\hat{h}\cdot \hat{n})^p
$$

> 同理, 这里的$k_s$也是表示吸收能量的比例.

!!!note "why is there a power $p$?"
    ![linear](../images/games101_4/1%20(3).png){: width="600px" .center}
    我们希望只有在观测方向与反射方向非常接近的时候, 光的强度才大, 而当观测方向与反射方向相差较大的时候, 光的强度就非常小. 但是如果不给$cos\alpha$项加上指数的话, 可以看到就算是相差了45度, 观测到的亮度还是较大. 所以我们需要给$cos\alpha$项加上一个指数, 使得当角度相差较大的时候, 光的强度迅速下降.

???tip
    ![linear](../images/games101_4/1%20(4).png){: width="600px" .center}
    将Diffusion Term和Specular Term结合起来, 我们观察$k_s$和$p$变化对渲染效果的影响. 可以看到随着$p$增大, 高亮更加集中.

#### Ambient Term

> 对环境光的影响, Blinn-Phong Model考虑的非常简单, 即给所有地方都加上一个相同的常量光强, 不过这个光强要乘以一个系数$k_a$.

![linear](../images/games101_4/1%20(5).png){: width="600px" .center}


综合上述三个Term, 我们可以得到Blinn-Phong Reflective Model的公式:

$$
L = L_a + L_d + L_s
$$

$$
L = k_d(I/r^2)max(0,\hat{n}\cdot \hat{l}) + k_s(I/r^2)max(0,\hat{h}\cdot \hat{n})^p + k_aL_a
$$

可以大致渲染出这样的结果:

![linear](../images/games101_4/1%20(6).png){: width="600px" .center}

### Shading Frequencies

> 我们刚刚考虑的都是第一个点进行着色, 但是实际的图形是连续的, 我们有多种手段将其变为离散的点, 而后在=再着色.

![linear](../images/games101_4/1%20(7).png){: width="600px" .center}

#### Flat Shading

![linear](../images/games101_4/1%20(8).png){: width="400px" .center}

> 每个三角形的小切面当做一个点, 然后给整个切面shade相同的颜色.

#### Gouraud Shading

![linear](../images/games101_4/1%20(9).png){: width="600px" .center}

> 对每个顶点着色, 然后三角形上的颜色由各个顶点之间插值得到.

#### Phong Shading

![linear](../images/games101_4/1%20(10).png){: width="600px" .center}

> 插值的到三角形区域上的所有法向量, 然后针对每个像素着色.

#### Comparison

![linear](../images/games101_4/1%20(11).png){: width="500px" .center}

其实只要我们把这些三角形分割地足够细致, Flat 和 Phong Shading效果相差无几.


#### Per-Vertex Normal Vectors

+ Best to get vertex normals from the underlying geometry.

![linear](../images/games101_4/1%20(12).png){: width="200px" .center}

比如我们知道需要渲染的是个球体, 那么每个顶点的法向量自然是圆心到顶点的连线方向. 

+ Otherwise have to infer vertex normals from triangle faces.

![linear](../images/games101_4/1%20(13).png){: width="200px" .center}

最简单的做法就是对该顶点所在的几个三角形面的法向量求平均, 再归一化:

$$
N_v = \frac{\sum_i N_i}{||\sum_i N_i||}
$$

当然有可能有些三角形面比较大, 对顶点的法向量影响会相对较大, 所以也可以对几个法向量加权求平均.

#### Per-Pixel Normal Vectors

![linear](../images/games101_4/1%20(14).png){: width="400px" .center}

由几个顶点之间的normal vectors插值得到, 具体插值方法后文介绍.


### Texture Mapping

![linear](../images/games101_4/1%20(15).png){: width="500px" .center}

> 在Blinn-Phong模型中, 我们可以使用$k_d$, $k_s$, $k_a$来表示表面的纹理特征, 比如上面这张图, 球和地板都有着自己的纹理. 
> 需要注意的是, $k_d$, $k_s$, $k_a$都是三元向量, 表示三个颜色通道, shading公式后面所乘的可以理解为只是亮度.

假设, 设计师预先为我们设计了一份纹理图样(一张2D图像), 并且标注了3D模型上的每个三角形对应的2D图样上的三角形.

![linear](../images/games101_4/1%20(16).png){: width="500px" .center}

> Each triangle vertex is assigned a texture coordinate $(u, v)$, which is a 2D point in $[0,1] \times [0,1]$.

那么我们就可以根据着色点在3D模型上的位置, 查询到其在2D图样上的位置, 然后根据这个位置来查询纹理特征.

一张纹理图样可能在三维模型中反复出现, 所以我们希望纹理的设计是可以自然拼接的, 即左边与右边, 上边与下边.

![linear](../images/games101_4/1%20(17).png){: width="400px" .center}

