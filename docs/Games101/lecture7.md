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


### Interpolation Across Triangles: Barycentric Coordinates

> 考虑这样的情景: 已知三角形三个顶点的某种属性, 我们希望得到三角形内部一个点的该属性. 我们需要使用重心坐标进行插值.

#### Barycentric Coordinates

![linear](../images/games101_5/1%20(1).png){: width="500px" .center}

> 如果任意一点满足$\alpha + \beta + \gamma = 1$, 那么这个点就在三角形所在的平面上, 同时, 如果$\alpha, \beta, \gamma$都大于0, 那么这个点就在三角形内部.

???tip "ways to calculate the barycentric coordinates"
    ![linear](../images/games101_5/1%20(2).png){: width="500px" .center}
    可以用顶点所对的面积比计算
    ![linear](../images/games101_5/1%20(3).png){: width="500px" .center}
    可以直接代公式

> 特别地, 重心对应的重心坐标为$(\frac{1}{3}, \frac{1}{3}, \frac{1}{3})$.

#### Interpolation

![linear](../images/games101_5/1%20(4).png){: width="500px" .center}

我们直接利用$\alpha$, $\beta$, $\gamma$三个参数对三个顶点的某个属性加权平均即可.

!!!warning "barycentric coordinates are not invariant under projection"
    重心坐标不具有投影不变性, 所以要注意计算重心坐标进行插值的时候, 一定要考虑清楚此时对应的三角形形态.
    比如, 我们想对一个三角形内部的点插值计算深度(depth), 那么此时就要求该点在三维空间三角形的重心坐标, 而不能求已经投影在二维平面之后的重心坐标.

## Texture Mapping

![linear](../images/games101_4/1%20(15).png){: width="500px" .center}

> 在Blinn-Phong模型中, 我们可以使用$k_d$来表示表面的纹理特征, 比如上面这张图, 球和地板都有着自己的纹理. 
> 需要注意的是, $k_d$是三元向量, 表示三个颜色通道, shading公式后面所乘的可以理解为只是亮度.

假设, 设计师预先为我们设计了一份纹理图样(一张2D图像), 并且标注了3D模型上的每个三角形对应的2D图样上的三角形.

![linear](../images/games101_4/1%20(16).png){: width="500px" .center}

> Each triangle vertex is assigned a texture coordinate $(u, v)$, which is a 2D point in $[0,1] \times [0,1]$.

那么我们就可以根据着色点在3D模型上的位置, 查询到其在2D图样上的位置, 然后根据这个位置来查询纹理特征.

一张纹理图样可能在三维模型中反复出现, 所以我们希望纹理的设计是可以自然拼接的, 即左边与右边, 上边与下边.

![linear](../images/games101_4/1%20(17).png){: width="400px" .center}

下面, 我们介绍几种将纹理映射用于shading的算法

### Simple Texture Mapping: Diffuse Color

![linear](../images/games101_5/1%20(5).png){: width="600px" .center}

这样的做法直接将图像上的点映射到纹理上, 采样, 获得$k_d$, 看上去很简单, 但是实际中会遇到很多问题.

### Texture too Small

> 想象一下, 如果我们希望渲染一面墙, 现在有一个墙纸的纹理, 但是墙的分辨率为$1024 \times 1024$, 而纹理的分辨率小得多, 那么将这样的纹理添加到墙上, 就会使得墙面看起来很模糊.

![linear](../images/games101_5/1%20(6).png){: width="600px" .center}

不过这个问题可以较好地通过插值解决, 最左侧的图像是不做处理, 直接采样得到的结果, 可以看出有很多锯齿, 中间和右边的图像分别使用双线性插值(考虑纹理上邻近的四个texel)和三次插值(考虑纹理上邻近的16个texel), 效果都不错, 但是三次插值的效果更好, 却处理复杂度更高.

### Texture too Large

![linear](../images/games101_5/1%20(7).png){: width="600px" .center}

> 为什么会出现上图的这种情况? 
> 因为一个screen上的pixel, 对应着纹理上一大片texel, 我们直接采样的话会取得那么一大片的中心texel, 自然不能很好地代表一大片texel的信息. 

![linear](../images/games101_5/1%20(8).png){: width="600px" .center}

#### Super Sampling

究其本质, 还是因为我们的采样频率太低, 导致一个采样点里面包含了太多信息. 所以我们可以将一个pixel细化成多个采样点, 最后pixel的颜色就是这些采样点的颜色的平均.

![linear](../images/games101_5/1%20(9).png){: width="600px" .center}

这样进行super sampling之后, 我们就可以得到更精细的纹理效果, 但是代价就是计算量更大.

#### Mipmap

> 我们在SuperSampling中做的其实就是去模拟一大片texel的平均值, 那么考虑有没有什么算法能够近似地在较快时间内查询到一片texel的平均值呢?
> Mipmap -- Allowing (fast, approx, square) range queries.

我们先要对纹理图样做一些预处理, 方便后续的快速区域查询.

![linear](../images/games101_5/1%20(10).png){: width="600px" .center}

![linear](../images/games101_5/1%20(11).png){: width="400px" .center}

我们将图像的分辨率不断降低, 直到变为一个pixel 而后将其堆叠成金字塔的形状.

???note "overhead"
    我们需要储存不同分辨率下的图样, 所以会有一定的overhead, 但其实这个overhead不大, 是一个级数求和问题(假设原始图像的大小为$S$):
    $$
    Total = \sum_{i=0}^{\inf} \frac{S}{2^i} = 2S
    $$

那么有了mipmap之后该如何获得一个pixel的图样信息呢?

![linear](../images/games101_5/1%20(12).png){: width="600px" .center}

我们找到该pixel邻近的几个pixel, 将他们全部变换到texel坐标, 计算彼此之间的距离, 取最大值, 将这个最大值作为边长, 构建一个正方形. 然后对这个正方形边长取$log$, 就可以知道这样大小的正方形对应着mipmap中哪一层, 而后在该层上采样即可近似求得这一片texel的平均值.

!!!note "Trilinear"
    ![linear](../images/games101_5/1%20(13).png){: width="600px" .center}
    实际上由于计算出来的$D$并非是一个整数, 所以我们也可以在texel上的相邻两层分别双线性插值, 然后在使用$D$将两层的结果线性插值的到最终的结果.

!!!fail "Shading Results"
    ![linear](../images/games101_5/1%20(14).png){: width="200px" .center}
    可以看到, 即使使用了三线性插值, 还是会出现一些瑕疵(overblur), 这是因为我们在mipmap中做了较多的近似, 导致结果不好. 
    比如, 我们假设了平均区域只能是正方形, 正方形长度取最大值等等.
    ![linear](../images/games101_5/1%20(15).png){: width="600px" .center}
    在实际的情况中, 一个pixel对应的texel区域多种多样, 可能是长方形, 斜着的图形等等.

### Anistotic Filtering(Ripmaps, EWA Filtering)

> 很自然的一个提升mipmap的想法就是将长方形区域也考虑进去, 这样就有了Ripmaps.

![linear](../images/games101_5/1%20(16).png){: width="600px" .center}

但是Ripmap的存储量要大于mipmap, 可以看到最后大约是原始图像存储量的四倍.

EWA filtering 是利用一系列大小不等的椭圆取近似图形, 可以解决不规则区域的问题, 精度更高, 但是显然计算复杂度和存储空间的需求更大.




