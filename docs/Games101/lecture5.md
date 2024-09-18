# Lecture 5 & 6: Rasterization

## Basic Concepts

>After the process of MVP, we will we do to a canonical cube? We need to project the cube to the screen, and then rasterize it.

+ What is a screen?
    + An array of pixels
    + Size of the array: resolution
+ Raster = Screen in German
    + Rasterize = drawing onto the screen
+ Pixel(short for picture element)
    + For now: A pixel is a little square with uniform color
    + Color is a mixture of (red, green , blue)

![linear](../images/games101_2/1%20(3).png){: width="600px" .center}

![linear](../images/games101_2/1%20(4).png){: width="600px" .center}

## Rasterization

| Polygon Meshes | Triangle Meshes |
| -------------- | --------------- |
| ![linear](../images/games101_2/1%20(5).png){: width="600px" .center} | ![linear](../images/games101_2/1%20(6).png){: width="500px" .center} |



### Triangles

![linear](../images/games101_2/1%20(7).png){: width="600px" .center}

![linear](../images/games101_2/1%20(8).png){: width="600px" .center}

> 假设现在有一个三角形已经投影到屏幕上了, 如何根据其三个顶点的位置确定给哪些pixel上色.

#### Sampling

![linear](../images/games101_2/1%20(9).png){: width="300px" .center}

我们可以根据一个像素的中心是否在三角形内部来判断该像素块是否着色.

> 判断一个点是否inside三角形可以使用lecture2中讲过的三次Corss Product的方法.

???warning "Real Cases"
    真实的应用中, 像素块不一定是方形的:
    ![linear](../images/games101_2/1%20(10).png){: width="600px" .center}

#### Aliasing(Jaggies)

Well, if we display the results of the above sampling, we will get something like this:

![linear](../images/games101_2/1%20(11).png){: width="300px" .center}

> It's far from being a perfect triangle.

Or like this:

![linear](../images/games101_2/1%20(12).png){: width="600px" .center}

### Sampling Artifects

> Artifects due to sampling -- "Aliasing".

+ Jaggies -- sampling in space   
+ Moire -- undersampling images

![linear](../images/games101_2/1%20(13).png){: width="600px" .center}

+ Wagon Wheel Effect -- sampling in time

![linear](../images/games101_2/1%20(14).png){: width="600px" .center}

> 顺时针旋转的轮盘看起来在倒着转(人眼sample的频率较低).

!!!note "Behind all these Aliasing Artifects"
    Signals are changing too fast (high frequency), but sampled too slowly.

### Anti-Aliasing Idea: Blurring(Pre-Filtering) Before Sampling

![linear](../images/games101_2/1%20(15).png){: width="600px" .center}

> 我们不采用原来的非0即1的二元着色, 而是在blur之后填充上渐变的颜色.

But why does this work?

#### Frequency Domain

![linear](../images/games101_2/1%20(16).png){: width="600px" .center}

通过Fourier变换, 我们可以将图像变为频率不同的波段, 低频部分对应着图像中连续变换的部分, 而高频部分对应着图像中突变的部分(边缘). 因为我们的sample 频率较低, 所以高频部分很难采样出真实的结果, 从而导致采样结果出现Aliasing. 如果我们使用blur等手段过滤掉图像的高频部分, 我们的采样就可以较好地还原真实图形, 从而的得到较好的结果.

Filtering = Getting rid of certain frequencies

#### Anti aliasing by Computing Average Pixel Value

In rasterizing one triangle, the average value inside a pixel area of f(x,y) = inside(triangle,x,y) is equal to the area of the pixel covered by the triangle.

![linear](../images/games101_2/1%20(17).png){: width="600px" .center}

#### Anti aliasing By Supersampling(MSAA)

> 根据刚才的想法, 我们希望计算出一个像素内部该三角形覆盖的面积占像素面积的比例, 但实际上这是一个较难求解的连续问题, 我们仍然使用离散化的手段来处理该问题.

Approximate the effect of the 1-pixel box filter by sampling multiple locations within a pixel and averaging their values:

![linear](../images/games101_2/1%20(18).png){: width="200px" .center}

> 比如说上面这张图, 我们将一个像素进一步细分成16个区域, 每个区域对应一个中心点, 每个分块仍然使用之前的方法判断是否在三角形内部, 最后统计在三角形内部的中心点的个数, 除以16, 再乘上三角形的颜色作为该像素最终的颜色.

Steps:

1. Take N*N samples within each pixel.
2. Average the N*N samples inside each pixel.

![linear](../images/games101_2/1%20(19).png){: width="600px" .center}

!!!info "Anti aliasing today"
    我们将每个像素块进一步细分, 带来的drawback必然是计算复杂性提升, 如今的state of the art会有一些更加高效的在每个pixel中分区的方法, 什么临近的pixels还可以共用某些小分区.
    ![linear](../images/games101_2/1%20(20).png){: width="600px" .center}


## Depth Control

> 我们讨论了如何将一个三角形表征在二维的离散的像素点上, 但实际上一个真实的图形会被分割成很多不在同一平面上的三角形, 我们需要将这些三角形全部投影到二维像素平面上. 于是我们就需要确定三角形的遮挡关系.

### Painter's Algorithm

> Inspired by how painters paint, we just paint from back to front, overwrite in the framebuffer.

![linear](../images/games101_3/1%20(1).png){: width="400px" .center}

对于这样一张油画来讲, 我们先画远山, 再画草地(会遮挡一部分远山), 最后画树木(会遮挡一部分远山和草地).

这样的算法需要我们想各个三角形按照深度排序, $\mathcal{O(NlogN)}$.

!!!failure
    ![linear](../images/games101_3/1%20(2).png){: width="400px" .center}
    This algorithm cannot resolve the situation in the above piceture.

### Z-Buffer

> This is the algorithm that eventually won.

+ Store current minimum depth(z-value) for each pixel in the framebuffer.
+ Needs an additional buffer for depth values.
    + frame buffer stores color values
    + depth buffer stores depth values

!!!warning
    For simplicity, we suppose $z$ is always positive.
    smaller z -> closer, larger z -> farther

!!!example
    ![linear](../images/games101_3/1%20(3).png){: width="600px" .center}
    R代表无穷大, 表示未处理之前像素对应的depth为无穷大.

Suppose that each triangle contains a limited number of pixels, the time complexity of z-buffering will be $\mathcal{O(N)}$.

!!!tip
    当然, 如果我们要使用MSAA等方法, 即在一个像素内设置多个采样点, 则需要对每个采样点维护一个depth和color的buffer.



