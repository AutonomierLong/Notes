# Lecture 14: Reinforcement Learning

## What is reinforcement learning?

![linear](./images/Lec14/1%20(2).png){: width="600px" .center}

在$state_i$下采取一个action $a_t$, 获得reward$r_t$, 产生下一个状态$state_{t+1}$.

### Cart-Pole Problem

![linear](./images/Lec14/1%20(3).png){: width="600px" .center}

### Robot Locomotion

![linear](./images/Lec14/1%20(4).png){: width="600px" .center}

### Atari Games

![linear](./images/Lec14/1%20(5).png){: width="600px" .center}

### Go

![linear](./images/Lec14/1%20(6).png){: width="600px" .center}

## Markov Decision Process

+ Mathematical formulation of the RL problem.
+ **Markov Property:** Current state completely characterizes the state of the world.

![linear](./images/Lec14/1%20(7).png){: width="600px" .center}

![linear](./images/Lec14/1%20(8).png){: width="600px" .center}

???example "A simple MDP: Grid World"
    ![linear](./images/Lec14/1%20(9).png){: width="600px" .center}
    ![linear](./images/Lec14/1%20(10).png){: width="600px" .center}

### The optimal policy $\pi^{*}$

![linear](./images/Lec14/1%20(11).png){: width="600px" .center}

### Value Function and Q-value Function

![linear](./images/Lec14/1%20(12).png){: width="600px" .center}

### Bellman Equation

![linear](./images/Lec14/1%20(13).png){: width="600px" .center}

## Solving for the optimal policy: Q-learning

Q-learning use a function approximator to estimate the action-value function:

$$
Q(s, a; \theta) \approx Q^*(s, a)
$$

where $\theta$ is the function parameters(weights).

If the function approximator is a deep neural network => deep 1-learning.

![linear](./images/Lec14/1%20(14).png){: width="600px" .center}

### Case Study: Playing Atari Games

+ Objective: Complete the game with highest score.
+ State: Raw pixel inputs of the game state.
+ Action: Game controls, left, right...
+ Reward: Score increase/decrease at each time step.

#### Q-network Architecture

![linear](./images/Lec14/1%20(15).png){: width="600px" .center}

最终生成四个值, 分别为上下左右的Q value. 我们使用最近的四帧来预测Q value.

> Number of actions between 4-18 depending on Atari game.

#### Experience Replay

![linear](./images/Lec14/1%20(16).png){: width="600px" .center}

![linear](./images/Lec14/1%20(17).png){: width="600px" .center}

## Policy Gradients

What is a problem with Q-learning?

The Q-function can be very complicated!

Example: a robot grasping an object has a very high-dimensional state => hard to learn exact value of every (state, action) pair. But the policy can be much simpler: just close your hand. Can we learn a policy directly, e.g. finding the best policy from a collection of
policies?

![linear](./images/Lec14/1%20(18).png){: width="600px" .center}

这里定义了对于一个Policy的价值, 即累计reward的期望.

![linear](./images/Lec14/1%20(19).png){: width="600px" .center}

一个Policy累计reward的期望其实就是对于所有轨迹求和依据概率取平均.

跳过复杂的数学推导, 得到下面的结论:

![linear](./images/Lec14/1%20(20).png){: width="600px" .center}

力争将能获得较大reward的action的概率拉大, 反之则减小. 我们期望在多次训练之后能将那些较为优秀的action凸显出来.

但是可能会在学习过程中遇到高方差问题。这意味着估计的值在不同的训练迭代之间波动很大，导致学习过程不稳定。同时, 在强化学习中，特别是在延迟奖励的情况下，确定某个特定动作对未来奖励的贡献是非常困难的。例如，某个动作可能会在很久之后才产生明显的效果。这使得准确地归因变得复杂，进一步增加了估计器的方差。高方差会导致学习过程产生严重的不稳定性, 所以需要采取一定手段减小方差.

![linear](./images/Lec14/1%20(21).png){: width="600px" .center}

![linear](./images/Lec14/1%20(22).png){: width="600px" .center}

![linear](./images/Lec14/1%20(23).png){: width="600px" .center}

![linear](./images/Lec14/1%20(1).png){: width="600px" .center}

