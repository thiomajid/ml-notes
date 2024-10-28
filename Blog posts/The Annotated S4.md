---
lien: https://srush.github.io/annotated-s4/
---
A state-space representation[^1] is a mathematical model of a physical system specified as a set of input, output, and [variables](https://en.wikipedia.org/wiki/Variable_(mathematics) "Variable (mathematics)") related by first-order [differential equations](https://en.wikipedia.org/wiki/Differential_equation "Differential equation") or [difference equations](https://en.wikipedia.org/wiki/Difference_equation "Difference equation").

![[Pasted image 20240720191114.png]]
>[!info]
>The figure above represents a set of linear state-space equations


An #SSM has the general form:
$$
\begin{align}
\dot{x}(t) &= Ax(t) + Bu(t) \\
y(t) &= Cx(t) + Du(t) \end{align}
$$

Where:
- $x(\cdot)$: is the *state vector*, $x(t) \in \mathbb{R}^n$
- $y(\cdot)$: is the *output vector*, $y(t) \in \mathbb{R}^q$
- $u(\cdot)$: is the *input (or control) vector*, $u(t) \in \mathbb{R}^p$
- $A(\cdot)$: is the *state (or system) matrix*, $\text{dim}[A(\cdot)] = n \times n$
- $B(\cdot)$: is the *input matrix*, $\text{dim}[B(\cdot)] = n \times p$
- $C(\cdot)$: is the *output matrix*, $\text{dim}[C(\cdot)] = q \times n$
- $D(\cdot)$: is the *feedthrough matrix*. In cases where the system model does not have a direct feedthrough, $D(\cdot)$ is the *zero matrix*, $\text{dim}[D(\cdot)] = q \times p$. 

Based on the type of system to represent, the model is defined as:

|           System Type            |                                             SSM                                              |
| :------------------------------: | :------------------------------------------------------------------------------------------: |
|    Continuous time-invariant     |       $$\begin{align}\dot{x}(t) &= Ax(t) + Bu(t) \\y(t) &= Cx(t) + Du(t) \end{align}$$       |
|      Continous time-variant      | $$\begin{align}\dot{x}(t) &= A(t)x(t) + B(t)u(t) \\y(t) &= C(t)x(t) + D(t)u(t) \end{align}$$ |
| Explicit discrete time-invariant |         $$\begin{align}x(k+1) &= Ax(k) + Bu(k) \\y(k) &= Cx(k) + Du(k) \end{align}$$         |
|  Explicit discrete time-variant  |   $$\begin{align}x(k+1) &= A(k)x(k) + B(k)u(k) \\y(k) &= C(k)x(k) + D(k)u(k) \end{align}$$   |

The system can also be a *Laplace domain of continuous time-invariant* or *Z-domain of discrete time-variant*.

## SSM model in Deep Learning
![[Pasted image 20240720191322.png]]
Building on top of that, a state space model (SSM) [^2] is model which maps a 1-D input signal $u(t)$ to an $N$-D latent state $x(t)$ before projecting to a 1-D output signal $y(t)$.

When training an SSM, the matrices $A, B, C, D$ are the learnable parameters using gradient descent. $D$ is sometimes omitted because the term $Du$ can be seen as a **skip connection**.

Given that an SSM model operates on a continuous signal $u(t)$
in order to be applied on a discrete input sequence, the SSM must be discretized by a *step size* $\Delta$. The inputs $u_k$ can be viewed  













[^1]: [State-space representation - Wikipedia](https://en.wikipedia.org/wiki/State-space_representation)
[^2]: [[2111.00396] Efficiently Modeling Long Sequences with Structured State Spaces (arxiv.org)](https://arxiv.org/abs/2111.00396)