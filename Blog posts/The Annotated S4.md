---
lien: https://srush.github.io/annotated-s4/
---
A state-space representation is a mathematical model of a physical system specified as a set of input, output, and [variables](https://en.wikipedia.org/wiki/Variable_(mathematics) "Variable (mathematics)") related by first-order [differential equations](https://en.wikipedia.org/wiki/Differential_equation "Differential equation") or [difference equations](https://en.wikipedia.org/wiki/Difference_equation "Difference equation").
Building on top of that, a state space model (SSM) is model which maps a 1-D input signal $u(t)$ to an $N$-D latent state $x(t)$ before projecting to a 1-D output signal $y(t)$.

Based on the type of system to represent, the model is defined as:



|           System Type            |                                             SSM                                              |
| :------------------------------: | :------------------------------------------------------------------------------------------: |
|    Continuous time-invariant     |       $$\begin{align}\dot{x}(t) &= Ax(t) + Bu(t) \\y(t) &= Cx(t) + Du(t) \end{align}$$       |
|      Continous time-variant      | $$\begin{align}\dot{x}(t) &= A(t)x(t) + B(t)u(t) \\y(t) &= C(t)x(t) + D(t)u(t) \end{align}$$ |
| Explicit discrete time-invariant |         $$\begin{align}x(k+1) &= Ax(k) + Bu(k) \\y(k) &= Cx(k) + Du(k) \end{align}$$         |
|  Explicit discrete time-variant  |   $$\begin{align}x(k+1) &= A(k)x(k) + B(k)u(k) \\y(k) &= C(k)x(k) + D(k)u(k) \end{align}$$   |
The system can also be a *Laplace domain of continuous time-invariant* or *Z-domain of discrete time-variant*.
