The goal of this paradigm is to train a **model** which describes a relationship between data it takes as **input** and its **output**. 
It can simply be defined as the following function: $y =f(x)$, where $y$ is the predicted data and $x$ the input one.

A model is also defined by a set of **parameters** $\phi$, which determine the relation between $x$ and $y$. Hence, the previous can be rewritten as:
$$
y = f(x, \phi)
$$

## Learning process
Training a model means finding the suitable set of parameters $\phi$
which describe at best the relation between $x$ and $y$. This is done using a **training dataset** of $I$ pairs of input and output examples $\{x_i,y_i\}$.
The **loss** $L$ quantifies the degree of mismatch in this mapping and can be treated as a function $L(\phi)$.

When training a model, we are seeking $\hat{\phi}$ the set of parameters that minimize this loss function.
$$
\hat{\phi} = \underset{\phi}{\text{argmin}} \left[L(\phi) \right]
$$
## 1D linear regression case
In the case of a 1D linear regression, the model is defined as:
$$
\begin{align*}
y &= f(x, \phi)\\
&= \phi_{0}+ \phi_1x
\end{align*}
$$

The **least square loss** can be used as a loss function in the case of linear regresssion:
$$
\begin{align*}
L(\phi) &= \sum_{i=1}^{I} (f(x_{i,}\phi) - y_i)^2\\
&= \sum_{i=1}^{I}(\phi_{0}+ \phi_{1} x_{i}- y_i)^2
\end{align*}
$$

Hence, the optimal set of parameters that we are looking for is defined as:
$$
\hat{\phi} = \underset{\phi}{\text{argmin}}[\sum_{i=1}^{I}(\phi_{0}+ \phi_{1} x_{i}- y_i)^2]
$$

## Notes
- Even though the terms *loss and cost functions* are used interchangeably, a loss function is the **individual** term associated with **a data point** and the cost function is the **overall** quantity that is minimized.
- Models defined as $y = f(x, \phi)$ are *discriminative* while *generative* ones are defined as $x = g(y, \phi)$ where real-world measurements $x$ are computed as a function of the output $y$.
- When computing partial derivatives like $\frac{\partial{L}}{\partial{\phi_0}}$, other terms in the expression are treated as **constants**.