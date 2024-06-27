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
## The linear regression case
