## The recurrent framework
Recurrent models follows the general formulation:
$$
\begin{align}
h_{t} &= Ah_{t-1} + Bx_{t}\\
y_{t} &= Ch_t
\end{align}
$$

- $A$: weighs the contribution of the system's state during its transition to a new state
- $B$ determines the extent to which the new input $x_t$ contributes to the new state
- $C$: computes the output the output at time step $t$ using the updated system state $h_t$.

## Long Short Term Memory
In an #LSTM layer, a cell is a computation unit with *three gates* and another component called *cell state*. The cell state is the cell's memory. The computation within an LSTM cell is expressed as:

$$
\begin{align}
i &= \sigma(W_{ii} x + b_{ii} + W_{hi} h + b_{hi}) \\
f &= \sigma(W_{if} x + b_{if} + W_{hf} h + b_{hf}) \\
g &= \tanh(W_{ig} x + b_{ig} + W_{hg} h + b_{hg}) \\
o &= \sigma(W_{io} x + b_{io} + W_{ho} h + b_{ho}) \\
c' &= f \odot c + i \odot g \\
h' &= o \odot \tanh(c')
\end{align}
$$
Where $\sigma$ is the sigmoid function and $\odot$ is the #Hadamard product.

### Cell state
The cell state is updated for each new input $x_t$. As a matter of fact, a cell is a *layer* in the LSTM and its state represents its *long term* memory meanwhile the hidden state is the *short term* memory.

## eXtended Long Short Term Memory
Using the formulation of a recurrent model, an LSTM can be expressed as:

$$
\begin{align}
c_{t} &= f_{t} \odot c_{t-1} + i_{t} \odot z_{t}\\
h_{t} &= o_{t} \odot \tilde{h_{t}}, &\tilde{h_{t}} &= \psi{(c_{t)}}\\
z_{t} &= \psi{(\tilde{z_{t})}},  &\tilde{z_{t}} &= w_{z}^{T}x_{t} + r_{z}h_{t-1} + b_{z} \\
i_{t}&= \sigma{(\tilde{i_{t})}}, &\tilde{i_{t}} &= w_{i}^{T}x_{t} + r_{i}h_{t-1} + b_{i} \\
f_{t}&= \sigma{(\tilde{f_{t})}}, &\tilde{f_{t}} &= w_{f}^{T}x_{t} + r_{f}h_{t-1} + b_{f} \\
o_{t}&= \sigma{(\tilde{o_{t})}}, &\tilde{o_{t}} &= w_{o}^{T}x_{t} + r_{o}h_{t-1} + b_{o} \\
\end{align}
$$

where $\psi$ is the #tanh function, $\sigma$ is the #sigmoid function and $A = f_{t}, B = i_{t}, C = o_{t}$.

The #xLSTM architecture drops the sigmoid activation for the gates and now uses #exponential gating instead. Doing so allows the LSTM to revise storage decisions. Knowing that the exponential function grows quickly, a #stabilizer state ($m_t$) has been introduced: $m_{t}= \text{max}(\log{(f_{t)}} + m_{t-1}, \log{(i_{t})})$

For normalization, a #normalizer state ($n_t$) is used.
The normalizer state that sums up the product of input gate times all future forget gates: $n_{t} = f_{t} \odot n_{t-1} + i_t$ 

Moreover, using a **matrix memory**, the storage capacity has been increased given that it is no longer a **scalar memory** that is updated.

>[!info]
>For more details on #sLSTM and #mLSTM formulations, check out the [xLSTM paper on arXiv](https://arxiv.org/abs/2405.04517)


## Linear attention
#self-attention , the core component of the #transformer architecture is expressed as follows:

$$
\begin{align}
&\forall t,\; Q_{t}= W^{Q}o_{t},\; K_{t} = W^{K}o_{t},\; V_{t} = W^{V}o_{t} \\

& \alpha_{1}\cdots \alpha_{T}= \text{softmax}\frac{[Q_t^{T}K_{1} \cdots Q_{t}^{T}K_{T}] }{ \sqrt{D}}\\

& y_{t} = \sum_{s=1}^{t} m_{s,t}\alpha_{s}V_{s} 
\end{align}
$$

Where $o_{t} \in \mathbb{R}^{D \times 1}$, $W \in \mathbb{R}^{N \times D}$, $Q_{t}, K_{t}, V_{t} \in \mathbb{R}^{N \times 1}$ and the *causal mask* $m_{s,t} = 1 (s \le t)$.

Using the #softmax non linearity makes that computing the attention scores has a complexity of $O(N^2)$ in both time and memory during training as well as inference. This complexity also scales quadraticly  with respect to the sequence length.
$O(N^2)$ because the #dot-product $Q \cdot K$  has to be computed first before normalizing the scores using softmax.

### Rethinking attention
Nonetheless, attention can be linearized. Two papers address this problem. #Linformer proves that by projecting #query and #key matrices into lower rank dimensions the complexity becomes linear. But we will focus on the paper **Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention**. Here authors prove that attention computation can be linearized by using a #kernel.

$$
A(x) = V' = \text{softmax}(\frac{QK^T}{\sqrt{D}})V
$$
In the attention formulation above, the softmax function is applied rowwise to $QK^T$. Hence, the formula can be rewritten as:

$$
V'_{i}= \frac{\sum_{j=1}^{N} \text{sim}(Q_{i},K_i)V_j}{\sum_{j=1}^{N} \text{sim}(Q_{i},K_i)}
$$

with $\text{sim}(q,k) = \exp{(\frac{q^{T}k}{\sqrt{D}})}$.
The only constraint imposed on the $\text{sim}(\cdot)$ function to make the equation above to define an attention is to be **non-negative**. This includes all kernels $k(x,y) : \mathbb{R}^{2 \times F} \rightarrow \mathbb{R}_{+}$ with a feature representation $\phi(x)$.

Hence, the equation can be rewritten as:
$$
\begin{align}
V_{i}' &= \frac{\sum_{j=1}^{N} \phi{(Q_{i})}^{T} \phi{(K_{j}) V_j}}{\sum_{j=1}^{N} \phi{(Q_{i})}^{T} \phi{(K_{j})}} \\

&= \frac{\phi{(Q_{i})}^{T} \sum_{j=1}^{N}  \phi{(K_{j}) V_j}}{\phi{(Q_{i})}^{T} \sum_{j=1}^{N} \phi{(K_{j})}} \\
\end{align}
$$

>[!note]
>See the computation in a vectorized form as follows:
>$(\phi(Q) \; \phi(K)^{T}) \; V = \phi(Q) \; (\phi(K)^{T} \; V)$

$\phi(\cdot)$ is applied row-wise to $Q$ and $K$.
By reconstructing the computation this way, we can compute $\sum_{j=1}^{N} \phi{(K_{j})} V_{j}^T$ and $\sum_{j=1}^{N} \phi{(K_{j})}$ once and reuse them for each query.

>[!info]
>By leveraging matrix multiplication associativity, we don't need to multiply the $Q_i$  by $K_jV_j^T$ at each step of the sum (which creates the $O(N^2)$ complexity but have a global value that summaries the whole valuation of the current computation). This is why this attention implementation has a time complexity of $O(N)$.

This equation can be rewritten to compute attention in a #causal language modeling setup. In causal LM, given that the $i$-th can only be influenced by tokens at position $j$ ($j \le i$), then the formula becomes:

$$
V_{i}'= \frac{\phi{(Q_{i})}^{T} \sum_{j=1}^{i}  \phi{(K_{j}) V_j}}{\phi{(Q_{i})}^{T} \sum_{j=1}^{i} \phi{(K_{j})}} 
$$

And by introducing $S_{i} = \sum_{j=1}^{N} \phi{(K_{j})} V_{j}^{T}$ and $Z_{i} = \sum_{j=1}^{N} \phi{(K_{j})}$, the equation becomes:

$$
V_{i}' = \frac{\phi{(Q_{i})}^{T} S_{i}}{\phi{(Q_{i})}^{T} Z_{i}}
$$

### Transformers are RNNs
With linear attention, inference time and memory cost is **constant**. Therefore, the $S_i$ matrix can be stored as an internal state and update it at every time step like an RNN.
Authors formalized the original attention formula to have an RNN formulation with two hidden states:
- $s$: the **attention memory** (think of xLSTM cell state $c_t$).
- $z$: **normalizer memory** (think of xLSTM normalizer state $n_t$).

Each layer $f_l$ function is rewritten as:
$$
\begin{align}
s_{0} &= 0, \\
z_{0} &= 0, \\
s_{i} &= s_{i-1} + \phi{(x_{i}W_{K})}(x_{i}W_{V})^{T}, \\
z_{i} &= z_{i-1}  + \phi{(x_{i}W_{K})}, \\
y_{i} &= f_{l}(\frac{\phi{(x_{i}W_{Q})}^{T}s_{i}}{\phi{(x_{i}W_{Q})}^{T}z_{i}} + x_i)

\end{align}
$$

## Distilling a Transformer into an xLSTM

### Inspiration
In the MOHAWK (Matrix Orientation, Hidden-State Alignment, Weight-Transfer and Knowledge Distillation) paper, it is stated that sequence model architectures ( #SSM ) can be broken down to two core components *sequence mixer* and *channel mixer*.
In the case of a #transformer model, the sequence #mixer is the #attention module and the #MLP the channel mixer.

A sequence mixer is a parameterized map on sequences $Y = f_{\theta}(X)$ where $X, Y \in \mathbb{R}^{(T,P)}$ with $\theta$ being a collection of parameters and $T$ the #time or #sequence axis. Hence, $X_{t}\in \mathbb{R}^P$.
It's a way of combining tokens at different time steps. Think of it as a process during which tokens interact with each other like the computation of the #attention scores.
Transformers *matrix mixer or sequence transformation matrix* can be represented as $Y = MX$ where $M \in \mathbb{R}^{(T, T)}$.

In the paper *The Mamba in the Llama: Distilling and Accelerating Hybrid Models*, two main technical challenges are faced in this work:
- Find a way to map the #transformer's weight to a #linear #RNN ones through distillation
- How to adapt Transformer's inference practices like speculative decoding to this architecture.

### Methodology
In this section, we shall try to find a way to initialize xLSTM blocks from attention layers, distill the Transformer and implement speculative decoding with xLSTM.

***WIP***