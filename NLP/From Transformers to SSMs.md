In the MOHAWK (Matrix Orientation, Hidden-State Alignment, Weight-Transfer and Knowledge Distillation) paper, it is stated that sequence model architectures can be broken down to two core components *sequence mixer* and *channel mixer*.
In the case of a #transformer model, the sequence #mixer is the #attention module and the #MLP the channel mixer.

A sequence mixer is a parameterized map on sequences $Y = f_{\theta}(X)$ where $X, Y \in \mathbb{R}^{(T,P)}$ with $\theta$ being a collection of parameters and $T$ the #time or #sequence axis. Hence, $X_{t}\in \mathbb{R}^P$.
It's a way of combining tokens at different time steps. Think of it as a process during which tokens interact with each other like the computation of the #attention scores.

Transformers *matrix mixer or sequence transformation matrix* can be represented as $Y = MX$ where $M \in \mathbb{R}^{(T, T)}$.


>[!Info]
>The following content comes from the paper: *The Mamba in the Llama: Distilling and Accelerating Hybrid Models*

Two main technical challenges are faced in this work:
- Find a way to map the #transformer's weight to a #linear #RNN ones through distillation
- How to adapt Transformer's inference practices like speculative decoding to this architecture.

The computation of attention score is formulated as:
$$
\begin{align}
&\forall t,\; Q_{t}= W^{Q}o_{t},\; K_{t} = W^{K}o_{t},\; V_{t} = W^{V}o_{t} \\

& \alpha_{1}\cdots \alpha_{T}= \text{softmax}\frac{[Q_t^{T}K_{1} \cdots Q_{t}^{T}K_{T}] }{ \sqrt{D}}\\

& y_{t} = \sum_{s=1}^{t} m_{s,t}\alpha_{s}V_{s} 
\end{align}
$$

Where $o_{t} \in \mathbb{R}^{D \times 1}$, $W \in \mathbb{R}^{N \times D}$, $Q_{t}, K_{t}, V_{t} \in \mathbb{R}^{N \times 1}$ and the *causal mask* $m_{s,t} = 1 (s \le t)$.


All linear RNN take the following formulation with $h$ being a **matrix-valued** hidden state and $A_{t}, B_{t}, C_{t}$ learnable parameters:

$$
h_{t}= A_th_{t-1} + B_tx_{t,}\;\;\;\;\; y_{t}= C_th_t
$$

By linearizing the attention formula and removing the *softmax* non linearity, the computation can be reconstructed as follows:

$$
\begin{align}
y_{t} &= \sum_{s=1}^{t}m_{s,t}\alpha_{s}V_{s}\\
&= \frac{1}{\sqrt{D}} Q_{t} \sum_{s=1}^{t}(m_{s,t}K^{T}_{s}V_{s})\\
&= \frac{1}{\sqrt{D}} Q_{t} \sum_{s=1}^{t}m_{s,t}K^{T}_{s}W^{V}o_{s}\\
\end{align}
$$

Think of this reconstruction of the linearized attention as follows:
- The term $\sum_{s=1}^{t}m_{s,t}K^{T}_{s}W^{V}o_{s}$ is somehow similar the the *KV cache* as it summaries all past information being measuring the magnitude of similarities between past and present keys and values.
- The whole sum is modulated by $Q_t$, the current query thus making the computation more efficient.

Hence, in an RNN spirit, $Q_t$ is the current input at time step $t$, ($Q_{t}= x_t$) and the term $\sum_{s=1}^{t}m_{s,t}K^{T}_{s}W^{V}o_{s}$ is the previous hidden state $h_{t-1}$ 