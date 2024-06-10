---
lien: https://huggingface.co/learn/deep-rl-course/unit0/introduction
---
## Reinforcement Learning - 101

The idea behind #RL is to make an #agent (an AI) learn from an #environment through **interaction** with it. RL is a framework used to solve *control tasks* also called *decision problems*.

### The RL framework

![[Pasted image 20240215140305.png]]


The figure above depicts an agent's learning process using RL. Based on the current state $S_t$, the agent takes the #action $A_t$ goes in a new #state $S_{t+1}$ and receives a #reward $R_{t+1}$ (it can a bonus or a malus).
The agent aims to maximize its cumulative reward, called the *expected return*. The agent does so because RL is based on the **reward hypothesis**.

### Markov property
RL is often called a #Markov decision process ( #MDP ). It's called MDP because the agent only need to take into account the current state to decide which action to take next. It doesn't need to the history of all of its previous actions.

Observations/states are information that the agent gets from the environment. Nonetheless, there's a subtle nuance between a state and an observation.
- A state $S$: is **a complete description of the state of the world** (there is no hidden information). In a fully observed environment.
- An observation $O$: is a **partial description of the state.** In a partially observed environment.

For instance, in a chess game, we receive a state from the environment since we have access to the whole chess board information. However, in a game like Super Mario, we receive observations given that we can't see the whole level.
![[Pasted image 20240215142058.png]]

The **action space** is the set of all available actions. It can be **discrete** or **continuous**. A discrete action space has a **finite** number of possible actions (in Super Mario, one can move in 4 directions only), where as a continuous one has **infinite** possible actions (self-driving car).

The cumulative reward at each time step $t$ can be written as:

$$
\begin{align}
R(\tau) &= r_{t+1} + r_{t+2} + \dots \\
&= \sum_{k=0}^{\infty} r_{t+k+1}
\end{align}
$$
Where $\tau$ is called #trajectory, a sequence of states and actions.
The rewards can't be just summed up like this. The rewards that come sooner (at the beginning of the game) **are more likely to happen** since they are more predictable than the long-term future reward.

![[Pasted image 20240215144046.png]]

Let’s say your agent is this tiny mouse that can move one tile each time step, and your opponent is the cat (that can move too). The mouse’s goal is **to eat the maximum amount of cheese before being eaten by the cat.** As we can see in the diagram, **it’s more probable to eat the cheese near us than the cheese close to the cat** (the closer we are to the cat, the more dangerous it is). Consequently, **the reward near the cat, even if it is bigger (more cheese), will be more discounted** since we’re not really sure we’ll be able to eat it.

To discount rewards, a discount rate $\gamma \in [0, 1]$ is defined. The larger the gamma, the smaller the discount. This means that the agent cares more about the **long-term reward**. Where as, a lower $\gamma$ makes the agent care more about **short-term reward**, in this case, the nearest cheese. Each reward is discounted by $\gamma$ to the exponent of the current time step. As the time step increases, the future reward is less and less likely to happen.

The discounted cumulative expected return can be written as follows:

$$
\begin{align}
R(\tau) &= r_{t+1} + \gamma r_{t+2} + \gamma^2 r_{t+3} + \gamma^3 r_{t+4} + \dots \\
&= \sum_{k=0}^{\infty} \gamma^k r_{t+k+1}
\end{align}
$$

### Tasks in RL
An instance of a RL problem is called a #task. It can be #episodic or #continuing. 
An episodic task is a task with a starting point and an ending one (**terminal state**). This create what we call an #episode, that is a list of states, actions, rewards and new states (complete a Super Mario level).
A continuing task is a task that continues forever (**no terminal state**). In this case, the agent must **learn how to choose the best actions and simultaneously interact with the environment** (automated stock trading).
### Exploration/Exploitation trade-off
- #Exploration consists of exploring the environment by trying random actions in order to **find more information about the environment.**
- #Exploitation is **exploiting known information to maximize the reward.**

### Solving RL problems
There are two main approaches for solving RL problems.
#### The policy $\pi$: the agent's brain
The #policy **$\pi$** is the **brain of our agent**, it’s the function that tells us what **action to take given the state we are in**. So it **defines the agent’s behavior** at a given time.

>[!note]
>The policy can be seen as a #strategy or a #heuristic.
 
![[Pasted image 20240216000116.png]]

This policy is a learnable function and the goal is to find the optimal one $\pi^*$ that maximizes the expected return. $\pi^*$ can be found through training.
There are two approaches to train our agent to find this optimal policy $\pi^*$:
- **Directly,** by teaching the agent to learn which **action to take,** given the current state: **Policy-Based Methods.**
- Indirectly, **teach the agent to learn which state is more valuable** and then take the action that **leads to the more valuable states**: Value-Based Methods.

#### Policy-Based Methods
In Policy-Based methods, **we learn a policy function directly.**

This function will define a mapping from each state to the best corresponding action. Alternatively, it could define **a probability distribution over the set of possible actions at that state.**

There are two types of policies:
- *Deterministic*: for a given state **will always return the same action**. In other words, $action = policy(state)$.
  $$
   a = \pi(s)
  $$  
- _Stochastic_: outputs **a probability distribution over actions.**
  $$
  \pi(a \vert s) = P[A\vert s]
  $$
#### Value-based methods
In value-based methods, instead of learning a policy function, we **learn a value function** that maps a state to the expected value **of being at that state.**

The value of a state is the **expected discounted return** the agent can get if it **starts in that state, and then acts according to our policy.**
“Act according to our policy” just means that our policy is **“going to the state with the highest value”.**

$$
v_{\pi}(s) = \mathbb{E}_{\pi}[R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \dots \vert S_t = s]
$$

![[Pasted image 20240222211954.png]]

### The "Deep" in Reinforcement Learning
Deep RL leverages neural networks, hence the name *Deep RL*.



