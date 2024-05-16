## Transformer architecture
Les #transformers[^1] sont les sucésseurs des [[Recurrent Neural Network|RNN]] dans le domaine du NLP. Vu leur efficacité, les RNN sont plus utilisés pour les tâches de time series analysis plutôt que le NLP.

Le succès des transformers est principalement dû au mécanisme d' #attention[^2] qu'ils utilisent. Dans un problème de type [[Recurrent Neural Network#Many-to-Many| many-to-many]] (traduction), on serait amener à traduire la phrase en entrée, mot par mot et cela peut donner une mauvaise traduction.
Plutôt que de prendre la phrase comme un tout et de la traiter, on devrait d'abord la *lire*, la *comprendre* et ensuite fournir une *traduction*.

En considérant le RNN tel un couplage #encoder - #decoder, on constate qu'il y a le dernier hidden state de l'encodeur qui doit résumer toute l'information avant de la passer au décodeur. Mais le problème est que pour une très grande séquence en entrée, on perd trop d'information. Et c'est là que le mécanisme d'attention entre en jeu.

 ```img-gallery
  path: DL Fundamentals/assets/translation-rnn-bottleneck
  type: vertical
  gutter: 10
  radius: 12
  columns: 2
```


## Le mécanisme d'attention
L'idée générale du mécanisme d'attention est de créer des #context vectors qui contiennent des informations par rapport à toute la séquence.

Le mécanisme d'attention consiste en premier temps à utiliser un RNN #bidirectionnel afin de traiter la séquence dans les deux sens et d'obtenir deux hidden states par élément $x^{(i)}$. Ainsi pour tout $x^{(i)}$, on a deux représentation $h_F^{(i)}$ pour le forward pass du RNN et $h_B^{(i)}$ lorsque le RNN traite le séquence par l'arrière. Les deux hidden states sont concaténés en un seul vecteur $h^{(i)}$. A chaque $h^{(i)}$ est associé un #score d'attention $\alpha_{(i)}$ Avec une somme pondérée, on calcule un #context vector $c_{(i)}$ avec:
$$
c = \sum_{i=1}^{n}\alpha_{(i)}h^{(i)}
$$
 
![[Attention mechanism - rnn 1.png]]

Dans une architecture many-to-many traditionnelle, le second RNN produisant les outputs, au lieu d'avoir le comportement normal d'un RNN, on ajoute le vecteur contextuel à chaque #time-step à l'input.

![[Full attention RNN.png]]

>[!Note]
>Les scores d'attention sont relatifs à chaque time-step. Ainsi pour une séquence de taille $N$, on a les scores $\alpha_{i,j}$ où $i$ est l'indice du time-step et $j$ la position de l'input auquel est associé le score ($1 \le j \le N$). Du coup, le context vector aussi est relatif à l'input d'indice $i$.

 ### Self-attention
 L'architecture #transformer est essentiellement basée sur le mécanisme de #self-attention et contrairement aux RNN **aucune récurrence** n'est requise. 
 Un context aware embedding vector est défini par:
$$
z^{(i)} = \sum_{j=1}^{T}\alpha_{ij} \cdot x^{(j)}
$$
Avec $i$ l'indice du time-step actuel.
Pour calculer les scores d'attention, on procède comme suit:
1. On calcule d'abord une mesure de #similiraté entre l'input à la position $i$ par rapport aux autres inputs à la position $j = 1 \dots T$, notée $\omega_{ij}$:
   $$
   \omega_{ij} = x^{(i)T} \cdot x^{(j)}
   $$
2. On normalise ensuite chaque valeur $\omega$ obtenue pour obtenir les scores d'attention $\alpha$:
   $$
   \alpha_{ij} = \frac{\exp{(\omega_{ij})}}
        {\sum_{j=1}^{T}\exp{(\omega_{ij})}} = 
        \text{softmax($[\omega_{ij}]_{j=1 \dots T}$)}
   $$
   <u>NB</u>: On utilise la fonction softmax par convention afin que la somme des scores d'attention soit égale à 1.

![[Basic self-attention mechanism.png]]

### Scaled dot-product attention
La forme de self-attention [[#Self-attention | précédente]] n'a aucun learnable weights. Pour pallier à cela on utilise le scaled #dot-product, c'est la forme de self-attention la plus utilisée.
Le scaled dot-product introduit 3 matrices qui représentent les poids à apprendre par le modèle:
- $U_q$: appelée #query sequence
- $U_k$: appelée #key sequence
- $U_v$: appelée #value sequence

On a donc:
$$
\begin{align}
\text{query sequence:} \; q^{(i)} = U_qx^{(i)} \; \forall i \in [1, \dots, T] \\
\text{key sequence:} \; k^{(i)} = U_kx^{(i)} \; \forall i \in [1, \dots, T] \\
\text{value sequence:} \; v^{(i)} = U_vx^{(i)} \; \forall i \in [1, \dots, T] \\
\end{align}
$$

1. **Query**: Think of the query as a question or something you want to learn more about. It's like the part of your brain that's looking for specific information. In the context of self-attention, the query helps the model focus on a particular aspect of the input data. It's what you're interested in understanding better.
    
2. **Key**: The key is like a tag or identifier associated with the information. You can think of it as a label that tells the model what each piece of information is. Keys help the model decide which parts of the input data are relevant to the query. They guide the model's attention to the right places.
    
3. **Value**: Values are the actual information or answers to the query. They are like the content that you want to retrieve based on your question (query) and the labels (keys) associated with the information. The values provide the model with the data it needs to give you a meaningful response.



>[!Info]
>Cette nomenclature de query, key, value est issu des systèmes d'information. C'est dire qu'une query est matchée à une key pour récupérer une value, sauf qu'ici c'est pour avoir le context vector.

A chaque input est associé ses valeurs $q^{(i)}, k^{(i)}$ et $v^{(i)}$. On calcule la métrique $\omega_{i,j}$ et multipliant la query de l'input actuel aux clés des autres, puis la normalisation des ces $\omega_{i,j}$ donnent les différents scores d'attention. Après quoi on pourra calculer le context vector de l'input actuel.

![[query-key-value attention scores.png]]


### Multi-head attention
Dans un contexte de #multi-head attention, on a $h$ matrices $U_q, U_k, U_v$ qui sont empilées. Ainsi on les note:
$$
U_{q_i}, U_{k_i}, U_{v_i}
$$
Où $i$ est la position de la tête dans la matrice ($i \in [1, \dots, h]$).
Pour le #single-head attention, le calcul est comme suit:
![[single-head attention.png]]

Pour le multi-head attention, le calcul reste inchangé, juste qu'on a $h$ calculs qui se font en parallèles. On obtient $h$ context vector qu'on le concatine, puis on les compresse en un seul context vector en les faisant passer dans une couche *Linear*.

```img-gallery
  path: DL Fundamentals/assets/multi-head attention
  type: vertical
  gutter: 10
  radius: 12
  columns: 2
```


## Masked multi-head attention
Le module #masked multi-head attention a pour but de pousser le decoder à ne produire qu'un token à la fois. Dans le cas de la tâche de #machine-translation, on aura un #token **\<BOS>**(beginning of sentence) et un autre **\<EOS>**(end of sentence). A chaque étape de la traduction, la masked multi-head attention cache le reste du résultat pour ne laisser entrevoir que la partie déjà traduite ou le \[BOS].

```img-gallery
  path: DL Fundamentals/assets/masked-attention
  type: vertical
  gutter: 10
  radius: 12
  columns: 2
```


## Positional encoding
On a besoin de l'encodage positionnel car le mécanisme de self-attention et les couches linéaires sont **invariants à la permutation**, c-à-d qu'ils n'ont pas la notion de **séquence**.
>[!Note]
>![[Pasted image 20231024093839.png]]
>On obtient le même résultat peu importe l'ordre des features. Donc c'est comme si on modifiait l'ordre des mots dans une phrase, on aurait une phrase qui n'a pas de sens.

Pour effectuer le #positional-encoding on ajoute aux vecteurs de tokens, des vecteurs positionnels qui garderont l'ordre des tokens dans la séquence.
![[Pasted image 20231024094420.png]]


[^1]: [Attention is all you need](https://arxiv.org/abs/1706.03762)
[^2]: [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473)