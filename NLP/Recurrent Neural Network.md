Les RNN[^1] permettent le traitement de données ***ordonnées séquentielles*** grâce  à leur mécanisme interne.

>[!Warning]
>Il est important de maintenir **l'ordre** sinon les données ne sont plus une séquence mais forment plutôt une collection. Un RNN est tout l'opposé de la plupart des algorithmes de ML qui considèrent que les inputs sont *independent and identically distributed(IID)*.

>[!Info]
>Les séries temporelles sont des données séquentielles auxquelles est ajoutée une dimension *temporelle*.

## Les types de modélisations séquentielles
Suivant le type de problème que l'on résout, il est impératif de choisir la bonne architecture. Ainsi on distingue:
### Many-to-one: 
Dans ce cas de figure l'input est une *séquence* et la sortie est un *vecteur de taille fixe ou un scalaire*. C'est le cas de l'analyse de sentiment dans le cas par exemple de l'évaluation de l'avis de personnes sur différents films.
  ![[rnn many-to-one-v1.png]]
### One-to-many: 
Ici l'input est une donnée au format *standard* mais la sortie est une *séquence*. Un exemple est la tâche d'image captioning où on donne une image en entrée pour obtenir un texte décrivant le contenu de l'image.
  ![[rnn one-to-many.png]]
### Many-to-Many: 
Dans ce genre de problèmes, l'entrée et la sortie sont toutes des *séquences*, mais on peut diviser ce problème en deux sous-catégories selon que l'input et l'output sont *synchronisées* ou qu'il y a un *retard* entre les deux.
Un cas de synchronisation serait la tâche de video classification, où à chaque frame est associée un label. Dans le cas où il y a un retard ce serait la tâche de machine translation.
  
 ```img-gallery
  path: NLP/assets/rnn-many-to-many
  type: vertical
  gutter: 12
  radius: 12
  columns: 2
```



## Principe des RNN
De part son fonctionnement un RNN effectue ses opérations en se basant sur le résultat de l'opération précédente ($h^{(t-1)}$) et du nouveau point de donnée en entrée $x^{(t)}$ pour produire une nouvelle sortie $h^{(t)}$. A $t=0$, $h^{(t-1)}$ est soit nul ou initialisé avec de petites valeurs aléatoires.

Dans un RNN, on distingue les poids suivants:
- $W_{xh}$: c'est la matrice de poids associée à l'input $x^{(t)}$ dans la couche $h$.
- $W_{hh}$: c'est la matrice de poids associée à la feedback loop.
- $W_{ho}$: c'est la matrice de poids entre la hidden layer et la couche de sortie.
![[rnn weight matrices.png]]

>[!Info]
>Parfois les matrices $W_{xh}$ et $W_{hh}$ sont combinées dans une seule et même matrice notée $W_{h} = [W_{xh}; W_{hh}]$.


Afin de calculer les activations d'un RNN, on calcule d'abord la pré-activation $z_h$ via une combinaison linéaire:
$$
z_{h}^{(t)} = W_{xh}x^{(t)} + W_{hh}h^{(t - 1)} + b_h 
$$

Après quoi, on peut calculer l'activation $h^{(t)}$ des hidden units via:
$$
h^{(t)} = \sigma_h(z_h^{(t)}) = \sigma_h(W_{xh}x^{(t)} + W_{hh}h^{(t - 1)} + b_h )
$$
<u>NB</u>: $\sigma(\cdot)$ représente la fonction d'activation de la hidden layer.

>[!Warning]
>On peut aussi noter $h^{(t)}$ avec la notation:
>$$
>h^{(t)} = \sigma_h(
>	\begin{bmatrix}
>		W_{xh} & W_{hh} 
>	\end{bmatrix}
>	\begin{bmatrix}
>		x^{(t)} \\
>		h^{(t-1)}
>	\end{bmatrix}
> + b_h)
>$$

Après avoir calculé $h^{(t)}$ on peut calculer les activations de la sortie $o^{(t)}$:
$$
o^{(t)} = \sigma_o(W_{ho}h^{(t)} + b_o)
$$

![[Computing RNN activations.png]]


Chaque sortie $h^{(t)}$ est appelée ***hidden state*** et le dernier hidden state lors des opérations du RNN est la prédiction de la couche.
Un hidden state est juste un vecteur représentant toute la séquence en entrée et dont la taille est un choix qui m'est échu.

![[internals of an RNN cell.png]]

Un RNN comprend 3 composantes:
- Une linear layer pour transformer le hidden state (en bleu)
- Une linear layer pour transformer la donnée en entrée (en rouge)
- Une fonction d'activation pour transformer la sommation des deux autres couches.

$$

\text{Après avoir initialisé $h_0$,} \;RNN = \begin{cases}

t_h = W_{hh}h^{(t-1)} + b_{hh} \\
t_x = W_{xh}x^{(t)} + b_{xh} \\
h^{(t)} = tanh (t_h + t_x)

\end{cases}
$$

Avec:
- $t_h$: transformed hidden state
- $t_x$: transformed data point
- $h^{(t)}$: updated hidden state


### Back propagation through time
Le calcul des gradients[^2] d'un RNN est un peu compliqué mais l'idée générale veut que le coût total $\mathcal{L}$ soit égale à la somme des coût de $t=1$ à $t=T$.
$$
\mathcal{L} = \sum_{1}^{T}\mathcal{L}^{(t)}
$$

Etant donné que le coût à un temps $t$ dépend des hidden units des étapes $1:t$, le gradient est calculé comme suit:
$$
\frac{\partial\mathcal{L}^{(t)}}{\partial W_{(hh)}} =
	\frac{\partial\mathcal{L}^{(t)}}{\partial o^{(t)}} \times
	\frac{\partial o^{(t)}}{\partial h^{(t)}} \times
	(
		\sum_{k=1}^{t} \frac{\partial h^{(t)}}{\partial h^{(k)}} \times
			\frac{\partial h^{(k)}}{\partial W_{(hh)}}
	)
$$

Il est à noté que $\frac{\partial h^{(t)}}{\partial h^{(k)}}$ est obtenue par multiplication des étapes adjacentes:
$$
\frac{\partial h^{(t)}}{\partial h^{(k)}} = 
	\prod_{i=k+1}^{t} \frac{\partial h^{(i)}}{\partial h^{(i-1)}}
$$

## Les types de récurrences dans un RNN
Lorsque la plupart du temps, on réutilise la sortie $h^{(t-1)}$ pour calculer la nouvelle sortie $h^{(t)}$, il est également possible d'utiliser la valeur $o^{(t)}$ dans la feedback loop.
Ainsi, $o^{(t)}$ peut être ajouté à $h^{(t)}$ de l'étape $t$ (*output-to-hidden recurrence*) ou à $o^{(t)}$ de l'étape $t$ actuelle (*output-to-output recurrence*).

![[Types of recurrent connections.png]]

De ce fait dans les poids associées à la relation de récurrence sont notés $W_{hh}$ pour une récurrence hidden-to-hidden, $W_{oh}$ pour la forme output-to-hidden et $W_{oo}$ pour la relation output-to-output. On peut généraliser la notation des poids de la relation de récurrence avec $W_{rec}$.







[^1]: [The Unreasonable Effectiveness of Recurrent Neural Networks (karpathy.github.io)](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)

[^2]: [Backpropagation through time: what it does and how to do it | IEEE Journals & Magazine | IEEE Xplore](https://ieeexplore.ieee.org/document/58337)