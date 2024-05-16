Introduit par #Rosenblatt en **1957**, le #perceptron [^1] est l'un des premiers algorithmes de ML. Il a été implémenté directement avec le hardware, c-à-d en manipulant les cables électriques et non du code.

Le perceptron fait des prédictions en utilisant une somme pondérée des entrées avec ses poids, ainsi qu'une fonction de seuil. On a donc:
$$
z^{[i]} = b + \sum_{j=1}^{m}x^{[i]}_jw_j \rightarrow 

\begin{cases}
1, & \text{si } z^{[i]} \gt 0 \\
0,& \text{sinon}
\end{cases}
$$

Avec $x^{[i]}$ le $i^{\text{ème}}$ échantillon du dataset et $j$ indiquant l'indice du feature en entrée.
$z$ étant la sortie, $z$ passe par la fonction de seuil pour produire 0 ou 1.

![[a perceptron.png]]

Afin d'ajuster les poids du perceptron pour de meilleures prédictions, on calcule d'abord l'erreur via:
$$
\begin{align}
error = y^{[i]} - \hat{y}^{[i]} \\

y^{[i]}: \text{la valeur à prédire} \\
\hat{y}^{[i]}: \text{la valeur prédite}
\end{align}
$$

Les poids et le terme de biais sont mis à jour en appliquant la formule:
$$
\begin{align}
&w_j = w_j + error \times x_j^{[i]} \\
&b = b + error
\end{align}
$$




[^1]: [rosenblatt-1957.pdf (umass.edu)](https://blogs.umass.edu/brain-wars/files/2016/03/rosenblatt-1957.pdf)