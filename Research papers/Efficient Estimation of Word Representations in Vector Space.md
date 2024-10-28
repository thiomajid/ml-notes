Un principe important en NLP est la représentation des données. En l'état actuel des choses, au moment de la publication du dit article, les mots étaient traités comme des unités *atomiques*, ils étaient représentés tout simplement comme des *indices* dans le vocabulaire. Avec cette approche, il manquait la notion de #similiraté entre les mots.

Cette approche offre certes simplicité, robustesse et aussi une meilleure performance des simples modèles tel que le #n-gram comparé aux solutions plus complexes, et ce malgré de grandes quantités de données. Mais dès lors qu'on se spécialise dans un domaine donné, vu que la quantité de données disponible diminue, alors ces simples solutions perdent en efficacité.

Le #word2vec vise à introduire une architecture plus robuste, moins coûteuse en temps et en ressources, comparée aux précédentes propositions.

## Comparaison d'architectures
En comparant différent architectures, il a été établi que la complexité d'entrainement est proportionnelle à:
$$
O = E \times T \times Q
$$
Avec:
- $E$: le nombre d'époques.
- $T$: le nombre de mots dans le training set.
- $Q$: est variable suivant l'architecture.

>[!Info]
>Généralement on a $E \in [3;50]$ et $T$ peut aller jusqu'à un milliard.
>De plus les modèles sont entrainés avec le SGD et la rétro-propagation.

### Feedforward Neural Net Language Model (NNLM)
