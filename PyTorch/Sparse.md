---
docs: https://pytorch.org/docs/stable/sparse.html#
---
Le module `torch.sparse`[^1] traite de tout ce qui est en rapport avec les tenseurs #creux.

De façon générale, les tenseurs sont stockés de façon **contiguë** dans la mémoire pour faciliter l'implémentation de divers d'algorithmes. Cependant des données telles que les *matrices d'adjacence, les #pruned weights, etc..* sont généralement des données assez #sparse.
Pour les représenter divers formats ont été proposés tel que: #COO, #CSR, #CSC, #LIL, etc...

>[!Info]
>[[Geometric | PyTorch Geometric]] utilise le format #COO pour la représentation des graphes.

Pour convertir un tenseur #dense en tenseur #creux, on peut simplement utiliser la méthode `to_sparse()`.



[^1]: [torch.sparse — PyTorch 2.1 documentation](https://pytorch.org/docs/stable/sparse.html#)