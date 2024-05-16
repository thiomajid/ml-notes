## Permute
La méthode `permute`[^1] définit sur les tensors permet de déplacer la position des dimensions dans le tensor.

```py
import torch

img = torch.randn(3, 256, 256)
plottable_img = img.permute(1, 2, 0)
```

Dans cet exemple, on a un tenseur qui peut être passer à un CNN pour le traiter mais on ne peut pas l'afficher avec `plt.imshow`. Pour ce faire on est obligé de modifier l'ordre des dimensions du tensor pour passer du format $C \times H \times W$ au format $H \times W \times C$ supporter par *matplotlib*.

Dans la méthode *permute* les valeurs passer en paramètre représentent l'indice de la dimension lorsqu'on appelle la méthode `size()` du tensor et l'ordre dans lequel on passe les indices à la fonction détermine le nouvel indice de la dimension du tensor.

Du coup, pour avoir la première dimension comme troisième dimension, on passe $0$ comme troisième paramètre à la fonction *permute* et ainsi de suite.

>[!Exemple]
>`permute(1,2,0)` fait que la deuxième dimension devienne la première (car premier paramètre), la troisième dimension la deuxième (car deuxième paramètre) et enfin la première dimension comme la troisième (car troisième paramètre)





[^1]: [torch.permute — PyTorch 2.1 documentation](https://pytorch.org/docs/stable/generated/torch.permute.html#torch.permute)