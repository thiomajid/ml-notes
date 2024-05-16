# Définition
---
Le neural style transfer[^1] s'appuit sur les CNN[[Chapitre 2 - Deep learning#^f9b712]] afin d'extraire une représentation d'une image donnée (content image) et celle d'une image représentant une oeuvre d'art (style image) de façon séparée puis de les unifier afin de créer une oeuvre d'art alliant les deux.
![[nst_illustration.png]]

![[nst_applied.png]]

# Procédé
---
Les chercheurs ont eu recours à des modèles entrainer sur la tâche d'object recognition afin d'extraire la `neural representation` de l'image en entrée afin de la reconstruire dans le style de la `style image`.

Le fait d'effectuer le rendu d'une image dans le style d'oeuvres d'art connues est une approche d'une branche de CV appelée `non-photorealistic rendering`[^2].

## Fonction de coût
---
On ne peut entièrement séparer le contenu et le style d'une image. Lors de la reconstruction d'une image, la fonction de coût a deux termes $\alpha$ et $\beta$ qui mettent l'accent respectivement sur le _contenu_ et le _style_ d'une image.
Si l'accent est mis sur le style, alors l'image résultant de l'opération ressemblera à l'oeuvre d'art *style image*.

Chaque entrée $\vec{x}$ est encodée par dans chaque couche du CNN par leurs filtres. Une couche avec $N_l$ filtres distincts a $N_l$ feature maps de taille $M_l$ avec $M_l = width * height$ de la feature map. La sortie d'une couche $l$ peut être représentée comme une matrice $F^l \in \mathbb{R}^{N_l x M_l}$  où $F_{ij}^l$ est l'activation du $i^{ème}$ filtre à la position $j$ dans la couche $l$.
Soit $\vec{p}$ et $\vec{x}$ respectivement l'image originale et l'image générée et$P^l$ et $F^l$ leurs représentations dans la couche $l$. La _MSE_ entre $P$ et $F$ est définie par: 
$$
\mathcal{L}_{content}(\vec{p}, \vec{x}, l) 
	= \frac{1}{2}\sum_{i,j}(F_{ij}^l - P_{ij}^l)^2
$$
La dérivée de $\mathcal{L}$ par rapport à l'activation est définie comme suit:
$$
\frac{\partial{\mathcal{L_{content}}}}{\partial{F_{ij}^l}}
	= \begin{cases}
			(F^l - P^l)_{ij} & si F_{ij}^l > 0 \\
			0 & si  F_{ij}^l < 0
		\end{cases}
$$

Outre les filtres des CNN, une représentation du style de l'image est construit en calculant la corrélation entre les différentes réponses de filtres. La corrélation mesure à quel point les réponses de deux filtres sont similaires ou liées les unes aux autres. Cette corrélation s'obtient via la matrice de _Gram_ $G^l \in \mathbb{R}^{N_l \times N_l}$ où $G$ est le produit scalaire des feature maps $i$ et $j$ dans la couche $l$.
$$
G_{ij}^l = 
	\sum_{k}F_{ik}^lF_{jk}^l
$$

### Le style d'une image
---
Afin d'avoir une texture correspondant au style de la "style image", on utilise une image avec du `bruit blanc`[^3] et par descente des gradients, on essaye de trouver une image dont le style correspond à la représentation de l'image originelle. Pour ce faire, on cherche à minimiser la distance quadratique entre les entrées de la matrice de Gram de l'image originelle et celles de l'image générée.
En considérant $\vec{a}$ l'image originale et $\vec{x}$ l'image générée et $A^l$ et $G^l$ la représentation de leur style dans la couche $l$, alors la contribution de $l$ à la fonction de coût est:

$$
\begin{equation}
E_l = 
	\frac{1}{4N_{l}^2M_{l}^2}\sum_{i,j}(G_{ij}^l - A_{ij}^l)^2
\end{equation}
$$

Et le coût total est:
$$
\mathcal{L_{style}}(\vec{a}, \vec{x}) = \sum_{l=0}^L w_lE_l
$$
Où $w_l$ est un facteur de pondération de la contribution de chaque couche à la perte totale.

La dérivée de $E_l$ par rapport aux activations de la couche $l$ est:
$$
\frac{\partial{E_l}}{\partial{F_{ij}^l}} = 
	\begin{cases}
		\frac{1}{N_l^2 M_l^2}((F^l)^T(G^l-A^l))_{ji} & si F_{ij}^l > 0 \\
		0 & si F_{ij}^l < 0
	\end{cases}
$$

Ainsi, afin de générer des images combinant le style d'une oeuvre d'art et d'une photographie donnée, on minimise la distance l'image à bruit blanc issue de la représentation de la photographie dans une couche et la "style representation" de l'oeuvre d'art dans un nombre de couches du CNN.
En considérant $\vec{p}$  la photographie et $\vec{a}$ l'oeuvre d'art, alors le coût total est définie par:
$$
\mathcal{L}_{total}(\vec{p}, \vec{a}, \vec{x}) = 
	\alpha\mathcal{L}_{content}(\vec{p}, \vec{x}) +
	\beta\mathcal{L}_{style}(\vec{a}, \vec{x})
$$
Avec $\alpha$ et $\beta$ qui sont respectivement des facteurs de pondérations pour la reconstruction du contenu de l'image et de son style.

[^1]: Article introduisant le NST: [[1508.06576] A Neural Algorithm of Artistic Style (arxiv.org)](https://arxiv.org/abs/1508.06576)

[^2]: [State of the "Art”: A Taxonomy of Artistic Stylization Techniques for Images and Video | IEEE Journals & Magazine | IEEE Xplore](https://ieeexplore.ieee.org/document/6243138)

[^3]: Le bruit blanc fait référence à une image générée aléatoirement qui contient des valeurs de pixel aléatoires et indépendantes. Plus précisément, il s'agit d'une image où les valeurs de pixel sont échantillonnées à partir d'une distribution aléatoire uniforme sur un intervalle donné, généralement entre 0 et 1. On peut avoir une image dont les valeurs sont dans l'intervalle $[0, 255]$ et utiliser _transforms.ToTensor()_ pour avoir l'intervalle $[0, 1]$