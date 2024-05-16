## 1 - Representing text as tensors 
Les données textuelles peuvent être encodées de différentes manières. On distingue ainsi:
- **L'encodage sur lettres**: ici on traite chaque lettre du texte de façon individuelle. Ainsi le mot _"Hello"_ serait représenté par un tenseur de taille $5 \times C$ tel que chaque lettre correspondrait à une colonne dans le one-hot encoding.
- **L'encodage sur mots**: Ici, on crée un ***vocabulaire*** à partir de chaque mot de notre corpus, puis chaque mot est encodé(on lui attribut un nombre) via le one-hot encoding. Cependant avec un grand dictionnaire, on aura beaucoup de matrices sparses.
### Terminologie:
On appelle ***token*** une unité **atomique** de texte. Ça peut être un mot, une lettre ou une partie de mot.

La ***tokenisation*** vise à transformer un texte en une séquence de token et la ***vectorisation*** est le processus par lequel chaque token est encodé.


>[!Warning]
>Comme les objets de type `ShardingFilterIterDataPipe`[^1] représentent les données sous forme de fichiers distribués, alors il faut d'abord créer un itérateur à partir d'eux ou les convertir en liste pour pouvoir les indexer.

Lors de la création d'un vocabulaire, `min_freq` permet de filtrer les mots dont le nombre d'occurrences est inférieure à la valeur de _min_freq_.

L'un des principaux inconvénients de la tokenisation est que certains mots font partis de mots en plusieurs parties. Pour pallier à cela, on procède à une représentation sous forme de ***n-gram***. C'est dire qu'au lieu de créer un vocabulaire à partir de tokens individuels, on va créer des ensembles de _n_ tokens dans le vocabulaire.

## 2 - Bag-of-Words and TF-IDF representations

Avec l'approche ***bag of words (BoW)*** chaque token est associé un indice du vecteur de représentation et chaque élément du vecteur contient le nombre d'occurrences d'un mot dans un document donné. Elle permet de garder le sens d'une phrase sans pour autant garder l'ordre des mots.

>Pour les tâches de classification, il est monnaie courante d'utiliser un objet de type `LongTensor` pour représenter les classes. Non seulement cela optimise le code, mais également facilite la représentation des entiers et aussi parce que le modèle attend des tenseurs en entrées.

### Term Frequency / Inverse Document Frequency:  TF-IDF

Avec un BoW chaque mot est pondéré de la même manière pourtant les mots tels que: le, la, etc.. sont moins importants. Le TF-IDF est une variation du BoW où on utilise une **valeur flottante** pour indiquer la fréquence d'un mot dans le corpus plutôt qu'une valeur binaire indiquant la présence ou non du token.
Le TF-IDF a pour formule:

$$
w_{ij} = tf_{ij} \times \log{(\frac{N}{df_i})}
$$

Avec:
- $i$: est un mot
- $j$: est un document
- $w_{ij}$​: représente le poids (l'importance) du mot dans le document
- $tf_{ij​}$: est le nombre d'occurrences du mot $i$ dans le document $j$, en d'autres termes la valeur de son BoW.
- N: le nombre de document dans la collection.
- $df_i$: est le nombre de document contenant le mot $i$ dans toute la collection.

>[!Info]
>L'IDF d'un terme qui évalue l'importance du dit terme dans l'ensemble du corpus est élevé si ce terme apparait dans tous les documents. Cela entraine qu'il est un faible TF-IDF révélant ainsi sa **faible importance**.


## 3 - Represent words with embeddings
En entrainant une couche d'embedding[^2], il est préférable d'avoir des vecteurs de _même_ taille pour faciliter l'entrainement et assurer une _consistence_ dans l'embedding. Pour ce faire on effectue une opération de **padding**.

L'idée derrière les embedding est d'associer les mots à des vecteur reflétant leurs **_sens sémantique_**.

Le sens sémantique d'un mot se réfère à la notion, à l'idée sous-jacente ou au contenu que le mot représente. C'est l'essence de ce que le mot transmet en termes de signification, de référence et d'association avec d'autres mots ou concepts.

Dans le langage naturel, les mots sont utilisés pour représenter des entités spécifiques, des actions, des qualités ou des idées abstraites. Le sens sémantique d'un mot englobe plusieurs aspects :

1. **Dénotation :** Il s'agit de la définition primaire, littérale ou de dictionnaire d'un mot. Par exemple, le mot "pomme" se réfère au fruit du pommier.
    
2. **Connotation :** La connotation se réfère aux significations émotionnelles, culturelles ou subjectives supplémentaires et aux associations qu'un mot véhicule au-delà de sa définition littérale. Par exemple, le mot "foyer" peut évoquer des sentiments de confort, de sécurité et d'appartenance.
    
3. **Synonymes et antonymes :** Le sens sémantique d'un mot peut être exploré en considérant ses synonymes (mots ayant des significations similaires) et ses antonymes (mots ayant des significations opposées).
    
4. **Contexte :** La signification d'un mot peut varier en fonction du contexte dans lequel il est utilisé. Les mots peuvent avoir des significations différentes dans différentes phrases, paragraphes ou conversations.
    
5. **Polysémie :** De nombreux mots ont plusieurs significations liées. Par exemple, le mot "banque" peut faire référence à une institution financière ou au bord d'une rivière.
    
6. **Associations de mots :** Le sens sémantique d'un mot est souvent défini par ses associations avec d'autres mots. Par exemple, le mot "océan" peut être associé à "vagues", "plage" et "eau salée".

Le processus d'incorporation de mots (word embedding) dans le NLP vise à capturer ces sens sémantiques en représentant les mots sous forme de vecteurs continus dans un espace multidimensionnel. Dans cet espace vectoriel, les mots ayant des sens similaires sont rapprochés, permettant aux modèles informatiques de comprendre les relations entre les mots et même d'effectuer des tâches telles que les analogies entre mots.

>[!Note]
>La taille des vecteurs de l'embedding représente sa dimensionnalité. On peut voir l'embedding comme un moyen de réduire la dimensionalité d'un vecteur représentant un mot.
>
>Dans le vocabulaire, l'indice $0$ est réservé pour le token de padding et l'indice $1$ pour le token représentant les mots inconnus. De ce fait en créant la couche d'embedding on peut ajouter **padding_idx=0**.

<u>NB</u>: La méthode **vocab.set_default_index** permet d'indiquer le token à retourner en cas de token *OOV(Out Of Vocabulary)* donc les tokens inconnus.

Plutôt que d'avoir des vecteurs de taille fixe que l'on ajuste avec le padding, on peut effectuer le calcul de l'embedding malgré la variation de taille de chaque séquence grâce à un **EmbeddingBag**[^3]. Il s'agit là d'une méthode qui opère plus vite et optimise l'utilisation de la mémoire.

### Principe de l'EmbeddingBag
Après avoir encodé chaque séquence, on calcule la taille des différents vecteurs.
Grâce aux différentes tailles, on peut effectuer une *somme cumulée croissante* afin de déterminer le décalage de chaque séquence dans le tenseur de sortie.

En fait après l'encodage, on fait une concaténation des différents vecteurs encodés sur `dim=0` afin d'avoir un *vecteur ligne* à la fin. Avec un vecteur et grâce aux la liste des cumsum, alors on saura où commence chaque séquence dans le tenseur de séquences concaténées.
L'avantage ici est qu'on ne prend plus la peine d'effectuer le padding comme avec un embedding normal.

```py
def offsetify(batch: tuple[int, str]):

    x = [torch.tensor(encode(sample[1])) for sample in batch] # encodage des séquences
    
    offset = [0] + [len(elt) for elt in x] # calcule de la taille de chaque vecteur

    offset = torch.tensor(offset[:-1]).cumsum(dim=0) # somme cumulée croissante des tailles de vecteur

    return (
        torch.LongTensor([sample[0] - 1 for sample in batch]), # ajustement des labels du batch
        torch.cat(x), # concatenation pour avoir un vecteur de taille (1, N)
        offset, # vecteur des offsets
    )
```


### nn.Embedding vs nn.EmbeddingBag
`nn.Embedding` and `nn.EmbeddingBag` serve different purposes and are suitable for different types of tasks in natural language processing (NLP) and related fields. Here's when you should prefer one over the other:

1. Use `nn.Embedding`:
   - **Individual Token Embeddings**: When you need to obtain embeddings for individual tokens (words or subwords) in a sequence, `nn.Embedding` is the preferred choice. Each token is associated with a unique embedding vector.
   - **Sequential Models**: If you're building sequential models like recurrent neural networks (RNNs), transformers, or convolutional neural networks (CNNs) that require token-level embeddings, `nn.Embedding` is the way to go.
   - **Fixed Sequence Length**: When your input sequences have fixed lengths and you want to maintain token-level embeddings for each position in the sequence.

   Example: Text classification, machine translation, text generation.

2. Use `nn.EmbeddingBag`:
   - **Aggregated Sequence Embeddings**: When you want to obtain a fixed-size, aggregated representation of variable-length sequences (bags) of tokens. It's particularly useful for tasks where input sequences have different lengths, and you want a single vector representation for each sequence.
   - **Text Classification**: In text classification tasks, you can use `nn.EmbeddingBag` to average or sum the embeddings of the tokens in a sentence to get a sentence-level embedding.
   - **Efficiency**: When you want to efficiently compute embeddings for sequences of variable lengths without padding. `nn.EmbeddingBag` automatically handles sequences of different lengths and can be more memory-efficient than padding and using `nn.Embedding`.

   Example: Text classification (e.g., sentiment analysis), document classification, document retrieval.

In summary, `nn.Embedding` is suitable for tasks that require individual token embeddings, while `nn.EmbeddingBag` is suitable for tasks where you need aggregated, fixed-size representations of variable-length sequences. The choice between them depends on the specific requirements of your NLP task.


## 4 - Capture patterns with recurrent neural networks
Non pas que la combinaison d'embedding et de couches linéaires est mauvaise mais le problème de cette association est qu'elle ne prend pas en compte *l'ordre* des mots dans la séquence. Les embeddings créent des aggrégations de données du coup on perd l'avantage de l'ordre. Pour pallier à cela, on utilise des couches [[Recurrent Neural Network| RNN]].

Le problème majeure des RNN est qu'ils ont les gradients qui disparaissent lors de la rétropropagation vu qu'il faut qu'ils traitent toute la séquence avant que le processus ne continue, du coup à la longue les gradients deviennent trop petits.
Pour pallier à cela,  on introduit un système de gestion d'état explicite à travers les ***gates***. C'est le mécanisme implémenté dans les [[Long Short Term Memory | LSTM]] et les [[Gated Relay Unit | GRU]].













[^1]: [ShardingFilter — TorchData main documentation (pytorch.org)](https://pytorch.org/data/main/generated/torchdata.datapipes.iter.ShardingFilter.html)

[^2]: [Embedding — PyTorch 2.0 documentation](https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html#torch.nn.Embedding)

[^3]: [EmbeddingBag — PyTorch 2.0 documentation](https://pytorch.org/docs/stable/generated/torch.nn.EmbeddingBag.html)