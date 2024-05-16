## BLEU
#BLEU est un métrique utiliser dans les tâches de #machine-translation .
Il assigne un score à une traduction afin de savoir la pertinence de cette dernière comparée à une ou plusieurs autres traductions.

BLEU compare les #n-gram du texte généré et ceux du texte de la référence. Par exemple des #uni-gram cela revient à faire le calcul:

$$
\text{Unigram precision} = \frac{\text{Number of word matches}}{\text{Number of words in generation}}
$$

Utiliser un uni-gram n'est pas fiable car le modèle peut répéter un des mots correctes plusieurs fois, ce qui peut conduire à une grande précision, or $precision \in [0, 1]$.
![[Pasted image 20231116104136.png]]

To handle this, BLEU uses a modified precision that **clips** the number of times to count a word, based on how many times it appears in the reference text. Thus the formula becomes:

$$
\text{Unigram precision} = \frac{\text{clip(Number of word matches)}}{\text{Number of words in generation}}
$$

Un autre problème est qu'en l'état actuel des choses, BLEU ne tient pas compte de l'ordre des mots. Pour ce faire, BLEU calcule la précision pour différent #n-gram et fait la moyenne de toutes ces dernières.

>[!Info]
>BLEU est en faite une moyenne géométrique des précisions des différents #n-gram 

### Avantages et inconvénients
![[Pasted image 20231116105124.png]]

Pour résoudre le problème lié au fait d'utiliser différent #tokenizer, il est recommandé d'utiliser la métrique #SacreBLEU.


## ROUGE
#ROUGE est une métrique utiliser pour mesurer la performance d'un modèle de #text-summarization. L'idée est de mesurer la qualité du résumé généré par rapport à des références.

L'approche utiliser par #ROUGE est de comparer les #n-gram du texte généré à ceux de la référence. On calcule le #recall de ROUGE avec le quotient *number of word matches* sur *number of words in reference*.

>[!Info]
>Lorsque ROUGE utilise des uni-grams, on l'appelle ROUGE-1.

Lorsque dans le résumé généré le même mot se répète n-fois alors on peut avoir un grand #recall mais ça ne traduit pas un bon résultat. Pour pallier à cela, on calcule également la #precision comme ça on aura également le #F1-Score.

$$
\begin{align}
\text{ROUGE-1}_{recall} = \frac{\text{number of word matches}}{\text{number of words in reference}} \\ \\

\text{ROUGE-1}_{precision} = \frac{\text{number of word matches}}{\text{number of words in summary}} \\ \\

\text{ROUGE-1}_{f1-score} = 2  (\frac{\text{precision $\cdot$ recall}}{\text{precision + recall}})
\end{align}
$$

On a également une variante appelée #ROUGE-L qui au lieu de traiter des #n-gram, traite toute la phrase comme une séquence et cherche **longest common sub-sequence** aussi notée #LCS.

$$
\begin{align}
\text{ROUGE-L}_{recall} = \frac{\text{LCS(gen, ref)}}{\text{number of words in reference}} \\ \\

\text{ROUGE-L}_{precision} = \frac{\text{LCS(gen, ref)}}{\text{number of words in summary}} \\ \\
\end{align}
$$

## Perplexity
#perplexity aussi noté #PPL 
