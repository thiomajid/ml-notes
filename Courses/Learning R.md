>[!info]
>In R Studio, to run the whole script type: alt + ctrl + R


## Chapitre 1 - Intro
En R, il existe pas ma de fonctions prédéfinies pour faire le job. Plus particulièrement, tout tourne autour de notions mathématiques. L'affectation de valeur se fait en utilisant l'opérateur `<-`.

R supporte le style de *lexical scoping* offert par Rust dans lequel, la dernière instruction d'une expression en est la valeur retournée.
On peut également déclarer un bloc d'instructions à l'intérieur d'un autre bloc.

```R
{
	a <- 2 + 3
	b <- a
	b
}
```

L'expression ci-dessus retourne la valeur à laquelle `b` est évaluée.