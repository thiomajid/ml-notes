![[design-choices-for-language.png]]

To go from source code to binary instructions that a computer can understand, the following steps are to be followed:

## Scanning
Also called **lexing** or **lexical analysis**, the scanning step is the first process through which source code goes through.

>[!note]
>"Lexical" comes from the Greek root *lex*, which means "word".
>

A lexer transforms a **linear** stream of characters into chunks of words called **tokens**. The lexer discards the *whitespace* character.

Given this sequence of characters: `var average = (min + max) / 2;`, a lexer will produce the following tokens:
![[Pasted image 20240627231319.png]]

## Parsing
