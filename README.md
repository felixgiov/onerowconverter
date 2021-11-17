# onerowconverter

Imagine that we have a keyboard with only one row and a space bar. 
That one row is the number row that usually located on the top row of a keyboard. 
Therefore, there are only 13 characters, including space, that we can use for typing.
But surely, we can’t construct proper sentences just by using these 13 characters. 
We need to represent all the remaining letters and symbols into these characters. 
So, when we type a certain character that not in one row keyboard, we look at its representation in the number row from the same column. 
For example, for letter ‘a’ we type ‘1’ and for letter ‘H’ we type ‘6’.

The goal is to build a model that can decode the sequences that we type using one row keyboard into a proper English words and sentences that consist of all characters available in a normal keyboard.
For example, given a string from one row keyboard as the input:
`5682 1653433 563 8865397 94 234581 163 852 0154968 563 016-291483 163 94569392 4722816 3708439`.
We want to output this sentence:
`This angered the Kingdom of Serbia and its patron, the Pan-Slavic and Orthodox Russian Empire.`


We collected 2 articles from English Wikipedia and used a simple Bidirectional LSTM as the model. 
We then splitted them into training data and evaluation data, with 80% for training data (1512 sentences) and 20% for evaluation data (378 sentences).
We showed that the model achieved accuracies of 0.95 and 0.80, for character-level and word-level respectively.
