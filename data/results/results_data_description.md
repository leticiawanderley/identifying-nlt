## Results file data description

Each line in this dataset represents one annotated learner error.

| Column | Type   | Description |
|--------|:------:|-------------|
| error_type | String | Code associated with the error, see [Nicholls 2003](http://ucrel.lancs.ac.uk/publications/CL2003/papers/nicholls.pdf) |
| Negative transfer? | Boolean | Gold standard negative language transfer label |
| Likely reason for mistake | String | Possible error cause |
| incorrect_ud_tags | String | Universal Dependencies part-of-speech tags extracted from the words in the error |
| incorrect_ud_tags_padded | String | Universal Dependencies part-of-speech tags extracted from one word before the error, the words in the error, and one word after the error |
| incorrect_ud_tags_unigram | String | Universal Dependencies part-of-speech tags extracted from the words in the error plus one word after the error |
| incorrect_ud_tags_bigram | String | Universal Dependencies part-of-speech tags extracted from the words in the error plus two words after the error |
| en | Float | Probability of part-of-speech sequence belonging to English |
| zhs | Float | Probability of part-of-speech sequence belonging to Chinese |
| nlt | Boolean | Negative language transfer label assigned by the model |
| result | Boolean | Whether the model assigned negative language transfer label matches the gold standard |
