# Negative Language Transfer Detection

### Datasets used to train language models

#### Chinese and English
- [Global Voices Parallel Corpus 2018Q4 (zhs-en)](http://casmacat.eu/corpus/global-voices.html)
- [WMT-News 2019 version](http://opus.nlpl.eu/WMT-News-v2019.php)

#### Spanish and English
- [1000 sentences](https://www.kaggle.com/bryanpark/parallelsents)
- [Language Identification dataset](https://www.kaggle.com/zarajamshaid/language-identification-datasst)
- [European Parliament Proceedings Parallel Corpus 1996-2011](https://www.kaggle.com/djonafegnem/europarl-parallel-corpus-19962011)
- [United Nations](http://opus.nlpl.eu/UN-v20090831.php)

### Datasets used to test language models
- [CLC FCE dataset](https://ilexir.co.uk/datasets/index.html)

### Spanish to English part-of-speech tags mapping
Mapping between [Spacy's Spanish](https://spacy.io/models/es#es_core_news_md) part-of-speech tags, based on the [AnCora corpus](https://github.com/UniversalDependencies/UD_Spanish-AnCora), to [Spacy's English](https://spacy.io/models/en#en_core_web_md) part-of-speech tags, based on the [Penn Treebank](https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html).

## Reference
Leticia Farias Wanderley and Carrie Demmans Epp. [Identifying negative language transfer in learner errors using POS information](https://www.aclweb.org/anthology/2021.bea-1.7/). In Proceedings of the 16th Workshop on Innovative Use of NLP for Building Educational Applications, pages 64–74, Online, April 2021. Association for Computational Linguistics.

```
@inproceedings{farias-wanderley-demmans-epp-2021-identifying,
    title = "Identifying negative language transfer in learner errors using {POS} information",
    author = "Farias Wanderley, Leticia  and
      Demmans Epp, Carrie",
    booktitle = "Proceedings of the 16th Workshop on Innovative Use of NLP for Building Educational Applications",
    month = apr,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2021.bea-1.7",
    pages = "64--74"
}
```