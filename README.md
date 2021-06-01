# Negative Language Transfer Identification

## Data

### Training data

#### Chinese and English
- [Global Voices Parallel Corpus 2018Q4 (zhs-en)](http://casmacat.eu/corpus/global-voices.html)
- [WMT-News 2019 version](http://opus.nlpl.eu/WMT-News-v2019.php)

#### Spanish and English
- [1000 sentences](https://www.kaggle.com/bryanpark/parallelsents)
- [Language Identification dataset](https://www.kaggle.com/zarajamshaid/language-identification-datasst)
- [European Parliament Proceedings Parallel Corpus 1996-2011](https://www.kaggle.com/djonafegnem/europarl-parallel-corpus-19962011)
- [United Nations](http://opus.nlpl.eu/UN-v20090831.php)

### Test data (Chinese and English)
- [Negative language transfer dataset](https://github.com/EdTeKLA/LanguageTransfer)

## Language modelling
Two language modelling techniques were used to create models. To learn more about the training, tuning, and testing procedures used, refer to their descriptions.
* [N-gram baseline](./n_gram_pipeline.md)
* [RNN approach](./rnn_pipeline.md)

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