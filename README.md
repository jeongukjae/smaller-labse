# smaller-LaBSE

LaBSE(Language-agnostic BERT Sentence Embedding) is a very good method to get sentence embeddings across languages. But it is hard to fine-tune due to the parameter size(~=471M) of this model. For instance, if I fine-tune this model with Adam optimizer, I need the GPU that has VRAM at least 7.5GB = 471M * (parameters 4 bytes + gradients 4 bytes + momentums 4 bytes + variances 4 bytes).

So I applied "Load What You Need: Smaller Multilingual Transformers" method to LaBSE to reduce parameter size since most of this model's parameter is the word embedding table(~=385M). The smaller version of LaBSE is evaluated for 14 languages using tatoeba dataset. It shows we can reduce LaBSE's parameters to 47% without a big performance drop.

***If you need the PyTorch version, see <https://github.com/Geotrend-research/smaller-transformers>. I followed most of the steps in the paper.***

| Model name                   | #param(transformer) | #param(word embedding) | #param(model) | vocab size |
| ---------------------------- | ------------------: | ---------------------: | ------------: | ---------: |
| LaBSE from tfhub (version 2) |               85.1M |                 384.9M |        470.9M |    501,153 |
| Smaller LaBSE                |               85.1M |                 133.1M |        219.2M |    173,347 |

## Used Languages

- English (`en` or `eng`)
- French (`fr` or `fra`)
- Spanish (`es` or `spa`)
- German (`de` or `deu`)
- Chinese (`zh`, `zh_classical` or `cmn`)
- Arabic (`ar` or `ara`)
- Italian (`it` or `ita`)
- Japanese (`ja` or `jpn`)
- Korean (`ko` or `kor`)
- Dutch (`nl` or `nld`)
- Polish (`pl` or `pol`)
- Portuguese (`pt` or `por`)
- Thai (`th` or `tha`)
- Turkish (`tr` or `tur`)
- Russian (`ru` or `rus`)

## Scripts

A smaller version of the vocab was constructed based on the frequency of tokens using Wikipedia dump data. I followed most of the algorithms in the paper to extract proper vocab for each language and rewrite it for TensorFlow.

### Convert weight

```sh
mkdir -p downloads/labse-2
curl -L https://tfhub.dev/google/LaBSE/2?tf-hub-format=compressed -o downloads/labse-2.tar.gz
tar -xf downloads/labse-2.tar.gz -C downloads/labse-2/
python save_as_weight_from_saved_model.py
```

### Select vocabs

```sh
./download_dataset.sh
python select_vocab.py
```

### Make smaller LaBSE

```sh
./make_smaller_labse.py
```

### Evaluate tatoeba

```sh
./download_tatoeba_dataset.sh
python evaluate_tatoeba.py --model HUB_STYLE_PATH --preprocess HUB_STYLE_PATH --lang spa
```

## Results

### Tatoeba dataset

| Model Name                 |    fr |    es |    de |    zh |    ar |    it |    ja |    ko |    nl |    pl |    pt |    th |    tr |    ru |   avg |
| -------------------------- | ----: | ----: | ----: | ----: | ----: | ----: | ----: | ----: | ----: | ----: | ----: | ----: | ----: | ----: | ----: |
| LaBSE from tf hub (en->xx) | 95.90 | 98.10 | 99.30 | 96.10 | 90.70 | 95.30 | 96.40 | 94.10 | 97.50 | 97.90 | 95.70 | 82.85 | 98.30 | 95.30 | 95.25 |
| LaBSE from tf hub (xx->en) | 96.00 | 98.80 | 99.40 | 96.30 | 91.20 | 94.00 | 96.50 | 92.90 | 97.00 | 97.80 | 95.40 | 83.58 | 98.50 | 95.30 | 95.19 |
| LaBSE for 14 lang (en->xx) | 95.20 | 98.00 | 99.20 | 96.10 | 90.50 | 95.20 | 96.30 | 93.50 | 97.50 | 97.90 | 95.80 | 82.85 | 98.30 | 95.40 | 95.13 |
| LaBSE for 14 lang (xx->en) | 95.40 | 98.70 | 99.40 | 96.30 | 91.10 | 94.00 | 96.30 | 92.70 | 96.70 | 97.80 | 95.40 | 83.58 | 98.50 | 95.20 | 95.08 |

*If the strategy to select vocabs is changed or the corpus used in the selection step is changed to the corpus similar to the evaluation dataset, it is expected to reduce the performance drop.*

## References

- Load What You Need: Smaller Versions of Multilingual BERT (Paper: <https://arxiv.org/abs/2010.05609>, GitHub: <https://github.com/Geotrend-research/smaller-transformers>)
- Language-agnostic BERT Sentence Embedding: <https://arxiv.org/abs/2007.01852>
- TFHub - LaBSE: <https://tfhub.dev/google/LaBSE/2>
- LaBSE blog post: <https://ai.googleblog.com/2020/08/language-agnostic-bert-sentence.html>
