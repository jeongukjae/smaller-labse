# smaller-labse

## Langs

- `en` English
- `fr` French
- `es` Spanish
- `de` German
- `zh` Chinese-simplified
- `ar` Arabic
- `zh-classical` Chinese-traditional
- `it` Italian
- `ja` Japanese
- `ko` Korean
- `nl` Dutch
- `pl` Polish
- `pt` Portuguese
- `th` Thai
- `tr` Turkish
- `ru` Russian

## Scripts

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

## References

- <https://arxiv.org/abs/2010.05609>
- <https://arxiv.org/abs/2007.01852>
- <https://tfhub.dev/google/LaBSE/2>
- <https://github.com/Geotrend-research/smaller-transformers>
- <https://ai.googleblog.com/2020/08/language-agnostic-bert-sentence.html>
