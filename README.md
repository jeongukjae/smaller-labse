# smaller-labse

* <https://arxiv.org/abs/2010.05609>
* <https://arxiv.org/abs/2007.01852>
* <https://tfhub.dev/google/LaBSE/2>
* <https://github.com/Geotrend-research/smaller-transformers>

## Langs

* `en` English
* `fr` French
* `es` Spanish
* `de` German
* `zh` Chinese-simplified
* `ar` Arabic
* `zh-classical` Chinese-traditional
* `it` Italian
* `ja` Japanese
* `ko` Korean
* `nl` Dutch
* `pl` Polish
* `pt` Portuguese
* `th` Thai
* `tr` Turkish
* `ru` Russia

## Converting weight

```sh
mkdir -p downloads/labse-2
curl -L https://tfhub.dev/google/LaBSE/2?tf-hub-format=compressed -o downloads/labse-2.tar.gz
tar -xf downloads/labse-2.tar.gz -C downloads/labse-2/
python save_as_weight_from_saved_model.py
```

## Selecting vocabs

```sh
./download_dataset.sh
python select_vocab.py
```
