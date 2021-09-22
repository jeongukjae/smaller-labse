# smaller-LaBSE

LaBSE(Language-agnostic BERT Sentence Embedding) is a very good method to get sentence embeddings across languages. But it is hard to fine-tune due to the parameter size(~=471M) of this model. For instance, if I fine-tune this model with Adam optimizer, I need the GPU that has VRAM at least 7.5GB = 471M * (parameters 4 bytes + gradients 4 bytes + momentums 4 bytes + variances 4 bytes). So I applied "Load What You Need: Smaller Multilingual Transformers" method to LaBSE to reduce parameter size since most of this model's parameter is the word embedding table(\~=385M).

The smaller version of LaBSE is evaluated for 14 languages using tatoeba dataset. It shows we can reduce LaBSE's parameters to 47% without a big performance drop.

***If you need the PyTorch version, see <https://github.com/Geotrend-research/smaller-transformers>. I followed most of the steps in the paper.***

| Model        | #param(transformer) | #param(word embedding) | #param(model) | vocab size |
| ------------ | ------------------: | ---------------------: | ------------: | ---------: |
| tfhub_LaBSE  |               85.1M |                 384.9M |        470.9M |    501,153 |
| 15lang_LaBSE |               85.1M |                 133.1M |        219.2M |    173,347 |

## Usage

You can use this model directly via [tfhub.dev/jeongukjae/smaller_LaBSE_15lang/1](https://tfhub.dev/jeongukjae/smaller_LaBSE_15lang/1).

```python
import tensorflow as tf
import tensorflow_text  # noqa
import tensorflow_hub as hub

# Loading models from tfhub.dev
encoder = hub.KerasLayer("https://tfhub.dev/jeongukjae/smaller_LaBSE_15lang/1")
preprocessor = hub.KerasLayer("https://tfhub.dev/jeongukjae/smaller_LaBSE_15lang_preprocess/1")

# Constructing model to encode texts into high-dimensional vectors
sentences = tf.keras.layers.Input(shape=(), dtype=tf.string, name="sentences")
encoder_inputs = preprocessor(sentences)
sentence_representation = encoder(encoder_inputs)["pooled_output"]
normalized_sentence_representation = tf.nn.l2_normalize(sentence_representation, axis=-1)  # for cosine similarity
model = tf.keras.Model(sentences, normalized_sentence_representation)
model.summary()

# Encoding multilingual sentences.
english_sentences = tf.constant(["dog", "Puppies are nice.", "I enjoy taking long walks along the beach with my dog."])
italian_sentences = tf.constant(["cane", "I cuccioli sono carini.", "Mi piace fare lunghe passeggiate lungo la spiaggia con il mio cane."])
japanese_sentences = tf.constant(["犬", "子犬はいいです", "私は犬と一緒にビーチを散歩するのが好きです"])

english_embeds = model(english_sentences)
italian_embeds = model(italian_sentences)
japanese_embeds = model(japanese_sentences)

# English-Italian similarity
print(tf.tensordot(english_embeds, italian_embeds, axes=[[1], [1]]))

# English-Japanese similarity
print(tf.tensordot(english_embeds, japanese_embeds, axes=[[1], [1]]))

# Italian-Japanese similarity
print(tf.tensordot(italian_embeds, japanese_embeds, axes=[[1], [1]]))
```

In addition, you can evaluate this model with Tatoeba dataset directly in [this colab link](https://colab.research.google.com/drive/1eby9SELCxp7ZOispa4WOHNcuSUsYHOL8?usp=sharing).

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

I selected the languages multilingual-USE supports.

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
# evaluate TFHub LaBSE
./evaluate_tatoeba.sh
# evaluate the smaller LaBSE
./evaluate_tatoeba.sh \
    --model models/LaBSE_en-fr-es-de-zh-ar-zh_classical-it-ja-ko-nl-pl-pt-th-tr-ru/1/ \
    --preprocess models/LaBSE_en-fr-es-de-zh-ar-zh_classical-it-ja-ko-nl-pl-pt-th-tr-ru_preprocess/1/
```

## Results

### Tatoeba

| Model               |    fr |    es |    de |    zh |    ar |    it |    ja |    ko |    nl |    pl |    pt |    th |    tr |    ru |   avg |
| ------------------- | ----: | ----: | ----: | ----: | ----: | ----: | ----: | ----: | ----: | ----: | ----: | ----: | ----: | ----: | ----: |
| tfHub_LaBSE(en→xx)  | 95.90 | 98.10 | 99.30 | 96.10 | 90.70 | 95.30 | 96.40 | 94.10 | 97.50 | 97.90 | 95.70 | 82.85 | 98.30 | 95.30 | 95.25 |
| tfHub_LaBSE(xx→en)  | 96.00 | 98.80 | 99.40 | 96.30 | 91.20 | 94.00 | 96.50 | 92.90 | 97.00 | 97.80 | 95.40 | 83.58 | 98.50 | 95.30 | 95.19 |
| 15lang_LaBSE(en→xx) | 95.20 | 98.00 | 99.20 | 96.10 | 90.50 | 95.20 | 96.30 | 93.50 | 97.50 | 97.90 | 95.80 | 82.85 | 98.30 | 95.40 | 95.13 |
| 15lang_LaBSE(xx→en) | 95.40 | 98.70 | 99.40 | 96.30 | 91.10 | 94.00 | 96.30 | 92.70 | 96.70 | 97.80 | 95.40 | 83.58 | 98.50 | 95.20 | 95.08 |

- *Accuracy(%) of the Tatoeba datasets.*
- *If the strategy to select vocabs is changed or the corpus used in the selection step is changed to the corpus similar to the evaluation dataset, it is expected to reduce the performance drop.*

## References

- Load What You Need: Smaller Versions of Multilingual BERT (Paper: <https://arxiv.org/abs/2010.05609>, GitHub: <https://github.com/Geotrend-research/smaller-transformers>)
- Language-agnostic BERT Sentence Embedding: <https://arxiv.org/abs/2007.01852>
- TFHub - LaBSE: <https://tfhub.dev/google/LaBSE/2>
- LaBSE blog post: <https://ai.googleblog.com/2020/08/language-agnostic-bert-sentence.html>
- Massively Multilingual Sentence Embeddings for Zero-Shot Cross-Lingual Transfer and Beyond: <https://arxiv.org/abs/1812.10464>
