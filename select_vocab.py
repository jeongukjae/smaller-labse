import json
import os
from collections import defaultdict

import tensorflow as tf
import tensorflow_text as text
from tqdm import tqdm

tokenizer = text.BertTokenizer("./downloads/labse-2/assets/cased_vocab.txt", token_out_type=tf.string)
languages = "en,fr,es,de,zh,ar,zh_classical,it,ja,ko,nl,pl,pt,th,tr,ru".split(",")

os.makedirs("data", exist_ok=True)
os.makedirs("tokens_freqs", exist_ok=True)
os.makedirs("selected_tokens", exist_ok=True)


for lang in languages:
    filename = f"data/{lang}.txt"
    print(f"select tokens for {lang}")

    num_long_lines = 0

    lang_tokens = defaultdict(lambda: 0)
    lang_tokens_unique = defaultdict(lambda: 0)

    ds = (
        tf.data.TextLineDataset([filename])
        .filter(lambda x: tf.strings.length(x, unit="UTF8_CHAR") > 5)
        .batch(1000)
        .map(lambda x: tokenizer.tokenize(x), num_parallel_calls=tf.data.AUTOTUNE)
        .prefetch(tf.data.AUTOTUNE)
        .take(1000)  # max: 1M sentence
    )

    for token_tensor in tqdm(ds):
        num_long_lines += token_tensor.nrows().numpy().item()
        for line in token_tensor:
            tokens = [token.decode('utf8') for token in line.flat_values.numpy()]
            for token in tokens:
                lang_tokens[token] += 1
            for token in list(set(tokens)):
                lang_tokens_unique[token] += 1

    print(f"# long lines: {num_long_lines}")

    # save frequencies
    with open('tokens_freqs/'+lang+'_freqs.json', 'w') as outfile:
        json.dump(lang_tokens, outfile)
    seuil = int(num_long_lines*0.005/100)
    num_selected_tokens = 0
    with open('selected_tokens/selected_'+lang+'_tokens.txt', 'w') as output:
        for tok in lang_tokens_unique:
            if lang_tokens_unique[tok] >= seuil:
                output.write(tok+'\n')
                num_selected_tokens += 1
