import json
import re
import tensorflow as tf
import tensorflow_text as text
import tensorflow_hub as hub
from official.nlp.modeling import networks
from official.nlp.tools import export_tfhub_lib
from absl import app, flags
from tqdm import tqdm
from save_as_weight_from_saved_model import create_model

FLAGS = flags.FLAGS
flags.DEFINE_string("langs", "en,fr,es,de,zh,ar,zh_classical,it,ja,ko,nl,pl,pt,th,tr,ru", help='language to use')
flags.DEFINE_integer("seq_len", 128, "default seq len")

def main(argv):
    print("langs:", FLAGS.langs)

    with open("./downloads/labse-2/assets/cased_vocab.txt") as f:
        original_tokens = [l.strip() for l in f.readlines()]
    kept_tokens = [token for token in original_tokens if re.match(r"^\[.+\]$", token)]
    print("Kept Token:", kept_tokens)

    tokens_for_langs = set()
    for lang in FLAGS.langs.split(","):
        with open(f"./selected_tokens/selected_{lang}_tokens.txt") as f:
            tokens_for_langs.update([l.strip() for l in f.readlines() if l])
    tokens_for_langs.remove("")
    tokens_for_langs.remove("[UNK]")
    target_vocab = kept_tokens + sorted(list(tokens_for_langs))
    print("# vocab:", len(target_vocab))
    vocab_file_path = f"./models/vocab-{'-'.join(FLAGS.langs.split(','))}.txt"
    with open(vocab_file_path, "w") as f:
        for token in target_vocab:
            print(token, file=f)
    preprocessing_path = f"./models/LaBSE_{'-'.join(FLAGS.langs.split(','))}_preprocess/1"
    export_tfhub_lib.export_preprocessing(
        export_path=preprocessing_path,
        vocab_file=vocab_file_path,
        do_lower_case=False,
        tokenize_with_offsets=False,
        default_seq_length=FLAGS.seq_len,
    )

    token_indices = {token: index for index, token in enumerate(original_tokens)}
    target_index = [[token_indices[t]] for t in tqdm(target_vocab)]
    assert all([original_tokens[val[0]] == target_vocab[index] for index, val in enumerate(target_index)])

    # original LaBSE
    sentences = tf.constant([
        "LaBSE 성능이 기대됩니다!",
        "네??",
        "Hi, there",
    ])
    labse_model, labse_encoder = create_model(load_encoder_weight=True, load_model_weight=False)
    preprocessor = hub.KerasLayer("https://tfhub.dev/google/universal-sentence-encoder-cmlm/multilingual-preprocess/2")
    original_repr = labse_model(preprocessor(sentences))["pooled_output"]

    # smaller LaBSE
    smaller_labse_inputs = {
        "input_mask": tf.keras.Input([None], dtype=tf.int32),
        "input_type_ids": tf.keras.Input([None], dtype=tf.int32),
        "input_word_ids": tf.keras.Input([None], dtype=tf.int32),
    }
    smaller_labse_encoder: networks.BertEncoder = networks.BertEncoder.from_config(
        {
            "vocab_size": len(target_vocab),
            "hidden_size": 768,
            "num_layers": 12,
            "num_attention_heads": 12,
            "max_sequence_length": 512,
            "type_vocab_size": 2,
            "output_range": None,
            "embedding_width": 768,
            "embedding_layer": None,
            "activation": "Text>gelu",
            "intermediate_size": 3072,
            "dropout_rate": 0.1,
            "attention_dropout_rate": 0.1,
            "dict_outputs": True,
            "return_all_encoder_outputs": False,
        }
    )
    smaller_labse_encoder.summary()

    smaller_labse_encoder._embedding_layer.set_weights([tf.gather_nd(labse_encoder.get_embedding_table(), tf.constant(target_index))])
    smaller_labse_encoder._position_embedding_layer.set_weights(labse_encoder._position_embedding_layer.get_weights())
    smaller_labse_encoder._type_embedding_layer.set_weights(labse_encoder._type_embedding_layer.get_weights())
    smaller_labse_encoder._embedding_norm_layer.set_weights(labse_encoder._embedding_norm_layer.get_weights())
    smaller_labse_encoder.pooler_layer.set_weights(labse_encoder.pooler_layer.get_weights())

    for target, source in zip(smaller_labse_encoder.transformer_layers, labse_encoder.transformer_layers):
        target.set_weights(source.get_weights())

    smaller_labse_outputs = smaller_labse_encoder(smaller_labse_inputs)
    smaller_labse_model = tf.keras.Model(smaller_labse_inputs, smaller_labse_outputs)
    smaller_labse_model.summary()
    smaller_labse_model.save_weights(f"./models/LaBSE_{'-'.join(FLAGS.langs.split(','))}")
    smaller_labse_model.save(f"./models/LaBSE_{'-'.join(FLAGS.langs.split(','))}/1")
    smaller_preprocessor = hub.KerasLayer(preprocessing_path)

    new_repr = smaller_labse_encoder(smaller_preprocessor(sentences))["pooled_output"]
    print(original_repr, new_repr)
    print("OriginalTokens:", [[original_tokens[id_] for id_ in instance if id_ != 0] for instance in preprocessor(sentences)['input_word_ids']])
    print("NewTokens:", [[target_vocab[id_] for id_ in instance  if id_ != 0] for instance in smaller_preprocessor(sentences)['input_word_ids']])
    tf.debugging.assert_near(original_repr, new_repr)


if __name__ == '__main__':
    app.run(main)
