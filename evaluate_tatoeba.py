import tensorflow as tf
import tensorflow_text as text
import tensorflow_hub as hub
from absl import app, flags, logging
from tqdm import tqdm
from save_as_weight_from_saved_model import create_model

FLAGS = flags.FLAGS
flags.DEFINE_string("lang", "ara", help='language to use')
flags.DEFINE_string("model", "./downloads/labse-2", help='model path')
flags.DEFINE_string("preprocess", "https://tfhub.dev/google/universal-sentence-encoder-cmlm/multilingual-preprocess/2", help='language to use')


def main(argv):
    ds = tf.data.Dataset.zip((
        tf.data.TextLineDataset([f"./data/tatoeba/v1/tatoeba.{FLAGS.lang}-eng.eng"]).batch(32),
        tf.data.TextLineDataset([f"./data/tatoeba/v1/tatoeba.{FLAGS.lang}-eng.{FLAGS.lang}"]).batch(32),
    ))
    logging.info(f"Element spec: {ds.element_spec}")

    preprocessor = hub.KerasLayer(FLAGS.preprocess)
    labse_encoder = hub.KerasLayer(FLAGS.model)

    inputs = tf.keras.Input([], dtype=tf.string)
    model_inputs = preprocessor(inputs)
    model_outputs = labse_encoder(model_inputs)
    if 'default' in model_outputs:
        output_node = tf.nn.l2_normalize(model_outputs["default"], axis=-1)  # for cosine similarity
    else:
        output_node = tf.nn.l2_normalize(model_outputs['pooled_output'], axis=-1)  # for cosine similarity
    model = tf.keras.Model(inputs, output_node)
    model.summary()

    src_embeddings = []
    tgt_embeddings = []
    for src, tgt in tqdm(ds):
        src_embeddings.append(model(src))
        tgt_embeddings.append(model(tgt))
    src_embeddings = tf.concat(src_embeddings, axis=0)
    tgt_embeddings = tf.concat(tgt_embeddings, axis=0)

    similarities_src_to_tgt = tf.tensordot(src_embeddings, tgt_embeddings, axes=[[1], [1]])
    answer = tf.range(tf.shape(src_embeddings)[0], dtype=tf.int64)
    src_to_tgt_acc = tf.math.count_nonzero(tf.argmax(similarities_src_to_tgt, axis=-1) == answer) / tf.size(answer, tf.int64)
    tgt_to_src_acc = tf.math.count_nonzero(tf.argmax(tf.transpose(similarities_src_to_tgt, perm=[1, 0]), axis=-1) == answer) / tf.size(answer, tf.int64)
    print(f"eng->{FLAGS.lang}:", src_to_tgt_acc)
    print(f"{FLAGS.lang}->eng:", tgt_to_src_acc)


if __name__ == '__main__':
    app.run(main)
