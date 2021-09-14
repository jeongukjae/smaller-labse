import json
import tensorflow as tf
import tensorflow_text as text
import tensorflow_hub as hub
from official.nlp.modeling import networks


def create_model():
    inputs = {
        "input_mask": tf.keras.Input([None], dtype=tf.int32),
        "input_type_ids": tf.keras.Input([None], dtype=tf.int32),
        "input_word_ids": tf.keras.Input([None], dtype=tf.int32),
    }
    bert_encoder = networks.BertEncoder.from_config(
        json.loads(
            """
    {
        "vocab_size": 501153,
        "hidden_size": 768,
        "num_layers": 12,
        "num_attention_heads": 12,
        "max_sequence_length": 512,
        "type_vocab_size": 2,
        "output_range": null,
        "embedding_width": 768,
        "embedding_layer": null,
        "activation": "Text>gelu",
        "intermediate_size": 3072,
        "dropout_rate": 0.1,
        "attention_dropout_rate": 0.1,
        "dict_outputs": true,
        "return_all_encoder_outputs": false
    }
    """
        )
    )
    outputs = bert_encoder(inputs)
    model = tf.keras.Model(inputs, outputs)
    model.summary()

    model.load_weights("./downloads/labse-2")
    return model, bert_encoder


english_sentences = tf.constant(["dog", "Puppies are nice.", "I enjoy taking long walks along the beach with my dog."])

preprocessor = hub.KerasLayer("https://tfhub.dev/google/universal-sentence-encoder-cmlm/multilingual-preprocess/2")
encoder = hub.KerasLayer("./downloads/labse-2")
model, bert_encoder = create_model()
english_embeds1 = model(preprocessor(english_sentences))
english_embeds2 = encoder(preprocessor(english_sentences))["default"]

tf.debugging.assert_near(english_embeds1["pooled_output"], english_embeds2)
bert_encoder.save_weights("./models/labse")
