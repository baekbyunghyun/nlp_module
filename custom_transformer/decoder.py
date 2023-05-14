import tensorflow as tf

from custom_transformer.multi_head_attention import MultiHeadAttention
from custom_transformer.position_encoder import PositionEncoder


def decoder_layer(dff, d_model, num_heads, dropout, name="decoder_layer"):
    inputs = tf.keras.Input(shape=(None, d_model), name="inputs")

    look_ahead_mask = tf.keras.Input(shape=(1, None, None), name="look_ahead_mask")

    first_attention = MultiHeadAttention(d_model, num_heads, name="first_attention")(inputs={
        'query': inputs, 'key': inputs, 'value': inputs,
        'mask': look_ahead_mask
    })
    first_attention = tf.keras.layers.LayerNormalization(epsilon=1e-6)(first_attention + inputs)

    encoder_outputs = tf.keras.Input(shape=(None, d_model), name="encoder_outputs")
    padding_mask = tf.keras.Input(shape=(1, 1, None), name="padding_mask")

    seconds_attention = MultiHeadAttention(d_model, num_heads, name="second_attention")(inputs={
        'query': first_attention, 'key': encoder_outputs, 'value': encoder_outputs,
        'mask': padding_mask
    })
    seconds_attention = tf.keras.layers.Dropout(rate=dropout)(seconds_attention)
    seconds_attention = tf.keras.layers.LayerNormalization(epsilon=1e-6)(seconds_attention + first_attention)

    outputs = tf.keras.layers.Dense(units=dff, activation='relu')(seconds_attention)
    outputs = tf.keras.layers.Dense(units=d_model)(outputs)

    outputs = tf.keras.layers.Dropout(rate=dropout)(outputs)
    outputs = tf.keras.layers.LayerNormalization(epsilon=1e-6)(outputs + seconds_attention)

    return tf.keras.Model(inputs=[inputs, encoder_outputs, look_ahead_mask, padding_mask], outputs=outputs, name=name)


def build_decoder(vocab_size, num_layers, dff, d_model, num_heads, dropout, name="decoder"):
    inputs = tf.keras.Input(shape=(None,), name="inputs")
    encoder_outputs = tf.keras.Input(shape=(None, d_model), name="encoder_outputs")

    look_ahead_mask = tf.keras.Input(shape=(1, None, None), name="look_ahead_mask")
    padding_mask = tf.keras.Input(shape=(1, 1, None), name="padding_mask")

    embeddings = tf.keras.layers.Embedding(vocab_size, d_model)(inputs)
    embeddings *= tf.math.sqrt(tf.cast(d_model, tf.float32))
    embeddings = PositionEncoder(vocab_size, d_model)(embeddings)

    outputs = tf.keras.layers.Dropout(rate=dropout)(embeddings)

    for i in range(num_layers):
        outputs = decoder_layer(dff=dff, d_model=d_model, num_heads=num_heads, dropout=dropout,
                                name="decoder_layer_{0}".format(i))(inputs=[outputs, encoder_outputs, look_ahead_mask, padding_mask])

    return tf.keras.Model(
        inputs=[inputs, encoder_outputs, look_ahead_mask, padding_mask], outputs=outputs, name=name
    )
