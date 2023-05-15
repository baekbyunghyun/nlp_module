import tensorflow as tf

from custom_transformer.encoder import build_encoder
from custom_transformer.decoder import build_decoder


def create_padding_mask(x):
    mask = tf.cast(tf.math.equal(x, 0), tf.float32)

    return mask[:, tf.newaxis, tf.newaxis, :]


def create_look_ahead_mask(x):
    sequence_length = tf.shape(x)[1]

    look_ahead_mask = 1 - tf.linalg.band_part(tf.ones((sequence_length, sequence_length)), -1, 0)
    padding_mask = create_padding_mask(x)

    return tf.maximum(look_ahead_mask, padding_mask)


def transformer(vocab_size, num_layers, dff, d_model, num_heads, dropout, name="custom_transformer"):
    inputs = tf.keras.Input(shape=(None,), name="inputs")

    decoder_inputs = tf.keras.Input(shape=(None,), name="decoder_inputs")

    encoder_padding_mask = tf.keras.layers.Lambda(create_padding_mask, output_shape=(1, 1, None), name="encoder_padding_mask")(inputs)
    look_ahead_mask = tf.keras.layers.Lambda(create_look_ahead_mask, output_shape=(1, None, None), name="look_ahead_mask")(decoder_inputs)
    decoder_padding_mask = tf.keras.layers.Lambda(create_padding_mask, output_shape=(1, 1, None), name="decoder_padding_mask")(inputs)

    encoder_outputs = build_encoder(
        vocab_size=vocab_size, num_layers=num_layers, dff=dff, d_model=d_model, num_heads=num_heads, dropout=dropout)\
        (inputs=[inputs, encoder_padding_mask]
    )

    decoder_outputs = build_decoder(
        vocab_size=vocab_size, num_layers=num_layers, dff=dff, d_model=d_model, num_heads=num_heads, dropout=dropout) \
        (inputs=[decoder_inputs, encoder_outputs, look_ahead_mask, decoder_padding_mask]
    )

    outputs = tf.keras.layers.Dense(units=vocab_size, name="outputs")(decoder_outputs)

    return tf.keras.Model(inputs=[inputs, decoder_inputs], outputs=outputs, name=name)


if __name__ == '__main__':
    small_transformer = transformer(
        vocab_size=9000,
        num_layers=4,
        dff=512,
        d_model=128,
        num_heads=4,
        dropout=0.3,
        name="small_transformer"
    )