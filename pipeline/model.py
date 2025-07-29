import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dropout, LayerNormalization, MultiHeadAttention, Add, Dense, GlobalAveragePooling1D

def transformer_block(inputs, head_size, num_heads, ff_dim, dropout=0.1):
    x = LayerNormalization(epsilon=1e-6)(inputs)
    x = MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(x, x)
    x = Add()([x, inputs])
    x = LayerNormalization(epsilon=1e-6)(x)
    ff = tf.keras.Sequential([
        Dense(ff_dim, activation="relu"),
        Dropout(dropout),
        Dense(inputs.shape[-1]),
    ])
    x = ff(x)
    return Add()([x, inputs])

def build_model(hp, input_shape):
    inputs = Input(shape=input_shape)
    x = LSTM(units=hp.Int("lstm_units", 32, 128, step=32), return_sequences=True)(inputs)
    x = Dropout(hp.Float("lstm_dropout", 0.1, 0.5, step=0.1))(x)
    x = transformer_block(
        x,
        head_size=hp.Choice("head_size", [16, 32, 64]),
        num_heads=hp.Choice("num_heads", [2, 4, 8]),
        ff_dim=hp.Choice("ff_dim", [32, 64, 128]),
        dropout=hp.Float("trans_dropout", 0.1, 0.5, step=0.1)
    )
    x = GlobalAveragePooling1D()(x)
    x = Dense(hp.Int("dense_units", 32, 128, step=32), activation="relu")(x)
    x = Dropout(hp.Float("final_dropout", 0.1, 0.5, step=0.1))(x)
    outputs = Dense(input_shape[-1])(x)
    model = Model(inputs, outputs)
    model.compile(optimizer="adam", loss="mse")
    return model