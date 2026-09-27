"""
Vision modal training script — MobileNetV2 + SE Attention
Full training code is in notebooks/fytopod_cnn.ipynb
This script documents the training configuration used.
"""
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, Model
from pathlib import Path

MODEL_SAVE_PATH = Path(__file__).parent.parent / "models" / "fytopod_final.keras"
IMG_SIZE        = 224
BATCH_SIZE      = 32
NUM_CLASSES     = 25


def se_block(x, ratio: int = 16):
    ch = x.shape[-1]
    se = layers.GlobalAveragePooling2D()(x)
    se = layers.Reshape((1, 1, ch))(se)
    se = layers.Dense(ch // ratio, activation="relu",
                      use_bias=False, name="se_squeeze")(se)
    se = layers.Dense(ch, activation="sigmoid",
                      use_bias=False, name="se_excite")(se)
    return layers.Multiply(name="se_scale")([x, se])


def build_model(num_classes: int = NUM_CLASSES) -> Model:
    # Image stream
    img_input = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3), name="image_input")
    base = MobileNetV2(include_top=False, weights="imagenet",
                       input_tensor=img_input)
    base.trainable = False

    x = base.output
    x = se_block(x)
    x = layers.GlobalAveragePooling2D()(x)

    # Sensor stream
    sensor_input = layers.Input(shape=(12, 6), name="sensor_input")
    s = layers.LSTM(32, dropout=0.2)(sensor_input)
    s = layers.Dense(64, activation="relu")(s)

    # Gated fusion
    v = layers.Dense(128)(x)
    sens = layers.Dense(128)(s)
    gate = layers.Multiply()([v, sens])
    gate = layers.Dense(128, activation="sigmoid")(gate)
    fused = layers.Add()([
        layers.Multiply()([v, gate]),
        layers.Multiply()([sens, gate])
    ])

    # Classification head
    out = layers.BatchNormalization()(fused)
    out = layers.Dropout(0.4)(out)
    out = layers.Dense(256, activation="relu")(out)
    out = layers.Dropout(0.3)(out)
    out = layers.Dense(num_classes, activation="softmax")(out)

    return Model(inputs=[img_input, sensor_input], outputs=out,
                 name="FytoPod_MobileNetV2_SE")


# Phase 1 config — frozen backbone
PHASE1_CONFIG = {
    "optimizer": "Adam(lr=1e-3)",
    "loss": "categorical_crossentropy",
    "epochs": 10,
    "callbacks": ["EarlyStopping(patience=4)", "ModelCheckpoint", "ReduceLROnPlateau"]
}

# Phase 2 config — fine-tune top 30 layers
PHASE2_CONFIG = {
    "unfrozen_layers": "base.layers[:-30]",
    "optimizer": "Adam(lr=1e-4)",
    "epochs": 15,
    "callbacks": ["EarlyStopping(patience=5)", "ModelCheckpoint", "ReduceLROnPlateau"]
}

if __name__ == "__main__":
    model = build_model()
    model.summary()
    print(f"\nTotal parameters: {model.count_params():,}")
    print("\nSee notebooks/fytopod_cnn.ipynb for full training.")