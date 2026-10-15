from __future__ import annotations

import argparse
from pathlib import Path

import keras
from keras.saving import register_keras_serializable
from keras import layers
import tensorflow as tf

@register_keras_serializable(package="Custom")
class AttentionPooling(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.score_dense = layers.Dense(1)

    def call(self, x):
        scores = self.score_dense(x)
        weights = tf.nn.softmax(scores, axis=1)
        return tf.reduce_sum(weights * x, axis=1)

    def get_config(self):
        return super().get_config()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True, help='Path ke best_model.keras')
    ap.add_argument('--output_dir', required=True, help='Folder output SavedModel')
    ap.add_argument('--weights_out', default='', help='Opsional: simpan juga weights .h5')
    args = ap.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)

    model = keras.models.load_model(
        input_path,
        custom_objects={'AttentionPooling': AttentionPooling},
        compile=False,
        safe_mode=False,
    )
    print('[OK] loaded:', input_path)
    model.export(output_dir)
    print('[OK] exported SavedModel to:', output_dir)

    if args.weights_out:
        model.save_weights(args.weights_out)
        print('[OK] saved weights to:', args.weights_out)

if __name__ == '__main__':
    main()
