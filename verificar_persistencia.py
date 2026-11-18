"""Verifica predicciones al cargar el modelo desde un proceso independiente."""

import numpy as np

from reconocimiento import MODEL_PATH, ROOT

TOLERANCE = 1e-6


def main() -> None:
    import tensorflow as tf

    sample_path = ROOT / "artefactos" / "muestras_persistencia.npz"
    with np.load(sample_path) as samples:
        inputs = samples["inputs"]
        expected = samples["expected"]
    model = tf.keras.models.load_model(MODEL_PATH)
    actual = model.predict(inputs, verbose=0)
    difference = float(np.max(np.abs(actual - expected)))
    if not np.allclose(actual, expected, atol=TOLERANCE, rtol=0.0):
        raise AssertionError(
            f"Persistencia fallida: diferencia máxima {difference:.8g}, "
            f"tolerancia {TOLERANCE}."
        )
    print(
        f"Persistencia verificada en proceso nuevo: {len(inputs)} muestras, "
        f"diferencia máxima={difference:.3g}, tolerancia={TOLERANCE}."
    )


if __name__ == "__main__":
    main()
