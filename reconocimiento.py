"""Utilidades compartidas de preprocesamiento y rutas del proyecto."""

from pathlib import Path

import numpy as np
from PIL import Image
from tensorflow import keras

ROOT = Path(__file__).resolve().parent
ARTIFACTS_DIR = ROOT / "artefactos"
RESULTS_DIR = ROOT / "resultados"
MODEL_PATH = ARTIFACTS_DIR / "modelo_final.keras"
METADATA_PATH = ARTIFACTS_DIR / "preprocesamiento.json"
HISTORY_PATH = RESULTS_DIR / "historial.csv"
OWN_DRAWINGS_PATH = RESULTS_DIR / "dibujos_propios.csv"
PERSONALIZED_DRAWINGS_PATH = RESULTS_DIR / "dibujos_propios_personalizado.csv"
CONFIDENCE_THRESHOLD = 0.70


def preparar_dibujo(image: Image.Image) -> np.ndarray:
    """Recorta tinta, conserva proporción y centra el dígito en una matriz 8x8."""
    gray = image.convert("L")
    pixels = np.asarray(gray)
    ys, xs = np.where(pixels > 10)
    if len(xs) == 0:
        raise ValueError("El lienzo está vacío.")

    cropped = gray.crop((int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1))
    width, height = cropped.size
    scale = min(6 / width, 6 / height)
    resized_size = (max(1, round(width * scale)), max(1, round(height * scale)))
    resized = cropped.resize(resized_size, Image.Resampling.LANCZOS)

    canvas = Image.new("L", (8, 8), 0)
    offset = ((8 - resized.width) // 2, (8 - resized.height) // 2)
    canvas.paste(resized, offset)
    return np.asarray(canvas, dtype=np.float32) / 255.0


def cargar_modelo() -> keras.Model:
    """Carga el modelo entrenado y comunica claramente si todavía no existe."""
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"No se encontró el modelo en {MODEL_PATH}. "
            "Ejecute primero «python modelo_ml.py»."
        )
    return keras.models.load_model(MODEL_PATH)
