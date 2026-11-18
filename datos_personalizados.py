"""Almacenamiento local y carga de dibujos etiquetados para entrenamiento."""

import csv
from datetime import datetime
from pathlib import Path

import numpy as np

from reconocimiento import RESULTS_DIR

TRAINING_DATA_PATH = RESULTS_DIR / "dibujos_entrenamiento.csv"
PIXEL_COLUMNS = [f"pixel_{index:02d}" for index in range(64)]
MIN_SAMPLES_PER_DIGIT = 5


def guardar_dibujo_entrenamiento(
    label: int, pixels: np.ndarray, path: Path = TRAINING_DATA_PATH
) -> None:
    if not 0 <= label <= 9:
        raise ValueError("La etiqueta debe estar entre 0 y 9.")
    flattened = np.asarray(pixels, dtype=np.float32).reshape(-1)
    if flattened.shape != (64,) or not np.isfinite(flattened).all():
        raise ValueError("El dibujo debe contener exactamente 64 valores finitos.")
    if np.any(flattened < 0.0) or np.any(flattened > 1.0):
        raise ValueError("Los píxeles deben estar normalizados entre 0 y 1.")

    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["fecha_hora", "etiqueta", *PIXEL_COLUMNS]
    with path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        if file.tell() == 0:
            writer.writerow(fields)
        writer.writerow(
            [
                datetime.now().astimezone().isoformat(timespec="seconds"),
                label,
                *(f"{value:.8f}" for value in flattened),
            ]
        )


def cargar_dibujos_entrenamiento(
    path: Path = TRAINING_DATA_PATH,
) -> tuple[np.ndarray, np.ndarray]:
    if not path.is_file():
        return np.empty((0, 64), dtype=np.float32), np.empty((0,), dtype=np.int64)

    features: list[list[float]] = []
    labels: list[int] = []
    expected_fields = ["fecha_hora", "etiqueta", *PIXEL_COLUMNS]
    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames != expected_fields:
            raise ValueError(
                f"El formato de {path} no es válido. "
                "No edite manualmente el archivo de dibujos de entrenamiento."
            )
        for line_number, row in enumerate(reader, start=2):
            try:
                label = int(row["etiqueta"])
                pixels = [float(row[column]) for column in PIXEL_COLUMNS]
            except (TypeError, ValueError, KeyError) as error:
                raise ValueError(
                    f"Fila inválida en {path}, línea {line_number}."
                ) from error
            if not 0 <= label <= 9:
                raise ValueError(
                    f"Etiqueta fuera de 0..9 en {path}, línea {line_number}."
                )
            if not np.isfinite(pixels).all() or any(
                value < 0.0 or value > 1.0 for value in pixels
            ):
                raise ValueError(
                    f"Píxeles inválidos en {path}, línea {line_number}."
                )
            labels.append(label)
            features.append(pixels)
    return np.asarray(features, dtype=np.float32), np.asarray(labels, dtype=np.int64)


def contar_dibujos_por_digito(path: Path = TRAINING_DATA_PATH) -> np.ndarray:
    _, labels = cargar_dibujos_entrenamiento(path)
    return np.bincount(labels, minlength=10)


def validar_cantidad_minima(labels: np.ndarray) -> np.ndarray:
    counts = np.bincount(labels, minlength=10)
    if len(labels) and np.any(counts < MIN_SAMPLES_PER_DIGIT):
        incomplete = [
            f"{digit}: {count}"
            for digit, count in enumerate(counts)
            if count < MIN_SAMPLES_PER_DIGIT
        ]
        raise ValueError(
            "Entrenamiento personalizado incompleto. Reúna al menos "
            f"{MIN_SAMPLES_PER_DIGIT} dibujos por dígito. Faltan: "
            + ", ".join(incomplete)
        )
    return counts
