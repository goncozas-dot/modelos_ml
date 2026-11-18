"""Validación de etiquetas y persistencia de los resultados en CSV."""

import csv
from datetime import datetime
from pathlib import Path

from reconocimiento import (
    CONFIDENCE_THRESHOLD,
    HISTORY_PATH,
    OWN_DRAWINGS_PATH,
)


def validar_etiqueta(value: str) -> int:
    try:
        label = int(value)
    except ValueError as error:
        raise ValueError("Indique un dígito entero entre 0 y 9.") from error
    if str(label) != value.strip() or not 0 <= label <= 9:
        raise ValueError("La etiqueta debe ser un dígito entero entre 0 y 9.")
    return label


def _append_row(path: Path, fields: list[str], row: list[object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = path.exists() and path.stat().st_size > 0
    with path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(fields)
        writer.writerow(row)


def guardar_resultado(label: int, prediction: int, confidence: float) -> None:
    """Agrega el reconocimiento al historial persistente."""
    if not 0 <= label <= 9:
        raise ValueError("La etiqueta debe estar entre 0 y 9.")
    timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
    correct = label == prediction
    _append_row(
        HISTORY_PATH,
        ["fecha_hora", "etiqueta_indicada", "prediccion", "probabilidad_maxima", "resultado"],
        [
            timestamp,
            label,
            prediction,
            f"{confidence:.8f}",
            "correcto" if correct else "incorrecto",
        ],
    )


def guardar_dibujo_propio(
    label: int,
    prediction: int,
    confidence: float,
    path: Path = OWN_DRAWINGS_PATH,
) -> None:
    """Guarda el primer reconocimiento de cada dibujo para la evaluación manual."""
    if not 0 <= label <= 9:
        raise ValueError("La etiqueta debe estar entre 0 y 9.")
    timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
    _append_row(
        path,
        [
            "fecha_hora",
            "etiqueta_indicada",
            "prediccion",
            "probabilidad_maxima",
            "correcto",
            "aviso_baja_confianza",
        ],
        [
            timestamp,
            label,
            prediction,
            f"{confidence:.8f}",
            int(label == prediction),
            int(confidence < CONFIDENCE_THRESHOLD),
        ],
    )
