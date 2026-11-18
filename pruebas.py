"""Pruebas básicas de preprocesamiento y validación de etiquetas."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from PIL import Image, ImageDraw

from datos_personalizados import (
    cargar_dibujos_entrenamiento,
    guardar_dibujo_entrenamiento,
    validar_cantidad_minima,
)
from reconocimiento import preparar_dibujo
from registro import validar_etiqueta


class PreprocessingTests(unittest.TestCase):
    def test_blank_image_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "vacío"):
            preparar_dibujo(Image.new("L", (280, 280), 0))

    def test_drawn_image_is_8_by_8_normalized_and_centered(self) -> None:
        image = Image.new("L", (280, 280), 0)
        ImageDraw.Draw(image).line((130, 40, 145, 230), fill=255, width=18)
        result = preparar_dibujo(image)

        self.assertEqual(result.shape, (8, 8))
        self.assertEqual(result.dtype, np.float32)
        self.assertGreater(float(result.max()), 0)
        self.assertLessEqual(float(result.max()), 1)
        self.assertEqual(float(result[0, 0]), 0)
        self.assertEqual(float(result[-1, -1]), 0)


class LabelTests(unittest.TestCase):
    def test_valid_digits_are_accepted(self) -> None:
        self.assertEqual(validar_etiqueta("0"), 0)
        self.assertEqual(validar_etiqueta("9"), 9)

    def test_invalid_labels_are_rejected(self) -> None:
        for value in ("-1", "10", "1.5", "x", ""):
            with self.subTest(value=value), self.assertRaises(ValueError):
                validar_etiqueta(value)


class PersonalizedDataTests(unittest.TestCase):
    def test_training_drawings_round_trip(self) -> None:
        pixels = np.linspace(0.0, 1.0, 64, dtype=np.float32)
        with TemporaryDirectory() as directory:
            path = Path(directory) / "drawings.csv"
            guardar_dibujo_entrenamiento(6, pixels, path=path)
            features, labels = cargar_dibujos_entrenamiento(path)

        self.assertEqual(features.shape, (1, 64))
        self.assertEqual(labels.tolist(), [6])
        np.testing.assert_allclose(features[0], pixels, atol=1e-7)

    def test_invalid_training_pixels_are_rejected(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "drawings.csv"
            with self.assertRaisesRegex(ValueError, "normalizados"):
                guardar_dibujo_entrenamiento(
                    2, np.full(64, 1.1, dtype=np.float32), path=path
                )

    def test_personalized_training_requires_five_samples_per_digit(self) -> None:
        incomplete = np.repeat(np.arange(10), 4)
        with self.assertRaisesRegex(ValueError, "al menos 5 dibujos"):
            validar_cantidad_minima(incomplete)

        complete = np.repeat(np.arange(10), 5)
        np.testing.assert_array_equal(validar_cantidad_minima(complete), [5] * 10)


if __name__ == "__main__":
    unittest.main()
