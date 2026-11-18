"""Entrena, compara y evalúa dos redes densas para load_digits."""

import csv
import json
import os
import subprocess
import sys

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers, regularizers

from datos_personalizados import (
    MIN_SAMPLES_PER_DIGIT,
    cargar_dibujos_entrenamiento,
    validar_cantidad_minima,
)
from reconocimiento import ARTIFACTS_DIR, MODEL_PATH, ROOT

SEED = 42
MAX_EPOCHS = 200
BATCH_SIZE = 32
PATIENCE = 15
PERSISTENCE_TOLERANCE = 1e-6


def crear_modelos() -> tuple[keras.Model, keras.Model, dict[str, dict[str, object]]]:
    """Construye la variante Sequential y la funcional con sus configuraciones."""
    sequential = keras.Sequential(
        [
            keras.Input(shape=(64,), name="pixeles"),
            layers.Dense(
                64,
                activation="relu",
                kernel_regularizer=regularizers.l2(1e-4),
            ),
            layers.Dropout(0.15),
            layers.Dense(
                32,
                activation="relu",
                kernel_regularizer=regularizers.l2(1e-4),
            ),
            layers.Dense(10, activation="softmax"),
        ],
        name="red_secuencial",
    )

    inputs = keras.Input(shape=(64,), name="pixeles")
    hidden = layers.Dense(
        96,
        activation="relu",
        kernel_regularizer=regularizers.l2(1e-4),
    )(inputs)
    hidden = layers.Dropout(0.20)(hidden)
    hidden = layers.Dense(
        48,
        activation="relu",
        kernel_regularizer=regularizers.l2(1e-4),
    )(hidden)
    outputs = layers.Dense(10, activation="softmax")(hidden)
    functional = keras.Model(inputs=inputs, outputs=outputs, name="red_funcional")

    configs = {
        "secuencial": {
            "capas_ocultas": [64, 32],
            "activacion": "relu",
            "regularizacion": "L2=0.0001 y Dropout=0.15",
            "tasa_aprendizaje": 0.001,
            "batch_size": BATCH_SIZE,
            "max_epocas": MAX_EPOCHS,
            "early_stopping": f"val_loss, paciencia={PATIENCE}, restaurar mejores pesos",
        },
        "funcional": {
            "capas_ocultas": [96, 48],
            "activacion": "relu",
            "regularizacion": "L2=0.0001 y Dropout=0.20",
            "tasa_aprendizaje": 0.001,
            "batch_size": BATCH_SIZE,
            "max_epocas": MAX_EPOCHS,
            "early_stopping": f"val_loss, paciencia={PATIENCE}, restaurar mejores pesos",
        },
    }

    for model in (sequential, functional):
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
    return sequential, functional, configs


def metricas(model: keras.Model, x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    predictions = np.argmax(model.predict(x, verbose=0), axis=1)
    return {
        "accuracy": float(accuracy_score(y, predictions)),
        "f1_macro": float(f1_score(y, predictions, average="macro")),
    }


def guardar_curvas(histories: dict[str, keras.callbacks.History]) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(12, 4))
    for name, history in histories.items():
        axes[0].plot(history.history["loss"], label=name)
        axes[1].plot(history.history["val_loss"], label=name)
    axes[0].set(title="Pérdida de entrenamiento", xlabel="Época", ylabel="Pérdida")
    axes[1].set(title="Pérdida de validación", xlabel="Época", ylabel="Pérdida")
    for axis in axes:
        axis.legend()
        axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(ARTIFACTS_DIR / "curvas_perdida.png", dpi=150)
    plt.close(figure)


def evaluar_ruido(
    model: keras.Model, x_test: np.ndarray, y_test: np.ndarray, test_indices: np.ndarray
) -> dict[str, object]:
    """Compara 30 ejemplos de prueba con ruido fijado tras congelar el modelo."""
    rng = np.random.default_rng(2026)
    selected = np.sort(rng.choice(len(x_test), size=30, replace=False))
    originals = x_test[selected]
    noisy = np.clip(
        originals + rng.normal(0.0, 0.15, size=originals.shape), 0.0, 1.0
    ).astype(np.float32)
    original_predictions = np.argmax(model.predict(originals, verbose=0), axis=1)
    noisy_predictions = np.argmax(model.predict(noisy, verbose=0), axis=1)
    labels = y_test[selected]

    with (ARTIFACTS_DIR / "evaluacion_ruido.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.writer(file)
        writer.writerow(
            ["indice_original", "real", "prediccion_original", "prediccion_ruidosa"]
        )
        for position, real, original, noisy_prediction in zip(
            test_indices[selected], labels, original_predictions, noisy_predictions
        ):
            writer.writerow(
                [int(position), int(real), int(original), int(noisy_prediction)]
            )
    return {
        "cantidad": 30,
        "desviacion_estandar": 0.15,
        "semilla": 2026,
        "accuracy_original": float(accuracy_score(labels, original_predictions)),
        "accuracy_ruidosa": float(accuracy_score(labels, noisy_predictions)),
        "archivo_detalle": "evaluacion_ruido.csv",
    }


def main() -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    keras.utils.set_random_seed(SEED)
    tf.config.experimental.enable_op_determinism()

    digits = load_digits()
    x = (digits.data / 16.0).astype(np.float32)
    y = digits.target.astype(np.int64)
    indices = np.arange(len(y))
    train_and_validation, test_indices = train_test_split(
        indices, test_size=0.20, random_state=SEED, stratify=y
    )
    train_indices, validation_indices = train_test_split(
        train_and_validation,
        test_size=0.25,
        random_state=SEED,
        stratify=y[train_and_validation],
    )
    np.savez_compressed(
        ARTIFACTS_DIR / "indices_split.npz",
        entrenamiento=train_indices,
        validacion=validation_indices,
        prueba=test_indices,
    )
    personal_x, personal_y = cargar_dibujos_entrenamiento()
    personal_counts = validar_cantidad_minima(personal_y)
    x_training = np.concatenate((x[train_indices], personal_x), axis=0)
    y_training = np.concatenate((y[train_indices], personal_y), axis=0)

    models = crear_modelos()
    model_map = {"secuencial": models[0], "funcional": models[1]}
    configs = models[2]
    histories: dict[str, keras.callbacks.History] = {}
    validation_metrics: dict[str, dict[str, float]] = {}
    for name, model in model_map.items():
        print(f"\nEntrenando modelo {name}...")
        stopping = keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=PATIENCE,
            restore_best_weights=True,
        )
        histories[name] = model.fit(
            x_training,
            y_training,
            validation_data=(x[validation_indices], y[validation_indices]),
            epochs=MAX_EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=[stopping],
            verbose=0,
        )
        validation_metrics[name] = metricas(
            model, x[validation_indices], y[validation_indices]
        )
        print(
            f"Validación: accuracy={validation_metrics[name]['accuracy']:.4f}, "
            f"F1 macro={validation_metrics[name]['f1_macro']:.4f}, "
            f"épocas={len(histories[name].history['loss'])}"
        )

    chosen_name = max(
        model_map,
        key=lambda name: (
            validation_metrics[name]["accuracy"],
            validation_metrics[name]["f1_macro"],
        ),
    )
    test_metrics = {
        name: metricas(model, x[test_indices], y[test_indices])
        for name, model in model_map.items()
    }
    chosen = model_map[chosen_name]
    chosen.save(MODEL_PATH)
    guardar_curvas(histories)

    predictions = np.argmax(chosen.predict(x[test_indices], verbose=0), axis=1)
    matrix = confusion_matrix(y[test_indices], predictions, labels=np.arange(10))
    figure, axis = plt.subplots(figsize=(8, 7))
    image = axis.imshow(matrix, cmap="Blues")
    figure.colorbar(image, ax=axis)
    axis.set(
        title=f"Matriz de confusión — modelo {chosen_name}",
        xlabel="Predicción",
        ylabel="Valor real",
        xticks=np.arange(10),
        yticks=np.arange(10),
    )
    figure.tight_layout()
    figure.savefig(ARTIFACTS_DIR / "matriz_confusion.png", dpi=150)
    plt.close(figure)

    probabilities = chosen.predict(x[test_indices], verbose=0)
    wrong = np.flatnonzero(predictions != y[test_indices])[:3]
    error_cases: list[dict[str, object]] = []
    with (ARTIFACTS_DIR / "analisis_errores.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.writer(file)
        writer.writerow(
            ["indice_original", "real", "prediccion", "probabilidad_predicha"]
        )
        for position in wrong:
            error = {
                "indice_original": int(test_indices[position]),
                "real": int(y[test_indices[position]]),
                "prediccion": int(predictions[position]),
                "probabilidad_predicha": float(np.max(probabilities[position])),
            }
            error_cases.append(error)
            writer.writerow(
                [
                    error["indice_original"],
                    error["real"],
                    error["prediccion"],
                    error["probabilidad_predicha"],
                ]
            )
    if len(wrong):
        figure, axes = plt.subplots(1, len(wrong), figsize=(3 * len(wrong), 3))
        for axis, position in zip(np.atleast_1d(axes), wrong):
            axis.imshow(digits.images[test_indices[position]], cmap="gray_r", vmin=0, vmax=16)
            axis.set_title(
                f"Real {y[test_indices[position]]} → pred. {predictions[position]}\n"
                f"Confianza {np.max(probabilities[position]):.1%}"
            )
            axis.axis("off")
        figure.suptitle("Errores reales del modelo elegido en el conjunto de prueba")
        figure.tight_layout()
        figure.savefig(ARTIFACTS_DIR / "analisis_errores.png", dpi=150)
        plt.close(figure)

    noise_results = evaluar_ruido(
        chosen, x[test_indices], y[test_indices], test_indices
    )
    fixed_inputs = x[test_indices[:5]]
    expected_probabilities = chosen.predict(fixed_inputs, verbose=0)
    np.savez_compressed(
        ARTIFACTS_DIR / "muestras_persistencia.npz",
        inputs=fixed_inputs,
        expected=expected_probabilities,
    )
    subprocess.run(
        [sys.executable, str(ROOT / "verificar_persistencia.py")],
        cwd=ROOT,
        check=True,
    )

    report = {
        "dataset": "sklearn.datasets.load_digits",
        "cantidad": len(y),
        "forma_imagen": [8, 8],
        "entradas": 64,
        "clases": list(range(10)),
        "semilla": SEED,
        "proporciones": {
            "entrenamiento": len(train_indices) / len(y),
            "validacion": len(validation_indices) / len(y),
            "prueba": len(test_indices) / len(y),
        },
        "cantidades": {
            "entrenamiento": len(train_indices),
            "validacion": len(validation_indices),
            "prueba": len(test_indices),
        },
        "dibujos_personalizados": {
            "cantidad": len(personal_y),
            "cantidad_por_digito": personal_counts.tolist(),
            "requisito_minimo_por_digito": MIN_SAMPLES_PER_DIGIT,
            "incluidos_solo_en_entrenamiento": True,
            "archivo": "resultados/dibujos_entrenamiento.csv",
        },
        "preprocesamiento": "valores originales enteros 0..16 divididos entre 16; float32",
        "configuraciones": configs,
        "comparacion": {
            name: {
                "validacion": validation_metrics[name],
                "prueba": test_metrics[name],
                "parametros": model.count_params(),
                "epocas_efectivas": len(histories[name].history["loss"]),
            }
            for name, model in model_map.items()
        },
        "modelo_elegido_por_validacion": chosen_name,
        "prueba_persistencia": {
            "muestras": 5,
            "tolerancia_absoluta": PERSISTENCE_TOLERANCE,
            "proceso_nuevo": True,
            "resultado": "superada",
        },
        "errores_analizados": len(wrong),
        "casos_de_error": error_cases,
        "interpretacion_errores": (
            "La figura muestra los píxeles recibidos por el modelo, la etiqueta "
            "real y la predicción. Revise las confusiones de trazos visualmente "
            "parecidos; los casos y la confianza exactos están en el CSV."
        ),
        "ruido": noise_results,
    }
    (ARTIFACTS_DIR / "informe_entrenamiento.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (ARTIFACTS_DIR / "preprocesamiento.json").write_text(
        json.dumps(
            {
                "entradas": 64,
                "forma": [8, 8],
                "escala": "load_digits 0..16 dividido entre 16",
                "tipo": "float32",
                "dibujo": (
                    "recortar pixeles >10; conservar proporcion; ajustar dentro de "
                    "6x6; centrar sobre 8x8 negro; trazo blanco; dividir gris 0..255 entre 255"
                ),
                "colores_invertidos": False,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nModelo elegido por validación: {chosen_name}")
    print(f"Modelo persistido en: {MODEL_PATH}")
    print(f"Informe: {ARTIFACTS_DIR / 'informe_entrenamiento.json'}")


if __name__ == "__main__":
    main()
