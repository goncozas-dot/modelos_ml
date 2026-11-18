"""Resume los primeros 20 dibujos propios sin descartar resultados incorrectos."""

import csv
import json

from reconocimiento import ARTIFACTS_DIR, PERSONALIZED_DRAWINGS_PATH, RESULTS_DIR


def main() -> None:
    if not PERSONALIZED_DRAWINGS_PATH.is_file():
        raise FileNotFoundError(
            "Aún no hay dibujos de evaluación del modelo personalizado. "
            "Use la aplicación y reconozca 20 dibujos nuevos."
        )
    with PERSONALIZED_DRAWINGS_PATH.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    if len(rows) < 20:
        raise ValueError(
            f"Hay {len(rows)} primeros intentos; se requieren 20 para el análisis."
        )
    training_report_path = ARTIFACTS_DIR / "informe_entrenamiento.json"
    if not training_report_path.is_file():
        raise FileNotFoundError(
            "Falta el informe de entrenamiento. Ejecute primero «python modelo_ml.py»."
        )
    training_report = json.loads(training_report_path.read_text(encoding="utf-8"))
    selected = rows[:20]
    correct = sum(int(row["correcto"]) for row in selected)
    low_confidence = sum(int(row["aviso_baja_confianza"]) for row in selected)
    chosen_model = training_report["modelo_elegido_por_validacion"]
    test_accuracy = training_report["comparacion"][chosen_model]["prueba"]["accuracy"]
    noise = training_report["ruido"]
    own_accuracy = correct / len(selected)
    report = {
        "cantidad": len(selected),
        "accuracy": own_accuracy,
        "cantidad_avisos_baja_confianza": low_confidence,
        "rango": "primeros 20 dibujos nuevos tras congelar el modelo personalizado",
        "comparacion": {
            "accuracy_prueba_completa": test_accuracy,
            "accuracy_original_30_muestras": noise["accuracy_original"],
            "accuracy_ruidosa_30_muestras": noise["accuracy_ruidosa"],
            "diferencia_dibujos_menos_prueba": own_accuracy - test_accuracy,
        },
        "explicacion": (
            "La prueba usa imágenes pequeñas centradas y etiquetadas de load_digits; "
            "los dibujos manuales varían en trazo, grosor, escala y centrado. La "
            "prueba ruidosa altera intensidades de una selección de solo 30 ejemplos, "
            "por lo que su resultado también depende de la muestra y de la semilla."
        ),
        "mejora_siguiente_version": (
            "Agregar ejemplos etiquetados con trazos, estilos y dispositivos "
            "distintos antes de congelar el próximo modelo; conservar una nueva "
            "prueba manual independiente."
        ),
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output = RESULTS_DIR / "informe_dibujos_personalizados.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Informe guardado en {output}")


if __name__ == "__main__":
    main()
