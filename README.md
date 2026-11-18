# Reconocimiento interactivo de dígitos

Prototipo local de escritorio con una red densa entrenada en `load_digits`.
No usa CNN ni servicios de internet. El modelo se entrena una sola vez y la
aplicación lo carga desde disco al iniciarse.

## Requisitos e inicio

Python 3.10 o posterior. Tkinter suele venir incluido con Python; en Windows,
instale Python desde python.org si la instalación actual no lo incluye.

```powershell
python -m pip install -r requirements.txt
python modelo_ml.py
python aplicacion.py
```

El entrenamiento genera el modelo y los resultados en `artefactos\`. No es
necesario volver a ejecutar `modelo_ml.py` para abrir la aplicación después de
cerrarla o reiniciar el equipo.

## Diseño y evaluación

- Los 1 797 ejemplos de `load_digits` se escalan dividiendo sus valores 0–16
  entre 16. Una semilla 42 fija y estratificación producen particiones
  aproximadas 60/20/20; los índices se conservan en `indices_split.npz`.
- Se comparan una red Sequential (64 y 32 neuronas ocultas) y una red funcional
  (96 y 48). Ambas usan ReLU, L2=0.0001, Adam con tasa 0.001, y EarlyStopping
  sobre pérdida de validación (paciencia 15, mejores pesos restaurados).
- Se selecciona por accuracy de validación y, en empate, F1 macro de
  validación. Solo después se reportan las métricas de prueba para ambos
  diseños. No se usa el conjunto de prueba para elegir el modelo.
- Las curvas de pérdida corresponden a ambos modelos; la matriz de confusión y
  el análisis visual de errores corresponden al modelo elegido. Se exportan
  métricas, configuraciones y resultados para 30 imágenes de prueba con ruido
  gaussiano (desviación 0.15, semilla 2026).
- La comprobación de persistencia carga el modelo guardado en un proceso
  separado y compara probabilidades de cinco muestras con tolerancia absoluta
  `1e-6`.
- Si se recopilan dibujos etiquetados propios, se incorporan únicamente al
  conjunto de entrenamiento. Se requieren al menos cinco ejemplos por cada
  dígito; validación y prueba permanecen solo con `load_digits`.

## Aplicación

Ejecute `python aplicacion.py`, escriba la etiqueta correcta entre 0 y 9,
dibuje con el botón izquierdo y seleccione **Reconocer y registrar**. El trazo
blanco sobre fondo negro mantiene la convención de `load_digits`. El
preprocesamiento recorta los píxeles con intensidad mayor que 10, preserva la
proporción, ajusta el trazo dentro de 6×6, lo centra en una imagen 8×8 negra y
divide la intensidad de gris entre 255. No se invierten los colores. La imagen
8×8 exacta entregada al modelo se muestra junto al lienzo.

La aplicación enseña la clase, las tres probabilidades mayores y todas las
probabilidades. Una probabilidad máxima menor que 0.70 produce un aviso
didáctico; no implica que el modelo esté calibrado. El lienzo vacío y las
etiquetas inválidas se rechazan. Los reconocimientos se agregan a
`resultados\historial.csv`.

### Personalización y prueba independiente

1. Antes de congelar el modelo, dibuje al menos cinco variantes de cada dígito
   (50 dibujos en total) y seleccione **Guardar dibujo como dato de
   entrenamiento**. Varíe el tamaño y el estilo. Estos ejemplos se guardan en
   `resultados\dibujos_entrenamiento.csv` y no se registran como prueba.
2. Ejecute `python modelo_ml.py`. El script incorpora esos dibujos solo al
   entrenamiento; si falta algún dígito con cinco ejemplos, se detiene y
   muestra cuáles faltan. Los dibujos propios no se usan para elegir el modelo
   ni para calcular las métricas de prueba.
3. Reinicie la aplicación para cargar el modelo personalizado. A partir de ahí,
   dibuje 20 ejemplos nuevos, dos por dígito, y use **Reconocer y registrar**.
   No guarde esos dibujos como datos de entrenamiento ni vuelva a entrenar
   antes de terminar la evaluación.
4. El primer intento de cada dibujo posterior al reentrenamiento se guarda en
   `resultados\dibujos_propios_personalizado.csv`. Los registros anteriores del
   modelo base permanecen en `resultados\dibujos_propios.csv` y no se mezclan.

Al completar los 20 nuevos dibujos, ejecute:

```powershell
python analizar_dibujos.py
```

El informe presenta accuracy, cantidad de avisos de baja confianza y la
comparación con prueba limpia y ruidosa. Las imágenes manuales pueden diferir
por estilo, grosor, escala y centrado frente a las imágenes del dataset.

## Verificación rápida

```powershell
python -m unittest pruebas -v
python verificar_persistencia.py
```

`verificar_persistencia.py` se ejecuta también automáticamente al terminar el
entrenamiento. Los CSV, imágenes y JSON de resultados son evidencia reproducible
del entrenamiento y las pruebas; los dibujos manuales deben recogerse en la
interfaz.
