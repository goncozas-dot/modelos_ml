"""Interfaz Tkinter para dibujar, reconocer y registrar dígitos."""

import tkinter as tk
from tkinter import messagebox, ttk

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from PIL import Image, ImageDraw, ImageTk

from datos_personalizados import (
    MIN_SAMPLES_PER_DIGIT,
    contar_dibujos_por_digito,
    guardar_dibujo_entrenamiento,
)
from reconocimiento import (
    CONFIDENCE_THRESHOLD,
    MODEL_PATH,
    PERSONALIZED_DRAWINGS_PATH,
    preparar_dibujo,
)
from registro import guardar_dibujo_propio, guardar_resultado, validar_etiqueta

CANVAS_SIZE = 280
BRUSH_WIDTH = 22


class DigitApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Reconocimiento interactivo de dígitos")
        self.root.geometry("1120x760")
        self.model = None
        self.last_point: tuple[int, int] | None = None
        self.recorded_current_drawing = False
        self.saved_training_current_drawing = False
        self.evaluated_current_drawing = False
        self.source = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), 0)
        self.draw = ImageDraw.Draw(self.source)
        self.preview_photo: ImageTk.PhotoImage | None = None

        try:
            from reconocimiento import cargar_modelo

            self.model = cargar_modelo()
            status = f"Modelo cargado: {MODEL_PATH.name}"
        except FileNotFoundError as error:
            status = str(error)
        except Exception as error:
            status = f"No se pudo cargar el modelo: {error}"

        self._build_ui(status)
        self._refresh_training_count()

    def _build_ui(self, status: str) -> None:
        self.root.columnconfigure(0, weight=1)
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(1, weight=1)

        ttk.Label(
            self.root,
            text="Dibuje un dígito (trazo blanco sobre fondo negro)",
            font=("Segoe UI", 13, "bold"),
        ).grid(row=0, column=0, columnspan=2, pady=(14, 8))

        left = ttk.Frame(self.root, padding=12)
        left.grid(row=1, column=0, sticky="nsew")
        right = ttk.Frame(self.root, padding=12)
        right.grid(row=1, column=1, sticky="nsew")

        self.canvas = tk.Canvas(
            left,
            width=CANVAS_SIZE,
            height=CANVAS_SIZE,
            background="black",
            highlightthickness=1,
            highlightbackground="#777",
        )
        self.canvas.pack(pady=(0, 10))
        self.canvas.bind("<Button-1>", self._start_stroke)
        self.canvas.bind("<B1-Motion>", self._draw_stroke)
        self.canvas.bind("<ButtonRelease-1>", self._end_stroke)

        label_row = ttk.Frame(left)
        label_row.pack(fill="x", pady=6)
        ttk.Label(label_row, text="Dígito que quiso dibujar (0-9):").pack(side="left")
        self.label_entry = ttk.Entry(label_row, width=6)
        self.label_entry.pack(side="left", padx=8)

        buttons = ttk.Frame(left)
        buttons.pack(fill="x", pady=8)
        ttk.Button(buttons, text="Limpiar", command=self.clear).pack(side="left")
        ttk.Button(buttons, text="Reconocer y registrar", command=self.recognize).pack(
            side="left", padx=8
        )
        ttk.Button(
            left,
            text="Guardar dibujo como dato de entrenamiento",
            command=self.save_training_sample,
        ).pack(anchor="w", pady=4)
        self.training_count_label = ttk.Label(left, text="", wraplength=480)
        self.training_count_label.pack(anchor="w", pady=4)
        ttk.Label(left, text=status, wraplength=480).pack(anchor="w", pady=8)
        ttk.Label(
            left,
            text=(
                "Fase 1: reúna al menos 5 dibujos distintos por dígito con el botón "
                "de entrenamiento; varíe la forma de escribirlos. Después ejecute "
                "python modelo_ml.py y reinicie la app. Fase 2: evalúe con 20 "
                "dibujos nuevos usando «Reconocer y registrar»."
            ),
            wraplength=480,
        ).pack(anchor="w")

        right.columnconfigure(0, weight=1)
        ttk.Label(right, text="Entrada real del modelo (8 × 8)").grid(row=0, column=0)
        self.preview = ttk.Label(right)
        self.preview.grid(row=1, column=0, pady=5)
        self.result_label = ttk.Label(
            right, text="Predicción: —", font=("Segoe UI", 16, "bold")
        )
        self.result_label.grid(row=2, column=0, pady=(8, 2))
        self.top_three = ttk.Label(right, text="Tres clases principales: —")
        self.top_three.grid(row=3, column=0, pady=2)
        self.warning = ttk.Label(right, text="", foreground="#b00020")
        self.warning.grid(row=4, column=0, pady=2)

        self.figure = Figure(figsize=(5.5, 3.2), dpi=100)
        self.axis = self.figure.add_subplot(111)
        self._draw_empty_chart()
        self.chart = FigureCanvasTkAgg(self.figure, master=right)
        self.chart.get_tk_widget().grid(row=5, column=0, sticky="nsew", pady=8)
        right.rowconfigure(5, weight=1)

    def _start_stroke(self, event: tk.Event) -> None:
        self.last_point = (event.x, event.y)
        self.recorded_current_drawing = False
        self.saved_training_current_drawing = False
        self.evaluated_current_drawing = False
        radius = BRUSH_WIDTH // 2
        self.draw.ellipse(
            (
                event.x - radius,
                event.y - radius,
                event.x + radius,
                event.y + radius,
            ),
            fill=255,
        )
        self.canvas.create_oval(
            event.x - radius,
            event.y - radius,
            event.x + radius,
            event.y + radius,
            fill="white",
            outline="white",
        )

    def _draw_stroke(self, event: tk.Event) -> None:
        if self.last_point is None:
            self._start_stroke(event)
            return
        point = (event.x, event.y)
        self.draw.line([self.last_point, point], fill=255, width=BRUSH_WIDTH)
        radius = BRUSH_WIDTH // 2
        self.draw.ellipse(
            (
                point[0] - radius,
                point[1] - radius,
                point[0] + radius,
                point[1] + radius,
            ),
            fill=255,
        )
        self.canvas.create_line(
            *self.last_point,
            *point,
            fill="white",
            width=BRUSH_WIDTH,
            capstyle=tk.ROUND,
            smooth=True,
        )
        self.last_point = point

    def _end_stroke(self, event: tk.Event) -> None:
        if self.last_point is not None and self.last_point != (event.x, event.y):
            self.draw.line(
                [self.last_point, (event.x, event.y)], fill=255, width=BRUSH_WIDTH
            )
            self.canvas.create_line(
                *self.last_point,
                event.x,
                event.y,
                fill="white",
                width=BRUSH_WIDTH,
                capstyle=tk.ROUND,
                smooth=True,
            )
        self.last_point = None

    def _draw_empty_chart(self) -> None:
        self.axis.clear()
        self.axis.set(
            title="Probabilidades por dígito",
            xlabel="Dígito",
            ylabel="Probabilidad",
            ylim=(0, 1),
            xticks=np.arange(10),
        )
        self.axis.grid(axis="y", alpha=0.25)
        self.figure.tight_layout()
        if hasattr(self, "chart"):
            self.chart.draw_idle()

    def clear(self) -> None:
        self.canvas.delete("all")
        self.source = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), 0)
        self.draw = ImageDraw.Draw(self.source)
        self.last_point = None
        self.recorded_current_drawing = False
        self.saved_training_current_drawing = False
        self.evaluated_current_drawing = False
        self.preview.configure(image="")
        self.preview_photo = None
        self.result_label.configure(text="Predicción: —")
        self.top_three.configure(text="Tres clases principales: —")
        self.warning.configure(text="")
        self._draw_empty_chart()

    def _refresh_training_count(self) -> bool:
        try:
            counts = contar_dibujos_por_digito()
        except (OSError, ValueError) as error:
            self.training_count_label.configure(
                text=f"No se pudo leer el conjunto personalizado: {error}"
            )
            messagebox.showerror("Datos de entrenamiento inválidos", str(error))
            return False
        count_summary = "  ".join(
            f"{digit}:{int(counts[digit])}" for digit in range(10)
        )
        total = int(counts.sum())
        goal = 10 * MIN_SAMPLES_PER_DIGIT
        self.training_count_label.configure(
            text=(
                f"Datos de entrenamiento: {total}/{goal} mínimos "
                f"({count_summary})"
            )
        )
        return True

    def save_training_sample(self) -> None:
        if self.evaluated_current_drawing:
            messagebox.showwarning(
                "Dibujo ya evaluado",
                "Este dibujo ya se reconoció para la evaluación. Dibuje uno nuevo "
                "antes de guardarlo como dato de entrenamiento.",
            )
            return
        if self.saved_training_current_drawing:
            messagebox.showwarning(
                "Dibujo ya guardado",
                "Este dibujo ya está guardado. Dibuje uno nuevo para evitar duplicados.",
            )
            return
        try:
            label = validar_etiqueta(self.label_entry.get())
        except ValueError as error:
            messagebox.showerror("Etiqueta inválida", str(error))
            return
        if self.source.getbbox() is None:
            messagebox.showwarning(
                "Lienzo vacío", "Dibuje un dígito antes de guardarlo."
            )
            return
        try:
            pixels = preparar_dibujo(self.source)
            guardar_dibujo_entrenamiento(label, pixels)
        except (OSError, ValueError) as error:
            messagebox.showerror(
                "No se pudo guardar el dibujo", str(error)
            )
            return
        self.saved_training_current_drawing = True
        if not self._refresh_training_count():
            return
        messagebox.showinfo(
            "Ejemplo guardado",
            f"Se guardó el dígito {label} como dato de entrenamiento. "
            "No lo use en la prueba final de 20 dibujos nuevos.",
        )

    def recognize(self) -> None:
        if self.model is None:
            messagebox.showerror(
                "Modelo no disponible",
                f"No se puede reconocer. Genere el modelo con:\n\n"
                f"python modelo_ml.py\n\nRuta esperada: {MODEL_PATH}",
            )
            return
        try:
            label = validar_etiqueta(self.label_entry.get())
        except ValueError as error:
            messagebox.showerror("Etiqueta inválida", str(error))
            return
        if self.source.getbbox() is None:
            messagebox.showwarning("Lienzo vacío", "Dibuje un dígito antes de reconocer.")
            return

        try:
            image_8x8 = preparar_dibujo(self.source)
            probabilities = self.model.predict(image_8x8.reshape(1, 64), verbose=0)[0]
            prediction = int(np.argmax(probabilities))
            confidence = float(probabilities[prediction])
            guardar_resultado(label, prediction, confidence)
            self.evaluated_current_drawing = True
            if (
                not self.saved_training_current_drawing
                and not self.recorded_current_drawing
            ):
                guardar_dibujo_propio(
                    label,
                    prediction,
                    confidence,
                    path=PERSONALIZED_DRAWINGS_PATH,
                )
                self.recorded_current_drawing = True
        except Exception as error:
            messagebox.showerror("Error al reconocer", f"No se completó el proceso:\n{error}")
            return

        self._show_image(image_8x8)
        self.result_label.configure(
            text=f"Predicción: {prediction}  (confianza {confidence:.1%})"
        )
        top_indices = np.argsort(probabilities)[-3:][::-1]
        self.top_three.configure(
            text="Tres clases principales: "
            + ", ".join(
                f"{digit} ({probabilities[digit]:.1%})" for digit in top_indices
            )
        )
        if confidence < CONFIDENCE_THRESHOLD:
            self.warning.configure(
                text=f"Aviso: baja confianza (< {CONFIDENCE_THRESHOLD:.0%})."
            )
        else:
            self.warning.configure(text="")
        self._draw_probabilities(probabilities)

    def _show_image(self, image_8x8: np.ndarray) -> None:
        image = Image.fromarray(np.rint(image_8x8 * 255).astype(np.uint8))
        enlarged = image.resize((160, 160), Image.Resampling.NEAREST)
        self.preview_photo = ImageTk.PhotoImage(enlarged)
        self.preview.configure(image=self.preview_photo)

    def _draw_probabilities(self, probabilities: np.ndarray) -> None:
        self.axis.clear()
        colors = ["#4c78a8"] * 10
        colors[int(np.argmax(probabilities))] = "#f58518"
        self.axis.bar(np.arange(10), probabilities, color=colors)
        self.axis.axhline(
            CONFIDENCE_THRESHOLD, color="#b00020", linestyle="--", linewidth=1
        )
        self.axis.set(
            title="Probabilidades por dígito (línea: umbral 0.70)",
            xlabel="Dígito",
            ylabel="Probabilidad",
            ylim=(0, 1),
            xticks=np.arange(10),
        )
        self.axis.grid(axis="y", alpha=0.25)
        self.figure.tight_layout()
        self.chart.draw_idle()


def main() -> None:
    root = tk.Tk()
    DigitApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
