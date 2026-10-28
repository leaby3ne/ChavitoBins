"""Interfaz gráfica de ChavitoBins (Tkinter).

Ventana con la imagen o la cámara en vivo a la izquierda (con cajas sobre los
objetos reconocidos) y un panel HUD a la derecha con la recomendación: franja
del color de la caneca, mensaje, advertencias y, cuando la cámara no puede
decidir sola, una pregunta de sí/no (por ejemplo, para llevar un empaque
metalizado al recipiente negro). El panel está al lado de la imagen para no
tapar el encuadre ni las cajas. Opcionalmente también se abre un pop-up.

Todo se ejecuta en el hilo principal con ``after``; no usa hilos.
"""

from __future__ import annotations

import time
import tkinter as tk
from collections.abc import Callable
from pathlib import Path
from tkinter import filedialog, messagebox
from tkinter import font as tkfont

import numpy as np
from PIL import Image, ImageTk

from src.camera import DEFAULT_CAMERA_INDEX, Camera, CameraError
from src.classifier import (
    Classification,
    ClassificationStatus,
    Recommendation,
    WasteClassifier,
    group_recommendations,
)
from src.display import NO_DETECTIONS_MESSAGE, NOT_RECOGNIZED_MESSAGE
from src.preprocessing import ImageLoadError, load_image_bgr
from src.stabilizer import RecommendationStabilizer
from src.vision import InferenceError, ObjectDetector

WINDOW_TITLE = "ChavitoBins - Clasificador de residuos"
CANVAS_WIDTH = 800
CANVAS_HEIGHT = 600
CANVAS_BACKGROUND = "#1e1e1e"
HUD_WIDTH = 360
HUD_BACKGROUND = "#f2f2f2"
HUD_TEXT_WIDTH = 300  # ancho de línea del texto de las tarjetas, en píxeles
HUD_IDLE_TEXT = "Muestra un objeto a la cámara\no abre una imagen.\n\nLa recomendación aparecerá aquí."
CAMERA_LOOP_DELAY_MS = 10  # pausa entre fotogramas para que la ventana responda
POPUP_WIDTH_CHARS = 440
POPUP_CASCADE_OFFSET = 30
POPUP_SIDE_GAP = 10

IMAGE_FILE_TYPES = [
    ("Imágenes", "*.jpg *.jpeg *.png *.bmp *.webp"),
    ("Todos los archivos", "*.*"),
]

# Color de las cajas según el estado. Los objetos no reconocidos no se dibujan,
# para no llenar la imagen de ruido (muebles, etc.).
BOX_COLORS = {
    ClassificationStatus.RECOMMENDED: "#2e7d32",
    ClassificationStatus.NEEDS_VERIFICATION: "#f9a825",
    ClassificationStatus.LOW_CONFIDENCE: "#757575",
}

# Franja de la tarjeta y del pop-up (fondo, texto) según la categoría, siguiendo
# el código de colores de la Resolución 2184 de 2019.
CATEGORY_STYLES = {
    "green_bin": ("#2e7d32", "#ffffff"),
    "white_bin": ("#eeeeee", "#212121"),
    "black_bin": ("#212121", "#ffffff"),
}
DEFAULT_CATEGORY_STYLE = ("#455a64", "#ffffff")
VERIFY_BANNER_TEXT = "Verifica antes de desechar: la imagen no basta para estar seguros."
VERIFY_BANNER_COLORS = ("#fff3cd", "#664d03")
YES_TEXT = "Sí"
NO_TEXT = "No"


class ChavitoBinsGui:
    """Ventana principal: análisis de imágenes y cámara en vivo."""

    def __init__(
        self,
        root: tk.Tk,
        detector: ObjectDetector,
        classifier: WasteClassifier,
        camera_index: int = DEFAULT_CAMERA_INDEX,
        # Fábrica reemplazable para poder probar sin cámara física.
        camera_factory: Callable[[int], Camera] = Camera,
    ) -> None:
        self._root = root
        self._detector = detector
        self._classifier = classifier
        self._camera_index = camera_index
        self._camera_factory = camera_factory
        self._camera: Camera | None = None
        self._after_id: str | None = None
        self._stabilizer = RecommendationStabilizer()
        self._popups: dict[str, tk.Toplevel] = {}
        self._photo: ImageTk.PhotoImage | None = None  # evita que Tk lo libere
        # Recomendaciones que muestra el HUD y respuestas del usuario a sus preguntas.
        self._hud_recommendations: dict[str, Recommendation] = {}
        self._answers: dict[str, bool] = {}

        base_font = tkfont.nametofont("TkDefaultFont")
        self._label_font = base_font.copy()
        self._label_font.configure(size=11, weight="bold")
        self._title_font = base_font.copy()
        self._title_font.configure(size=18, weight="bold")
        self._card_title_font = base_font.copy()
        self._card_title_font.configure(size=15, weight="bold")

        self._build_widgets()
        self._render_hud()
        root.title(WINDOW_TITLE)
        # Al cerrar la ventana se libera la cámara antes de salir.
        root.protocol("WM_DELETE_WINDOW", self.close)

    def _build_widgets(self) -> None:
        toolbar = tk.Frame(self._root)
        toolbar.pack(fill="x", padx=8, pady=6)
        tk.Button(toolbar, text="Abrir imagen…", command=self.open_image).pack(side="left")
        self._camera_button = tk.Button(
            toolbar, text="Iniciar cámara", command=self.toggle_camera
        )
        self._camera_button.pack(side="left", padx=8)
        # El pop-up es opcional: el HUD ya muestra la misma información.
        self._popup_enabled = tk.BooleanVar(value=False)
        tk.Checkbutton(
            toolbar, text="Ventana emergente", variable=self._popup_enabled
        ).pack(side="left", padx=8)
        self._status = tk.StringVar(value="Abre una imagen o inicia la cámara.")
        tk.Label(toolbar, textvariable=self._status, anchor="w").pack(
            side="left", fill="x", expand=True
        )

        body = tk.Frame(self._root)
        body.pack(padx=8, pady=(0, 8))
        self._canvas = tk.Canvas(
            body, width=CANVAS_WIDTH, height=CANVAS_HEIGHT,
            bg=CANVAS_BACKGROUND, highlightthickness=0,
        )
        self._canvas.pack(side="left")
        # Panel HUD a la derecha: no se superpone a la imagen ni a las cajas.
        self._hud = tk.Frame(body, width=HUD_WIDTH, height=CANVAS_HEIGHT, bg=HUD_BACKGROUND)
        self._hud.pack(side="left", padx=(8, 0))
        self._hud.pack_propagate(False)

    # ---- Modo imagen -----------------------------------------------------

    def open_image(self) -> None:
        """Pide un archivo de imagen y lo analiza."""
        path = filedialog.askopenfilename(
            title="Elige una imagen", filetypes=IMAGE_FILE_TYPES
        )
        if path:
            self.analyze_image(Path(path))

    def _new_scene(self) -> None:
        """Avisa al detector de que la próxima imagen no continúa a la anterior."""
        invalidate = getattr(self._detector, "invalidate_cache", None)
        if invalidate is not None:
            invalidate()

    def analyze_image(self, path: Path) -> None:
        """Analiza una imagen estática y muestra una recomendación por regla."""
        self.stop_camera()
        self._new_scene()
        try:
            frame = load_image_bgr(path)
            results = self._classifier.classify_all(self._detector.detect(frame))
        except (ImageLoadError, InferenceError) as exc:
            messagebox.showerror("ChavitoBins", str(exc))
            return

        recommendations = group_recommendations(results)
        self._render(frame, results)
        discarded = getattr(self._detector, "last_discarded", 0)
        if not results and discarded:
            self._status.set(f"Se descartó {discarded} detección(es) por coincidir con una persona.")
        elif not results:
            self._status.set(NO_DETECTIONS_MESSAGE)
        elif not recommendations:
            self._status.set(NOT_RECOGNIZED_MESSAGE)
        else:
            self._status.set(f"{path.name}: {len(recommendations)} recomendación(es).")
        # En imágenes estáticas no hace falta estabilizar: se muestra de inmediato.
        self._hud_recommendations = {rec.rule_id: rec for rec in recommendations}
        self._answers.clear()
        self._render_hud()
        if self._popup_enabled.get():
            for recommendation in recommendations:
                self._show_popup(recommendation)

    # ---- Modo cámara -----------------------------------------------------

    def toggle_camera(self) -> None:
        """Enciende la cámara si está apagada y la apaga si está encendida."""
        if self._camera is None:
            self.start_camera()
        else:
            self.stop_camera()

    def start_camera(self) -> None:
        """Abre la cámara e inicia el ciclo de análisis."""
        if self._camera is not None:
            return
        camera = self._camera_factory(self._camera_index)
        try:
            camera.open()
        except CameraError as exc:
            messagebox.showerror("Cámara", str(exc))
            return
        self._camera = camera
        self._stabilizer.reset()
        self._new_scene()
        self._camera_button.config(text="Detener cámara")
        self._camera_step()

    def stop_camera(self) -> None:
        """Detiene el ciclo y libera la cámara. Es seguro llamarlo siempre."""
        if self._after_id is not None:
            self._root.after_cancel(self._after_id)
            self._after_id = None
        was_running = self._camera is not None
        if self._camera is not None:
            self._camera.release()
            self._camera = None
        self._stabilizer.reset()
        self._camera_button.config(text="Iniciar cámara")
        if was_running:
            # Al apagar la cámara el HUD vuelve a su estado inicial (sin recomendaciones viejas).
            self._hud_recommendations.clear()
            self._answers.clear()
            self._render_hud()

    def _camera_step(self) -> None:
        """Procesa un fotograma y programa el siguiente."""
        if self._camera is None:
            return
        try:
            frame = self._camera.read()
        except CameraError as exc:
            self.stop_camera()
            messagebox.showerror("Cámara", str(exc))
            return

        started = time.perf_counter()
        try:
            results = self._classifier.classify_all(self._detector.detect(frame))
        except InferenceError as exc:
            self.stop_camera()
            messagebox.showerror("ChavitoBins", str(exc))
            return
        elapsed_ms = (time.perf_counter() - started) * 1000

        self._render(frame, results)
        status = f"Cámara activa · {elapsed_ms:.0f} ms por fotograma"
        discarded = getattr(self._detector, "last_discarded", 0)
        if discarded:
            status += f" · filtro de personas: {discarded} descartada(s)"
        self._status.set(status)

        # Solo se muestran recomendaciones estables (ver stabilizer.py).
        confirmed = self._stabilizer.update(group_recommendations(results))
        if self._sync_hud(confirmed):
            self._render_hud()
        if self._popup_enabled.get():
            for recommendation in confirmed:
                self._show_popup(recommendation)
        self._after_id = self._root.after(CAMERA_LOOP_DELAY_MS, self._camera_step)

    def _sync_hud(self, confirmed: list[Recommendation]) -> bool:
        """Actualiza las recomendaciones del HUD. Devuelve True si algo cambió."""
        changed = False
        for recommendation in confirmed:
            self._hud_recommendations[recommendation.rule_id] = recommendation
            changed = True
        # Lo que dejó de estar en escena sale del HUD junto con su respuesta.
        active = self._stabilizer.active_rule_ids
        for rule_id in [rid for rid in self._hud_recommendations if rid not in active]:
            del self._hud_recommendations[rule_id]
            self._answers.pop(rule_id, None)
            changed = True
        return changed

    # ---- Dibujo de la imagen ---------------------------------------------

    def _render(self, frame_bgr: np.ndarray, results: list[Classification]) -> None:
        """Dibuja el fotograma ajustado al lienzo y las cajas de lo reconocido."""
        height, width = frame_bgr.shape[:2]
        scale = min(CANVAS_WIDTH / width, CANVAS_HEIGHT / height)
        size = (max(1, int(width * scale)), max(1, int(height * scale)))
        image = Image.fromarray(frame_bgr[:, :, ::-1]).resize(size, Image.BILINEAR)
        self._photo = ImageTk.PhotoImage(image)
        offset_x = (CANVAS_WIDTH - size[0]) // 2
        offset_y = (CANVAS_HEIGHT - size[1]) // 2

        self._canvas.delete("all")
        self._canvas.create_image(offset_x, offset_y, anchor="nw", image=self._photo)
        for result in results:
            color = BOX_COLORS.get(result.status)
            if color is None:
                continue
            x1, y1, x2, y2 = result.detection.box
            left, top = offset_x + x1 * scale, offset_y + y1 * scale
            right, bottom = offset_x + x2 * scale, offset_y + y2 * scale
            self._canvas.create_rectangle(left, top, right, bottom, outline=color, width=3)

            label = f"{result.display_name} {result.detection.confidence:.0%}"
            text_id = self._canvas.create_text(
                left + 5, top + 3, anchor="nw", text=label, fill="white", font=self._label_font
            )
            x0, y0, x1b, y1b = self._canvas.bbox(text_id)
            background = self._canvas.create_rectangle(
                x0 - 3, y0 - 1, x1b + 3, y1b + 1, fill=color, outline=color
            )
            self._canvas.tag_lower(background, text_id)

    # ---- Panel HUD ---------------------------------------------------------

    def _resolve(self, recommendation: Recommendation) -> Recommendation:
        """Aplica la respuesta del usuario (si la hay) a una recomendación."""
        answer = self._answers.get(recommendation.rule_id)
        if recommendation.follow_up is not None and answer is not None:
            return self._classifier.answer_follow_up(recommendation, answer)
        return recommendation

    def _answer(self, rule_id: str, yes: bool) -> None:
        """Guarda la respuesta del usuario a una pregunta y redibuja el HUD."""
        self._answers[rule_id] = yes
        self._render_hud()

    def _render_hud(self) -> None:
        """Redibuja las tarjetas de recomendación del panel derecho."""
        for child in self._hud.winfo_children():
            child.destroy()
        if not self._hud_recommendations:
            tk.Label(
                self._hud, text=HUD_IDLE_TEXT, bg=HUD_BACKGROUND, fg="#666666",
                justify="center", wraplength=HUD_TEXT_WIDTH,
            ).pack(expand=True)
            return
        for recommendation in self._hud_recommendations.values():
            self._build_card(self._resolve(recommendation), recommendation.rule_id)

    def _build_card(self, recommendation: Recommendation, rule_id: str) -> None:
        """Construye la tarjeta de una recomendación (franja, texto y pregunta)."""
        background, foreground = CATEGORY_STYLES.get(
            recommendation.category.id, DEFAULT_CATEGORY_STYLE
        )
        card = tk.Frame(
            self._hud, bg="white", highlightbackground="#bdbdbd", highlightthickness=1
        )
        card.pack(fill="x", padx=8, pady=(8, 0))
        tk.Label(
            card, text=recommendation.category.name, font=self._card_title_font,
            bg=background, fg=foreground, pady=8,
        ).pack(fill="x")

        if (
            recommendation.status is ClassificationStatus.NEEDS_VERIFICATION
            and recommendation.follow_up is None
        ):
            banner_bg, banner_fg = VERIFY_BANNER_COLORS
            tk.Label(
                card, text=VERIFY_BANNER_TEXT, bg=banner_bg, fg=banner_fg,
                wraplength=HUD_TEXT_WIDTH, justify="left", padx=8, pady=4,
            ).pack(fill="x")

        body = tk.Frame(card, bg="white", padx=10, pady=8)
        body.pack(fill="x")
        tk.Label(
            body, text=f"Detectado: {', '.join(recommendation.names)}",
            font=self._label_font, bg="white", anchor="w",
        ).pack(fill="x")
        tk.Label(
            body, text=recommendation.message, bg="white", wraplength=HUD_TEXT_WIDTH,
            justify="left", anchor="w",
        ).pack(fill="x", pady=(6, 0))
        for warning in recommendation.warnings:
            tk.Label(
                body, text=f"• {warning}", bg="white", fg="#555555",
                wraplength=HUD_TEXT_WIDTH, justify="left", anchor="w",
            ).pack(fill="x", pady=(4, 0))

        follow_up = recommendation.follow_up
        if follow_up is not None:
            # Pregunta que la cámara no puede responder sola.
            tk.Label(
                body, text=follow_up.question, bg="white", font=self._label_font,
                wraplength=HUD_TEXT_WIDTH, justify="left", anchor="w",
            ).pack(fill="x", pady=(10, 4))
            buttons = tk.Frame(body, bg="white")
            buttons.pack(anchor="w")
            tk.Button(
                buttons, text=YES_TEXT, width=8,
                command=lambda: self._answer(rule_id, True),
            ).pack(side="left", padx=(0, 8))
            tk.Button(
                buttons, text=NO_TEXT, width=8,
                command=lambda: self._answer(rule_id, False),
            ).pack(side="left")

    # ---- Pop-up opcional -----------------------------------------------------

    def _show_popup(self, recommendation: Recommendation) -> None:
        """Abre el pop-up de una recomendación (o lo trae al frente si ya existe)."""
        existing = self._popups.get(recommendation.rule_id)
        if existing is not None and existing.winfo_exists():
            existing.lift()
            return

        background, foreground = CATEGORY_STYLES.get(
            recommendation.category.id, DEFAULT_CATEGORY_STYLE
        )
        window = tk.Toplevel(self._root)
        window.title("ChavitoBins")
        window.resizable(False, False)
        window.transient(self._root)

        tk.Label(
            window, text=recommendation.category.name, font=self._title_font,
            bg=background, fg=foreground, padx=24, pady=14,
        ).pack(fill="x")
        if recommendation.status is ClassificationStatus.NEEDS_VERIFICATION:
            banner_bg, banner_fg = VERIFY_BANNER_COLORS
            tk.Label(
                window, text=VERIFY_BANNER_TEXT, bg=banner_bg, fg=banner_fg,
                wraplength=POPUP_WIDTH_CHARS, justify="left", padx=12, pady=6,
            ).pack(fill="x")

        body = tk.Frame(window, padx=16, pady=12)
        body.pack(fill="both")
        tk.Label(
            body, text=f"Detectado: {', '.join(recommendation.names)}",
            font=self._label_font, anchor="w",
        ).pack(fill="x")
        tk.Label(
            body, text=recommendation.message, wraplength=POPUP_WIDTH_CHARS,
            justify="left", anchor="w",
        ).pack(fill="x", pady=(8, 0))
        for warning in recommendation.warnings:
            tk.Label(
                body, text=f"• {warning}", wraplength=POPUP_WIDTH_CHARS,
                justify="left", anchor="w", fg="#555555",
            ).pack(fill="x", pady=(6, 0))
        if recommendation.follow_up is not None:
            tk.Label(
                body, text=f"Verifica: {recommendation.follow_up.question} "
                "Responde en el panel de la ventana principal.",
                wraplength=POPUP_WIDTH_CHARS, justify="left", anchor="w",
                font=self._label_font,
            ).pack(fill="x", pady=(10, 0))
        tk.Button(window, text="Entendido", command=window.destroy).pack(pady=(0, 12))
        self._place_popup(window)

        rule_id = recommendation.rule_id
        self._popups[rule_id] = window
        # <Destroy> también se dispara para los widgets hijos; se filtra la ventana.
        window.bind(
            "<Destroy>",
            lambda event: self._popups.pop(rule_id, None) if event.widget is window else None,
        )

    def _place_popup(self, window: tk.Toplevel) -> None:
        """Coloca el pop-up a la derecha de la ventana principal si cabe en pantalla.

        Si no hay espacio, se superpone, y los pop-ups abiertos a la vez se escalonan.
        """
        window.update_idletasks()
        popup_width = window.winfo_reqwidth()
        right_x = self._root.winfo_rootx() + self._root.winfo_width() + POPUP_SIDE_GAP
        cascade = POPUP_CASCADE_OFFSET * len(self._popups)
        if right_x + popup_width <= self._root.winfo_screenwidth():
            x, y = right_x, self._root.winfo_rooty() + cascade
        else:
            x = self._root.winfo_rootx() + 60 + cascade
            y = self._root.winfo_rooty() + 60 + cascade
        window.geometry(f"+{x}+{y}")

    # ---- Cierre ----------------------------------------------------------

    def close(self) -> None:
        """Libera la cámara, cierra los pop-ups y la ventana principal."""
        self.stop_camera()
        for window in list(self._popups.values()):
            window.destroy()
        self._root.destroy()


def run_gui(
    detector: ObjectDetector,
    classifier: WasteClassifier,
    camera_index: int = DEFAULT_CAMERA_INDEX,
) -> None:
    """Abre la ventana principal y espera hasta que el usuario la cierre."""
    root = tk.Tk()
    ChavitoBinsGui(root, detector, classifier, camera_index)
    root.mainloop()
