"""
Air Writer — Tkinter UI Edition
Requirements: pip install opencv-python mediapipe numpy Pillow
Run: python air_writer_tkinter.py
"""

import cv2
import mediapipe as mp
import numpy as np
import random
import math
import time
import threading
import tkinter as tk
from tkinter import ttk, font as tkfont
from PIL import Image, ImageTk

# ── MediaPipe setup ───────────────────────────────────────────────
mp_hands       = mp.solutions.hands
mp_draw        = mp.solutions.drawing_utils
mp_draw_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    model_complexity=0
)

# ── Color palette ─────────────────────────────────────────────────
COLORS = {
    "Red":    (0,   0,   255),
    "Green":  (0,   255, 0  ),
    "Blue":   (255, 0,   0  ),
    "Yellow": (0,   255, 255),
    "Purple": (255, 0,   255),
    "Cyan":   (255, 255, 0  ),
    "White":  (255, 255, 255),
    "Orange": (0,   165, 255),
}
COLOR_HEX = {
    "Red":    "#FF3B3B",
    "Green":  "#3BFF6A",
    "Blue":   "#3B7FFF",
    "Yellow": "#FFE83B",
    "Purple": "#CC3BFF",
    "Cyan":   "#3BFFFF",
    "White":  "#FFFFFF",
    "Orange": "#FFA53B",
}

# ── Gesture helpers ───────────────────────────────────────────────
def finger_states(lm):
    return (
        lm[8].y  < lm[6].y,
        lm[12].y < lm[10].y,
        lm[16].y < lm[14].y,
        lm[20].y < lm[18].y,
    )

def detect_gesture(lm):
    i, m, r, p = finger_states(lm)
    thumb_tucked = (lm[4].x - lm[5].x) < 0
    if not i and not m and not r and not p: return "fist"
    if i and not m and not r and not p:     return "index"
    if i and m and not r and not p:         return "two_fingers"
    if i and m and r and not p:             return "three_fingers"
    if not i and not m and not r and p:     return "pinky"
    if i and m and r and p:
        return "four_fingers" if thumb_tucked else "open_palm"
    return "none"

PALM_PTS = [0, 1, 5, 9, 13, 17]
def palm_center(lm, w, h):
    cx = int(sum(lm[i].x for i in PALM_PTS) / len(PALM_PTS) * w)
    cy = int(sum(lm[i].y for i in PALM_PTS) / len(PALM_PTS) * h)
    return cx, cy

HOLD_DURATION = 1.0

# ── Main App ──────────────────────────────────────────────────────
class AirWriterApp:
    def __init__(self, root):
        self.root = root
        self.root.title("✏️ Air Writer")
        self.root.configure(bg="#0F0F1A")
        self.root.resizable(True, True)

        # State
        self.canvas_img    = None
        self.prev_point    = None
        self.mirror_mode   = False
        self.particle_mode = False
        self.particles     = []
        self.draw_color    = COLORS["Red"]
        self.draw_color_name = "Red"
        self.hold_gesture    = None
        self.hold_start_time = None
        self.hold_triggered  = False
        self.last_gesture    = "none"
        self.pinky_was_up    = False
        self.running         = True
        self.gesture_label_text = tk.StringVar(value="✋ No Hand")
        self.fps_text        = tk.StringVar(value="FPS: --")
        self.color_var       = tk.StringVar(value="Red")
        self.brush_size      = tk.IntVar(value=12)

        self._build_ui()
        self._start_capture()

    # ── UI Layout ─────────────────────────────────────────────────
    def _build_ui(self):
        # Left sidebar
        sidebar = tk.Frame(self.root, bg="#1A1A2E", width=220)
        sidebar.pack(side=tk.LEFT, fill=tk.Y, padx=0, pady=0)
        sidebar.pack_propagate(False)

        # Title
        title_font = tkfont.Font(family="Helvetica", size=15, weight="bold")
        tk.Label(sidebar, text="✏️ Air Writer", font=title_font,
                 bg="#1A1A2E", fg="#E0E0FF").pack(pady=(18, 4))
        tk.Label(sidebar, text="Draw with your hand!",
                 bg="#1A1A2E", fg="#6060AA", font=("Helvetica", 9)).pack(pady=(0, 14))

        ttk.Separator(sidebar, orient="horizontal").pack(fill=tk.X, padx=12, pady=4)

        # Gesture status
        tk.Label(sidebar, text="GESTURE", bg="#1A1A2E", fg="#6060AA",
                 font=("Helvetica", 8, "bold")).pack(anchor="w", padx=16, pady=(10, 2))
        self.gesture_lbl = tk.Label(sidebar, textvariable=self.gesture_label_text,
                                    bg="#252540", fg="#FFFFFF",
                                    font=("Helvetica", 12, "bold"),
                                    relief="flat", pady=6, padx=8)
        self.gesture_lbl.pack(fill=tk.X, padx=12, pady=(0, 10))

        # FPS
        tk.Label(sidebar, textvariable=self.fps_text,
                 bg="#1A1A2E", fg="#404070",
                 font=("Helvetica", 8)).pack(pady=(0, 4))

        ttk.Separator(sidebar, orient="horizontal").pack(fill=tk.X, padx=12, pady=4)

        # Color picker
        tk.Label(sidebar, text="COLOR", bg="#1A1A2E", fg="#6060AA",
                 font=("Helvetica", 8, "bold")).pack(anchor="w", padx=16, pady=(10, 6))
        color_grid = tk.Frame(sidebar, bg="#1A1A2E")
        color_grid.pack(padx=12, pady=(0, 8))
        for idx, (name, hexval) in enumerate(COLOR_HEX.items()):
            btn = tk.Button(color_grid, bg=hexval, width=3, height=1,
                            relief="flat", cursor="hand2",
                            command=lambda n=name: self._set_color(n))
            btn.grid(row=idx // 4, column=idx % 4, padx=3, pady=3)

        # Color indicator
        self.color_indicator = tk.Label(sidebar, text="  Red  ",
                                        bg=COLOR_HEX["Red"], fg="#000000",
                                        font=("Helvetica", 10, "bold"),
                                        relief="flat", pady=4)
        self.color_indicator.pack(fill=tk.X, padx=12, pady=(0, 10))

        # Brush size
        tk.Label(sidebar, text="BRUSH SIZE", bg="#1A1A2E", fg="#6060AA",
                 font=("Helvetica", 8, "bold")).pack(anchor="w", padx=16, pady=(4, 2))
        brush_frame = tk.Frame(sidebar, bg="#1A1A2E")
        brush_frame.pack(fill=tk.X, padx=12, pady=(0, 10))
        tk.Scale(brush_frame, from_=4, to=40, orient=tk.HORIZONTAL,
                 variable=self.brush_size, bg="#1A1A2E", fg="#FFFFFF",
                 highlightthickness=0, troughcolor="#252540",
                 activebackground="#3B3BFF", length=180).pack(fill=tk.X)

        ttk.Separator(sidebar, orient="horizontal").pack(fill=tk.X, padx=12, pady=4)

        # Toggles
        tk.Label(sidebar, text="MODES", bg="#1A1A2E", fg="#6060AA",
                 font=("Helvetica", 8, "bold")).pack(anchor="w", padx=16, pady=(10, 6))

        self.mirror_btn = tk.Button(sidebar, text="🪞 Mirror: OFF",
                                    bg="#252540", fg="#8888CC",
                                    font=("Helvetica", 10), relief="flat",
                                    cursor="hand2", pady=6,
                                    command=self._toggle_mirror)
        self.mirror_btn.pack(fill=tk.X, padx=12, pady=3)

        self.particle_btn = tk.Button(sidebar, text="✨ Particles: OFF",
                                      bg="#252540", fg="#8888CC",
                                      font=("Helvetica", 10), relief="flat",
                                      cursor="hand2", pady=6,
                                      command=self._toggle_particles)
        self.particle_btn.pack(fill=tk.X, padx=12, pady=3)

        # Clear button
        tk.Button(sidebar, text="🗑️ Clear Canvas",
                  bg="#3B1A1A", fg="#FF6666",
                  font=("Helvetica", 10, "bold"), relief="flat",
                  cursor="hand2", pady=8,
                  command=self._clear_canvas).pack(fill=tk.X, padx=12, pady=(16, 4))

        ttk.Separator(sidebar, orient="horizontal").pack(fill=tk.X, padx=12, pady=8)

        # Gesture guide
        tk.Label(sidebar, text="GESTURE GUIDE", bg="#1A1A2E", fg="#6060AA",
                 font=("Helvetica", 8, "bold")).pack(anchor="w", padx=16, pady=(0, 4))
        gestures = [
            ("☝️", "Index", "Draw"),
            ("✌️", "Two fingers", "Erase"),
            ("🤟", "Pinky", "Cycle color"),
            ("🖐️", "Open palm", "Clear"),
            ("🤘", "Three (hold)", "Particles"),
            ("🖖", "Four (hold)", "Mirror"),
        ]
        for icon, name, action in gestures:
            row = tk.Frame(sidebar, bg="#1A1A2E")
            row.pack(fill=tk.X, padx=12, pady=1)
            tk.Label(row, text=f"{icon} {name}", bg="#1A1A2E", fg="#AAAACC",
                     font=("Helvetica", 8), width=16, anchor="w").pack(side=tk.LEFT)
            tk.Label(row, text=action, bg="#1A1A2E", fg="#6060AA",
                     font=("Helvetica", 8)).pack(side=tk.RIGHT)

        # Video area
        self.video_label = tk.Label(self.root, bg="#0F0F1A", cursor="none")
        self.video_label.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=4, pady=4)

    # ── Controls ──────────────────────────────────────────────────
    def _set_color(self, name):
        self.draw_color = COLORS[name]
        self.draw_color_name = name
        hex_c = COLOR_HEX[name]
        self.color_indicator.config(text=f"  {name}  ", bg=hex_c,
                                    fg="#000000" if name in ("White", "Yellow", "Cyan") else "#FFFFFF")

    def _toggle_mirror(self):
        self.mirror_mode = not self.mirror_mode
        if self.mirror_mode:
            self.mirror_btn.config(text="🪞 Mirror: ON", bg="#1A3A1A", fg="#3BFF6A")
        else:
            self.mirror_btn.config(text="🪞 Mirror: OFF", bg="#252540", fg="#8888CC")

    def _toggle_particles(self):
        self.particle_mode = not self.particle_mode
        if self.particle_mode:
            self.particle_btn.config(text="✨ Particles: ON", bg="#2A1A3A", fg="#CC3BFF")
        else:
            self.particle_btn.config(text="✨ Particles: OFF", bg="#252540", fg="#8888CC")

    def _clear_canvas(self):
        if self.canvas_img is not None:
            self.canvas_img[:] = 0
        self.particles.clear()

    # ── Particle system ───────────────────────────────────────────
    def _spawn_particles(self, x, y):
        for _ in range(2):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 4)
            hue   = random.randint(0, 179)
            color = cv2.cvtColor(np.uint8([[[hue, 255, 255]]]),
                                 cv2.COLOR_HSV2BGR)[0][0].tolist()
            self.particles.append([
                float(x), float(y),
                random.randint(4, 10), color, 255.0,
                math.cos(angle) * speed, math.sin(angle) * speed,
            ])

    def _tick_particles(self, frame):
        dead = []
        for idx, p in enumerate(self.particles):
            p[0] += p[5]; p[1] += p[6]
            p[2] = max(0.0, p[2] - 0.2)
            p[4] = max(0.0, p[4] - 8.0)
            if p[4] <= 0 or p[2] <= 0:
                dead.append(idx); continue
            alpha   = p[4] / 255.0
            overlay = frame.copy()
            cv2.circle(overlay, (int(p[0]), int(p[1])), int(p[2]), p[3], -1)
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        for idx in reversed(dead):
            self.particles.pop(idx)

    # ── Hold logic ────────────────────────────────────────────────
    HOLDABLE = {"three_fingers", "four_fingers"}

    def _update_hold(self, gesture):
        if gesture != self.last_gesture:
            self.last_gesture    = gesture
            self.hold_triggered  = False
            if gesture in self.HOLDABLE:
                self.hold_gesture    = gesture
                self.hold_start_time = time.time()
            else:
                self.hold_gesture    = None
                self.hold_start_time = None

        if gesture not in self.HOLDABLE or self.hold_start_time is None:
            return 0.0, False

        elapsed  = time.time() - self.hold_start_time
        progress = min(elapsed / HOLD_DURATION, 1.0)

        if progress >= 1.0 and not self.hold_triggered:
            self.hold_triggered = True
            return 1.0, True

        return progress, False

    # ── Draw helpers ──────────────────────────────────────────────
    def _draw_line(self, canvas, p1, p2, color, thickness, w):
        cv2.line(canvas, p1, p2, color, thickness)
        if self.mirror_mode:
            cv2.line(canvas, (w - p1[0], p1[1]), (w - p2[0], p2[1]), color, thickness)

    def _draw_hold_arc(self, frame, cx, cy, progress, color):
        radius = 68
        angle  = int(360 * progress)
        prev   = None
        for deg in range(-90, -90 + angle, 3):
            rad = math.radians(deg)
            pt  = (int(cx + radius * math.cos(rad)), int(cy + radius * math.sin(rad)))
            if prev:
                cv2.line(frame, prev, pt, color, 3)
            prev = pt

    def _draw_palm_ring(self, frame, cx, cy, gesture, color, hold_progress=0.0):
        styles = {
            "index":         [(52, color, 2), (44, color, 1)],
            "two_fingers":   [(52, (255, 255, 0), 2), (44, (200, 200, 0), 1)],
            "three_fingers": [(52, (255, 0, 255), 2), (44, (200, 0, 200), 1)],
            "four_fingers":  [(52, (0, 255, 255), 2), (44, (0, 200, 200), 1)],
            "open_palm":     [(52, (0, 140, 255), 2), (44, (0, 100, 200), 1)],
            "fist":          [(52, (80, 80, 80), 1)],
            "pinky":         [(52, color, 2), (60, color, 1)],
        }
        for radius, col, thickness in styles.get(gesture, []):
            cv2.circle(frame, (cx, cy), radius, col, thickness)
        if hold_progress > 0.0:
            arc_color = (255, 0, 255) if gesture == "three_fingers" else (0, 255, 255)
            self._draw_hold_arc(frame, cx, cy, hold_progress, arc_color)

    # ── Capture thread ────────────────────────────────────────────
    def _start_capture(self):
        self.cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self._last_time = time.time()
        self._update_frame()

    def _update_frame(self):
        if not self.running:
            return

        ret, frame = self.cap.read()
        if not ret:
            self.root.after(30, self._update_frame)
            return

        frame = cv2.flip(frame, 1)
        h, w  = frame.shape[:2]

        if self.canvas_img is None or self.canvas_img.shape != frame.shape:
            self.canvas_img = np.zeros_like(frame)

        result  = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        gesture = "none"
        mode_text = "No Hand"

        if result.multi_hand_landmarks:
            lm = result.multi_hand_landmarks[0].landmark
            mp_draw.draw_landmarks(
                frame, result.multi_hand_landmarks[0],
                mp_hands.HAND_CONNECTIONS,
                mp_draw_styles.get_default_hand_landmarks_style(),
                mp_draw_styles.get_default_hand_connections_style(),
            )

            gesture = detect_gesture(lm)
            tip_x   = int(lm[8].x * w)
            tip_y   = int(lm[8].y * h)
            cx, cy  = palm_center(lm, w, h)
            hold_progress, just_triggered = self._update_hold(gesture)
            self._draw_palm_ring(frame, cx, cy, gesture, self.draw_color, hold_progress)

            if gesture == "fist":
                mode_text = "✊ Idle"
                self.prev_point = None; self.pinky_was_up = False

            elif gesture == "index":
                mode_text = "✏️ Drawing"
                brush = self.brush_size.get()
                cv2.circle(frame, (tip_x, tip_y), brush // 2, self.draw_color, -1)
                cv2.circle(frame, (tip_x, tip_y), brush // 2, (255, 255, 255), 1)
                if self.prev_point:
                    self._draw_line(self.canvas_img, self.prev_point,
                                    (tip_x, tip_y), self.draw_color, brush, w)
                if self.particle_mode:
                    self._spawn_particles(tip_x, tip_y)
                self.prev_point = (tip_x, tip_y); self.pinky_was_up = False

            elif gesture == "two_fingers":
                mode_text = "✌️ Erasing"
                self.prev_point = None; self.pinky_was_up = False
                ix, iy = int(lm[8].x * w),  int(lm[8].y * h)
                mx, my = int(lm[12].x * w), int(lm[12].y * h)
                ex = (ix + mx) // 2; ey = (iy + my) // 2
                spread = int(math.hypot(mx - ix, my - iy) // 2)
                radius = max(20, min(spread, 80))
                cv2.circle(self.canvas_img, (ex, ey), radius, (0, 0, 0), -1)
                cv2.circle(frame, (ex, ey), radius, (255, 255, 0), 2)

            elif gesture == "three_fingers":
                self.prev_point = None; self.pinky_was_up = False
                if just_triggered:
                    self._toggle_particles()
                    mode_text = "✨ Particles toggled!"
                else:
                    mode_text = "⏳ Hold for Particles..."

            elif gesture == "four_fingers":
                self.prev_point = None; self.pinky_was_up = False
                if just_triggered:
                    self._toggle_mirror()
                    mode_text = "🪞 Mirror toggled!"
                else:
                    mode_text = "⏳ Hold for Mirror..."

            elif gesture == "open_palm":
                mode_text = "🖐️ Cleared!"
                self.prev_point = None; self.pinky_was_up = False
                self._clear_canvas()

            elif gesture == "pinky":
                self.prev_point = None
                if not self.pinky_was_up:
                    names = list(COLORS.keys())
                    idx   = (names.index(self.draw_color_name) + 1) % len(names)
                    self._set_color(names[idx])
                    self.pinky_was_up = True
                mode_text = f"🎨 Color: {self.draw_color_name}"

            else:
                mode_text = "🤔 Unknown"
                self.prev_point = None; self.pinky_was_up = False

        else:
            self.prev_point = None; self.pinky_was_up = False
            self.hold_gesture = None; self.hold_start_time = None
            self.hold_triggered = False; self.last_gesture = "none"

        # Composite
        output = cv2.addWeighted(frame, 1.0, self.canvas_img, 1.0, 0)
        if self.particle_mode or self.particles:
            self._tick_particles(output)
        if self.mirror_mode:
            cv2.line(output, (w // 2, 0), (w // 2, h), (0, 255, 255), 1)

        # FPS
        now = time.time()
        fps = 1.0 / max(now - self._last_time, 1e-9)
        self._last_time = now

        # Update Tkinter labels
        self.gesture_label_text.set(mode_text)
        self.fps_text.set(f"FPS: {fps:.0f}")

        # Convert frame to Tkinter image
        rgb   = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
        pil   = Image.fromarray(rgb)
        # Resize to fit the window area (preserve aspect)
        photo = ImageTk.PhotoImage(image=pil)
        self.video_label.config(image=photo)
        self.video_label.image = photo

        self.root.after(1, self._update_frame)

    def on_close(self):
        self.running = False
        self.cap.release()
        hands.close()
        self.root.destroy()


# ── Entry point ───────────────────────────────────────────────────
if __name__ == "__main__":
    root = tk.Tk()
    root.geometry("1280x760")
    app  = AirWriterApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()
