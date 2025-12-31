

import sys
import traceback
# Import the hand detector early so MediaPipe native libs load before PyQt5
# This avoids a known DLL load conflict on Windows when Qt loads certain DLLs first.
from gestures import HandDetector
import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QLabel, QWidget, QVBoxLayout, QHBoxLayout, QSizePolicy
from PyQt5.QtGui import QImage, QPixmap, QPainter, QColor, QPen, QFont
from PyQt5.QtCore import QTimer, Qt

import ui_styles as style

class CanvasWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        # allow the canvas to stretch horizontally and take a larger vertical area
        self.setMinimumHeight(style.CANVAS_HEIGHT)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        # animation value (current) and target for smooth easing
        self.animation_val = 0.5  # 0..1
        self.target_animation = self.animation_val
        # easing factor per update (0..1). Larger = snappier, smaller = smoother/slower
        self.animation_ease = 0.18
        self.stack_offset = 18

    def update_from_gesture(self, gesture_info):
        # gesture_info comes from HandDetector.classify
        # map two_hands_distance or pinch to animation_val
        if gesture_info.get('two_hands_distance') is not None:
            d = gesture_info['two_hands_distance']
            # clamp and map to 0..1
            v = max(30.0, min(500.0, d))
            mapped = (v - 30.0) / (500.0 - 30.0)
            # for two-hands distance larger -> more open (mapped as-is)
            self.target_animation = mapped
        elif gesture_info['hands']:
            # if any pinch, use smallest pinch distance
            pinches = [h['pinch_dist'] for h in gesture_info['hands']]
            v = min(pinches) if pinches else 200
            v = max(10.0, min(300.0, v))
            # pinch smaller -> larger animation (invert)
            mapped = 1.0 - ((v - 10.0) / (300.0 - 10.0))
            self.target_animation = mapped
        else:
            # decay target slowly toward 0.5 (neutral)
            self.target_animation = 0.5

        # Easing step: move current value toward target smoothly
        # Using simple exponential ease: current += (target - current) * ease
        diff = self.target_animation - self.animation_val
        self.animation_val += diff * self.animation_ease

        # ensure bounds
        self.animation_val = max(0.0, min(1.0, self.animation_val))

        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(*style.BACKGROUND))
        w = self.width()
        h = self.height()
        center_x = w // 2
        center_y = h // 2
        # Draw stacked rectangles from left (pink/purple) to right (green)
        # More layers + stronger perspective to match reference visuals
        layers = 20
        # Base plane size relative to canvas height
        base_h = h * 0.48
        base_w = base_h * 1.25
        # spread increases with animation_val to fan out when open
        spread = int(self.stack_offset * (1 + self.animation_val * 3.0))

        # compute a pixel gap corresponding to ~1 cm on the display (fallback to 40 px)
        try:
            dpi = self.logicalDpiX()
            gap_px = max(24, int(dpi / 2.54))  # 1 cm ~ dpi/2.54
        except Exception:
            gap_px = 40

        # Compute the i=0 (nearest) rectangle widths for left and right so we can place their inner faces close
        # left base_scale at t=0 (matches code below)
        left_base_scale0 = (0.45 + 0.9 * (1 - 0) + 0.25 * self.animation_val * 0)
        left_uniform0 = (0.5 + 0.9 * self.animation_val)
        left_rect_w0 = base_w * left_base_scale0 * left_uniform0

        # right base_scale at t=0
        right_base_scale0 = (0.5 + 0.75 * 0 + 0.25 * self.animation_val * (1 - 0))
        right_uniform0 = (0.6 + 0.9 * self.animation_val)
        right_rect_w0 = base_w * right_base_scale0 * right_uniform0

        # position anchors so the nearest (i=0) faces are gap_px apart centered around center_x
        left_start_x = int(center_x - gap_px / 2 - left_rect_w0 / 2)
        right_start_x = int(center_x + gap_px / 2 + right_rect_w0 / 2)

        def draw_plane(painter, anchor_x, anchor_y, w_px, h_px, color, layer_idx=0, total_layers=1, tilt_sign=-1, shadow_alpha=60):
            """Draw a tilted plane centered at anchor (uses painter transform)."""
            painter.save()
            # rotation and shear vary by layer to create a fan/arc effect
            t_layer = layer_idx / max(1, (total_layers - 1))
            rot_base = -28  # stronger base rotation for more dramatic tilt
            rot = rot_base * tilt_sign * (0.5 + 1.0 * t_layer)
            painter.translate(anchor_x, anchor_y)
            painter.rotate(rot)
            # stronger shear to accent perspective and depth
            shear_amount = (0.36 + 0.26 * t_layer) * tilt_sign
            painter.shear(shear_amount, 0)
            # draw shadow slightly below the plane
            shadow = QColor(0, 0, 0, shadow_alpha)
            painter.setBrush(shadow)
            painter.setPen(Qt.NoPen)
            # shadow offset depends on size and layer depth (outer layers cast longer shadows)
            shadow_h = max(8, int(h_px * (0.08 + 0.05 * t_layer)))
            shadow_w = int(w_px * (0.9 + 0.25 * t_layer))
            shadow_x = int(-shadow_w/2 + w_px*0.08 * tilt_sign)
            # push shadow further for outer layers to mimic long cast shadows
            shadow_y = int(h_px/2 + shadow_h*0.9 + 8 * t_layer)
            painter.drawRect(shadow_x, shadow_y, shadow_w, shadow_h)
            # draw the colored plane
            painter.setBrush(color)
            painter.setPen(QPen(Qt.black, 1.25))
            painter.drawRect(int(-w_px/2), int(-h_px/2), int(w_px), int(h_px))
            painter.restore()

        # left stack: stronger gradient and per-layer variation (magenta/pink family)
        for i in range(layers):
            t = i / max(1, (layers - 1))
            # larger offsets so the stack fans outward more; amplify with animation (stretch)
            layer_offset = int(i * max(12, int(spread // 1.6 * (1 + self.animation_val * 1.8))))
            # slight vertical stagger to mimic perspective stacking
            y = center_y - base_h/2 + i * 4
            # richer magenta/pink gradient tuned to reference
            r = int(245 - t * 60)
            g = int(40 + t * 60)
            b = int(160 + t * 30)
            color = QColor(r, g, b)
            # compute a uniform scale factor (applied to both width and height)
            # when stretched, outer planes grow slightly and inner ones shrink to emphasize depth
            base_scale = (0.45 + 0.9 * (1 - t) + 0.25 * self.animation_val * t)
            uniform_scale = (0.5 + 0.9 * self.animation_val)
            rect_w = base_w * base_scale * uniform_scale
            rect_h = base_h * base_scale * uniform_scale
            # anchor x so rectangles expand/shrink around the anchor instead of pushing left/right
            anchor_x = left_start_x - layer_offset
            # draw tilted plane with stronger shadow; pass layer index
            # stronger outline when stretched so planes read clearly
            outline_w = 1.0 + 2.5 * self.animation_val
            # temporarily increase pen thickness in draw_plane by setting a stronger pen via color alpha
            draw_plane(painter, anchor_x, y + rect_h/5, rect_w, rect_h, color, layer_idx=i, total_layers=layers, tilt_sign=-1, shadow_alpha=int(120 + 80 * self.animation_val))

        # right stack: greens to blue gradient and mirrored tilt
        # right stack anchor: move further right when stretched
        right_start_x = center_x + int(base_w * (1.2 + 1.6 * self.animation_val))
        for i in range(layers):
            t = i / max(1, (layers - 1))
            layer_offset = int(i * max(12, int(spread // 1.6 * (1 + self.animation_val * 1.8))))
            y = center_y - base_h/2 + i * 4
            # gradient from green to blue tuned to reference
            r = int(30 + t * 50)
            g = int(200 - t * 100)
            b = int(110 + t * 140)
            color = QColor(r, g, b)
            # compute a uniform scale for the right stack as well; emphasize outer planes when stretched
            base_scale = (0.5 + 0.75 * t + 0.25 * self.animation_val * (1 - t))
            uniform_scale = (0.6 + 0.9 * self.animation_val)
            rect_w = base_w * base_scale * uniform_scale
            rect_h = base_h * base_scale * uniform_scale
            # anchor x so rectangles expand/shrink around the anchor instead of pushing left/right
            anchor_x = right_start_x + layer_offset
            # draw tilted plane mirrored to the right with slightly stronger shadow
            draw_plane(painter, anchor_x, y + rect_h/5, rect_w, rect_h, color, layer_idx=i, total_layers=layers, tilt_sign=1, shadow_alpha=int(120 + 80 * self.animation_val))

        painter.end()

class VideoWidget(QLabel):
    def __init__(self, w=style.WINDOW_WIDTH, h=style.VIDEO_HEIGHT, parent=None):
        super().__init__(parent)
        # keep a larger fixed height for the webcam area but allow horizontal expansion
        self.setFixedHeight(h)
        self.setMinimumWidth(w)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setStyleSheet('background-color: rgb(30,30,30);')
        self.setAlignment(Qt.AlignCenter)
        self.font = QFont('Arial', 13)

    def show_frame(self, frame, landmarks=None, gesture_info=None):
        # frame: BGR
        h, w, ch = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        bytes_per_line = ch * w
        qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        # Scale the pixmap to cover the widget area (crop center if needed)
        pixmap = QPixmap.fromImage(qimg)
        scaled = pixmap.scaled(self.width(), self.height(), Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation)
        # Crop center
        x_off = (scaled.width() - self.width()) // 2
        y_off = (scaled.height() - self.height()) // 2
        pix = scaled.copy(x_off, y_off, self.width(), self.height())

        # draw overlay
        painter = QPainter(pix)
        painter.setRenderHint(QPainter.Antialiasing)
        pen = QPen(QColor(*style.TEXT_COLOR), 2)
        painter.setPen(pen)
        painter.setFont(self.font)

        # draw landmarks if provided -- simplified: do not draw all landmarks
        # (user requested only a single connecting line + distance label)
        # We intentionally skip drawing per-landmark dots to avoid visual clutter.
        # Optionally, we could draw small wrist markers, but leaving them off keeps the view clean.

        # show a simple distance/pinch indicator if gesture_info present
        if gesture_info and gesture_info['hands']:
            g = gesture_info['hands'][0]
            text = ''
            if g['is_pinch']:
                text = f'pinch {int(g["pinch_dist"])} px'
            else:
                text = f'fingers up: {g["fingers_up"]}'
            painter.setPen(QColor(*style.TEXT_COLOR))
            painter.drawText(20, 30, text)

        # if two hands present, draw elastic/rubber connectors between corresponding landmarks
        if gesture_info and gesture_info.get('two_hands_distance') is not None and len(landmarks) >= 2:
            # Identify left and right hands (MediaPipe labels them 'Left'/'Right')
            left = None
            right = None
            for hand_item in landmarks:
                if hand_item.get('handedness', '').lower().startswith('l'):
                    left = hand_item
                elif hand_item.get('handedness', '').lower().startswith('r'):
                    right = hand_item
            # fallback: use index order if labels not present
            if left is None or right is None:
                left = landmarks[0]
                right = landmarks[1]

            # draw a single straight line between the two wrists (landmark 0) and label the pixel distance
            l_lms = left['landmarks']
            r_lms = right['landmarks']

            # use wrist landmark (index 0) as representative points for hands
            lx, ly, _ = l_lms[0]
            rx, ry, _ = r_lms[0]
            fx1 = int(lx * pix.width() / w)
            fy1 = int(ly * pix.height() / h)
            fx2 = int(rx * pix.width() / w)
            fy2 = int(ry * pix.height() / h)

            # draw single connecting line
            painter.setPen(QPen(QColor(255, 255, 255), 4, Qt.SolidLine, Qt.RoundCap))
            painter.drawLine(fx1, fy1, fx2, fy2)

            # compute and draw distance label near midpoint
            pd = int(((fx1 - fx2) ** 2 + (fy1 - fy2) ** 2) ** 0.5)
            mid_fx = (fx1 + fx2) // 2
            mid_fy = (fy1 + fy2) // 2
            painter.setPen(QPen(QColor(*style.TEXT_COLOR), 1))
            painter.drawText(mid_fx - 40, mid_fy - 10, f'distance: {pd} px')

        painter.end()
        self.setPixmap(pix)

class MainWindow(QWidget):
    def __init__(self, camera_index=0):
        super().__init__()
        self.setWindowTitle('exvapt - Hand Gesture UI')
        self.detector = HandDetector()
        self.cap = cv2.VideoCapture(camera_index)
        self.video_widget = VideoWidget()
        self.canvas = CanvasWidget()

        layout = QVBoxLayout()
        # tighten margins so top video and bottom canvas sit closer together
        layout.setContentsMargins(4,4,4,4)
        layout.setSpacing(4)
        layout.addWidget(self.video_widget)
        layout.addWidget(self.canvas)
        self.setLayout(layout)

        self.timer = QTimer()
        self.timer.timeout.connect(self._update_frame)
        self.timer.start(30)

    def _update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return
        hands = self.detector.find_hands(frame)
        gesture_info = self.detector.classify(hands)
        # update widgets
        try:
            self.video_widget.show_frame(frame, landmarks=hands, gesture_info=gesture_info)
            self.canvas.update_from_gesture(gesture_info)
        except Exception as e:
            print('UI render error:')
            traceback.print_exc()

    def closeEvent(self, event):
        try:
            self.cap.release()
        except Exception:
            pass
        event.accept()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    w = MainWindow()
    w.resize(style.WINDOW_WIDTH, style.VIDEO_HEIGHT + style.CANVAS_HEIGHT + 40)
    w.show()
    sys.exit(app.exec_())
