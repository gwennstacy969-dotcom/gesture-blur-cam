import cv2
import math
import time
import random
import os
import numpy as np
from collections import deque
from cvzone.HandTrackingModule import HandDetector

# ==========================================
#  Gesture Camera - Ultimate Edition (16 Gestures)
#  --- 1 Tangan ---
#  1. Peace Sign (V) = Blur Seluruh Layar
#  2. Jempol & Telunjuk = Blur dalam kotak
#  3. Rock Sign (Telunjuk+Kelingking) = Glitch/VHS
#  4. Telunjuk Saja = Spotlight
#  5. Kelingking Saja = Color Invert
#  6. Kepalan Tangan = Night Vision
#  7. Telapak Terbuka (1 tangan) = Freeze Frame
#  --- 2 Tangan ---
#  8. Segitiga △ (dua tangan) = Grayscale
#  9. Love/Heart ❤️ (dua tangan) = Efek Hati
# 10. Kotak □ (dua tangan terbuka) = Anime Filter
#  --- BARU ---
# 11. Jari Tengah Saja = Thermal Vision
# 12. Jari Manis Saja = Underwater Effect
# 13. Jempol Saja = Slow Motion Replay
# 14. Telunjuk+Tengah+Manis = Color Pop
# 15. Keyboard S = Screenshot
# 16. Keyboard R = Video Recording
# ==========================================

def distance(p1, p2):
    """Hitung jarak Euclidean antara dua titik."""
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

def angle_between(a, b, c):
    """Hitung sudut di titik b, dari vektor ba ke bc (dalam derajat)."""
    ba = (a[0] - b[0], a[1] - b[1])
    bc = (c[0] - b[0], c[1] - b[1])
    dot = ba[0] * bc[0] + ba[1] * bc[1]
    mag_ba = math.sqrt(ba[0]**2 + ba[1]**2)
    mag_bc = math.sqrt(bc[0]**2 + bc[1]**2)
    if mag_ba * mag_bc == 0:
        return 0
    cos_angle = max(-1, min(1, dot / (mag_ba * mag_bc)))
    return math.degrees(math.acos(cos_angle))

def is_triangle(p1, p2, p3, min_side=40, angle_range=(20, 160)):
    """
    Cek apakah 3 titik membentuk segitiga yang valid.
    - Semua sisi harus >= min_side
    - Semua sudut harus dalam range angle_range
    """
    d12 = distance(p1, p2)
    d23 = distance(p2, p3)
    d13 = distance(p1, p3)

    if d12 < min_side or d23 < min_side or d13 < min_side:
        return False

    a1 = angle_between(p2, p1, p3)
    a2 = angle_between(p1, p2, p3)
    a3 = angle_between(p1, p3, p2)

    for a in [a1, a2, a3]:
        if a < angle_range[0] or a > angle_range[1]:
            return False

    return True

def draw_hand_landmarks(frame, hand, color=(0, 255, 150)):
    """Gambar titik landmark dan koneksi tangan secara manual."""
    lmList = hand["lmList"]

    # Koneksi antar landmark (mengikuti standar MediaPipe)
    connections = [
        (0, 1), (1, 2), (2, 3), (3, 4),        # Jempol
        (0, 5), (5, 6), (6, 7), (7, 8),         # Telunjuk
        (0, 9), (9, 10), (10, 11), (11, 12),    # Tengah
        (0, 13), (13, 14), (14, 15), (15, 16),  # Manis
        (0, 17), (17, 18), (18, 19), (19, 20),  # Kelingking
        (5, 9), (9, 13), (13, 17),              # Penghubung pangkal jari
    ]

    # Warna gradient per ujung jari (BGR)
    finger_tip_colors = {
        4:  (0, 230, 255),   # Jempol - kuning
        8:  (255, 200, 0),   # Telunjuk - cyan
        12: (255, 0, 220),   # Tengah - magenta
        16: (0, 255, 100),   # Manis - hijau
        20: (0, 150, 255),   # Kelingking - oranye
    }

    # Gambar garis koneksi dengan efek glow (double line)
    for c in connections:
        x1, y1 = lmList[c[0]][0], lmList[c[0]][1]
        x2, y2 = lmList[c[1]][0], lmList[c[1]][1]
        # Outer glow (tebal, warna gelap)
        glow = (color[0] // 3, color[1] // 3, color[2] // 3)
        cv2.line(frame, (x1, y1), (x2, y2), glow, 5, cv2.LINE_AA)
        # Inner line (tipis, terang)
        cv2.line(frame, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)

    # Gambar titik landmark dengan gradient warna per jari
    for i, lm in enumerate(lmList):
        x, y = lm[0], lm[1]
        if i in finger_tip_colors:
            fc = finger_tip_colors[i]
            # Outer glow ring
            cv2.circle(frame, (x, y), 11, (fc[0]//3, fc[1]//3, fc[2]//3), 2, cv2.LINE_AA)
            # Filled circle warna jari
            cv2.circle(frame, (x, y), 7, fc, cv2.FILLED)
            # White border
            cv2.circle(frame, (x, y), 9, (255, 255, 255), 2, cv2.LINE_AA)
        else:
            cv2.circle(frame, (x, y), 4, color, cv2.FILLED)
            cv2.circle(frame, (x, y), 5, (color[0]//2, color[1]//2, color[2]//2), 1, cv2.LINE_AA)


# ==========================================
#  Heart/Love Gesture - Helper Functions
# ==========================================

def generate_heart_points(cx, cy, size, num_points=100):
    """
    Generate titik-titik membentuk hati menggunakan persamaan parametrik.
    Hati menghadap ke atas (seperti emoji ❤️).
    """
    points = []
    for i in range(num_points):
        t = 2 * math.pi * i / num_points
        # Persamaan parametrik hati
        x = 16 * (math.sin(t) ** 3)
        y = -(13 * math.cos(t) - 5 * math.cos(2*t) - 2 * math.cos(3*t) - math.cos(4*t))
        # Scale dan posisi
        px = int(cx + x * size / 17)
        py = int(cy + y * size / 17)
        points.append((px, py))
    return points

def draw_glowing_heart(frame, cx, cy, size, alpha=1.0, pulse=0.0):
    """
    Gambar hati bersinar dengan efek glow berlapis.
    pulse: 0.0-1.0 untuk efek denyut (pulsing).
    """
    overlay = frame.copy()
    pulse_size = size + int(pulse * size * 0.15)

    # Warna-warna hati (BGR): dari luar ke dalam semakin terang
    layers = [
        (pulse_size + 12, (40, 20, 120), 5),     # Outer glow - gelap
        (pulse_size + 8,  (60, 40, 180), 4),      # Mid glow
        (pulse_size + 4,  (80, 50, 220), 3),      # Inner glow
        (pulse_size,      (100, 80, 255), 2),      # Core - merah terang
        (pulse_size - 3,  (180, 130, 255), 2),     # Highlight - pink
    ]

    for lsize, color, thickness in layers:
        pts = generate_heart_points(cx, cy, lsize, num_points=120)
        if len(pts) >= 3:
            pts_array = np.array(pts, dtype=np.int32)
            cv2.polylines(overlay, [pts_array], True, color, thickness, cv2.LINE_AA)

    # Fill hati dengan warna semi-transparan
    fill_pts = generate_heart_points(cx, cy, pulse_size - 2, num_points=120)
    if len(fill_pts) >= 3:
        fill_array = np.array(fill_pts, dtype=np.int32)
        cv2.fillPoly(overlay, [fill_array], (80, 50, 200))

    # Blend overlay ke frame
    blend_alpha = 0.4 * alpha
    cv2.addWeighted(overlay, blend_alpha, frame, 1 - blend_alpha, 0, frame)

    # Gambar garis hati terang di atas (agar terlihat jelas)
    bright_pts = generate_heart_points(cx, cy, pulse_size, num_points=120)
    if len(bright_pts) >= 3:
        bright_array = np.array(bright_pts, dtype=np.int32)
        cv2.polylines(frame, [bright_array], True, (130, 100, 255), 2, cv2.LINE_AA)

    # Highlight kecil di bagian atas kiri hati (seperti pantulan cahaya)
    highlight_cx = cx - int(pulse_size * 0.25)
    highlight_cy = cy - int(pulse_size * 0.15)
    cv2.circle(frame, (highlight_cx, highlight_cy), max(2, int(pulse_size * 0.08)),
               (220, 200, 255), cv2.FILLED, cv2.LINE_AA)


class HeartParticle:
    """Partikel hati kecil yang melayang ke atas."""
    def __init__(self, x, y, frame_w, frame_h):
        self.x = float(x + random.randint(-40, 40))
        self.y = float(y + random.randint(-20, 20))
        self.vx = random.uniform(-1.5, 1.5)
        self.vy = random.uniform(-3.0, -1.0)
        self.life = 1.0  # 1.0 = baru lahir, 0.0 = mati
        self.decay = random.uniform(0.008, 0.02)
        self.size = random.randint(4, 12)
        self.frame_w = frame_w
        self.frame_h = frame_h
        # Variasi warna merah-pink (BGR)
        self.color = (
            random.randint(80, 180),   # B
            random.randint(50, 120),   # G
            random.randint(200, 255),  # R
        )
        self.is_sparkle = random.random() < 0.3  # 30% jadi sparkle

    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.vy -= 0.02  # sedikit percepatan ke atas
        self.vx *= 0.99  # perlambatan horizontal
        self.life -= self.decay
        self.size = max(1, int(self.size * (0.98 + self.life * 0.01)))

    def is_alive(self):
        return self.life > 0 and 0 <= self.x < self.frame_w and 0 <= self.y < self.frame_h

    def draw(self, frame):
        alpha = max(0.0, min(1.0, self.life))
        ix, iy = int(self.x), int(self.y)

        if self.is_sparkle:
            # Sparkle: bintang kecil berkilau
            spark_size = max(1, int(self.size * alpha))
            # Garis silang
            c = (
                int(self.color[0] * alpha + 255 * (1 - alpha)),
                int(self.color[1] * alpha + 255 * (1 - alpha)),
                int(self.color[2] * alpha),
            )
            cv2.line(frame, (ix - spark_size, iy), (ix + spark_size, iy), c, 1, cv2.LINE_AA)
            cv2.line(frame, (ix, iy - spark_size), (ix, iy + spark_size), c, 1, cv2.LINE_AA)
            # Diagonal
            d = max(1, spark_size // 2)
            cv2.line(frame, (ix - d, iy - d), (ix + d, iy + d), c, 1, cv2.LINE_AA)
            cv2.line(frame, (ix - d, iy + d), (ix + d, iy - d), c, 1, cv2.LINE_AA)
            # Titik tengah terang
            cv2.circle(frame, (ix, iy), max(1, spark_size // 3), (255, 255, 255), cv2.FILLED)
        else:
            # Mini heart
            s = max(2, int(self.size * alpha))
            pts = generate_heart_points(ix, iy, s, num_points=30)
            if len(pts) >= 3:
                pts_array = np.array(pts, dtype=np.int32)
                fill_color = (
                    int(self.color[0] * alpha),
                    int(self.color[1] * alpha),
                    int(self.color[2] * alpha),
                )
                cv2.fillPoly(frame, [pts_array], fill_color)
                # Outline tipis
                outline_color = (
                    min(255, int(fill_color[0] + 60)),
                    min(255, int(fill_color[1] + 60)),
                    min(255, int(fill_color[2] + 30)),
                )
                cv2.polylines(frame, [pts_array], True, outline_color, 1, cv2.LINE_AA)


def draw_heart_vignette(frame, alpha=0.3):
    """Tambah vignette merah/pink di pinggir frame saat love gesture aktif."""
    h, w = frame.shape[:2]
    overlay = np.zeros_like(frame, dtype=np.uint8)

    # Gradient dari pinggir (merah gelap) ke tengah (transparan)
    for i in range(min(80, h // 4)):
        intensity = int(60 * (1 - i / 80) * alpha)
        color = (max(0, intensity // 3), 0, max(0, intensity))
        # Atas
        cv2.line(overlay, (0, i), (w, i), color, 1)
        # Bawah
        cv2.line(overlay, (0, h - 1 - i), (w, h - 1 - i), color, 1)
        # Kiri
        cv2.line(overlay, (i, 0), (i, h), color, 1)
        # Kanan
        cv2.line(overlay, (w - 1 - i, 0), (w - 1 - i, h), color, 1)

    cv2.add(frame, overlay, frame)


def is_heart_gesture(hands, detector):
    """
    Deteksi gesture hati/love dari dua tangan.
    Hati dibentuk dengan:
    - Kedua ujung telunjuk saling bertemu di atas (puncak hati)
    - Kedua ujung jempol saling bertemu di bawah (dasar hati)
    ATAU sebaliknya.
    Jari-jari lain (tengah, manis, kelingking) harus menekuk.

    Returns: (detected: bool, center: tuple, size: int)
    """
    if len(hands) != 2:
        return False, None, 0

    lm1 = hands[0]["lmList"]
    lm2 = hands[1]["lmList"]

    # Ujung jari kedua tangan
    thumb1 = (lm1[4][0], lm1[4][1])
    index1 = (lm1[8][0], lm1[8][1])
    thumb2 = (lm2[4][0], lm2[4][1])
    index2 = (lm2[8][0], lm2[8][1])

    # Pangkal jari (wrist) untuk menentukan orientasi
    wrist1 = (lm1[0][0], lm1[0][1])
    wrist2 = (lm2[0][0], lm2[0][1])

    # Cek jari-jari lain menekuk (tengah, manis, kelingking)
    f1 = detector.fingersUp(hands[0])
    f2 = detector.fingersUp(hands[1])

    # Untuk love gesture: telunjuk & jempol bisa UP, sisanya DOWN
    # f1/f2 = [thumb, index, middle, ring, pinky]
    other_fingers_down_1 = (f1[2] == 0 and f1[3] == 0 and f1[4] == 0)
    other_fingers_down_2 = (f2[2] == 0 and f2[3] == 0 and f2[4] == 0)

    if not (other_fingers_down_1 and other_fingers_down_2):
        return False, None, 0

    thresh = 80  # Jarak ujung jari harus dekat

    # Skenario 1: Telunjuk bertemu di atas, Jempol bertemu di bawah
    index_dist = distance(index1, index2)
    thumb_dist = distance(thumb1, thumb2)

    if index_dist < thresh and thumb_dist < thresh:
        # Titik pertemuan
        top = ((index1[0] + index2[0]) // 2, (index1[1] + index2[1]) // 2)
        bottom = ((thumb1[0] + thumb2[0]) // 2, (thumb1[1] + thumb2[1]) // 2)

        # Center dan ukuran hati
        cx = (top[0] + bottom[0]) // 2
        cy = (top[1] + bottom[1]) // 2
        heart_size = max(30, int(distance(top, bottom) * 0.7))

        return True, (cx, cy), heart_size

    # Skenario 2: Jempol bertemu di atas, Telunjuk bertemu di bawah
    if thumb_dist < thresh and index_dist < thresh * 1.5:
        top = ((thumb1[0] + thumb2[0]) // 2, (thumb1[1] + thumb2[1]) // 2)
        bottom = ((index1[0] + index2[0]) // 2, (index1[1] + index2[1]) // 2)

        cx = (top[0] + bottom[0]) // 2
        cy = (top[1] + bottom[1]) // 2
        heart_size = max(30, int(distance(top, bottom) * 0.7))

        return True, (cx, cy), heart_size

    # Skenario 3: Ujung telunjuk 1 dekat ujung jempol 2 DAN sebaliknya
    # (bentuk hati alternatif)
    cross_dist_1 = distance(index1, thumb2)
    cross_dist_2 = distance(index2, thumb1)

    if cross_dist_1 < thresh and cross_dist_2 < thresh:
        # Titik pertemuan atas dan bawah
        top1 = ((index1[0] + thumb2[0]) // 2, (index1[1] + thumb2[1]) // 2)
        top2 = ((index2[0] + thumb1[0]) // 2, (index2[1] + thumb1[1]) // 2)

        cx = (top1[0] + top2[0]) // 2
        cy = (top1[1] + top2[1]) // 2
        heart_size = max(30, int(distance(top1, top2) * 0.8))

        return True, (cx, cy), heart_size

    return False, None, 0


# ==========================================
#  Rectangle/Anime Gesture - Helper Functions
# ==========================================

def apply_anime_filter(roi):
    """
    Terapkan filter anime/kartun ke region of interest.
    Teknik: bilateral filter + edge detection + color quantization.
    Hasilnya seperti gambar anime/manga.
    """
    if roi.size == 0:
        return roi

    h_roi, w_roi = roi.shape[:2]
    if h_roi < 10 or w_roi < 10:
        return roi

    # Step 1: Bilateral filter → kulit halus seperti anime
    smooth = cv2.bilateralFilter(roi, 9, 75, 75)
    smooth = cv2.bilateralFilter(smooth, 9, 75, 75)

    # Step 2: Color quantization → warna flat seperti anime
    div = 24
    smooth = (smooth // div) * div + div // 2

    # Step 3: Edge detection → garis tebal seperti manga
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 7)
    edges = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY,
        blockSize=9, C=2
    )
    edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    # Step 4: Gabungkan warna flat + garis tepi
    cartoon = cv2.bitwise_and(smooth, edges_bgr)

    # Step 5: Boost saturasi → warna vibrant ala anime
    hsv = cv2.cvtColor(cartoon, cv2.COLOR_BGR2HSV)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1].astype(np.float32) * 1.5, 0, 255).astype(np.uint8)
    # Sedikit cerahkan
    hsv[:, :, 2] = np.clip(hsv[:, :, 2].astype(np.float32) * 1.1, 0, 255).astype(np.uint8)
    cartoon = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    return cartoon


def is_rectangle_gesture(hands, detector):
    """
    Deteksi gesture kotak/rectangle dari dua tangan.
    Kedua tangan harus membuka semua jari (telapak terbuka).
    Membentuk "frame" kotak dari posisi kedua tangan.

    Returns: (detected: bool, rect_coords: tuple(x_min, y_min, x_max, y_max))
    """
    if len(hands) != 2:
        return False, None

    f1 = detector.fingersUp(hands[0])
    f2 = detector.fingersUp(hands[1])

    # Kedua tangan harus buka semua jari (open palm)
    # Minimal 4 dari 5 jari harus UP di masing-masing tangan
    if sum(f1) < 4 or sum(f2) < 4:
        return False, None

    lm1 = hands[0]["lmList"]
    lm2 = hands[1]["lmList"]

    # Kumpulkan semua titik ujung jari dari kedua tangan
    tips_1 = [(lm1[i][0], lm1[i][1]) for i in [4, 8, 12, 16, 20]]
    tips_2 = [(lm2[i][0], lm2[i][1]) for i in [4, 8, 12, 16, 20]]

    # Tambah wrist untuk batas bawah yang lebih baik
    all_points = tips_1 + tips_2 + [(lm1[0][0], lm1[0][1]), (lm2[0][0], lm2[0][1])]

    all_x = [p[0] for p in all_points]
    all_y = [p[1] for p in all_points]

    x_min = min(all_x)
    x_max = max(all_x)
    y_min = min(all_y)
    y_max = max(all_y)

    # Pastikan kotak cukup besar
    rect_w = x_max - x_min
    rect_h = y_max - y_min
    if rect_w < 80 or rect_h < 80:
        return False, None

    # Pastikan kedua tangan cukup berjauhan (bukan numpuk)
    cx1 = sum(p[0] for p in tips_1) / len(tips_1)
    cx2 = sum(p[0] for p in tips_2) / len(tips_2)
    hand_separation = abs(cx1 - cx2)
    if hand_separation < 60:
        return False, None

    return True, (x_min, y_min, x_max, y_max)


def draw_anime_frame(frame, x_min, y_min, x_max, y_max, alpha=1.0):
    """
    Gambar border kotak bergaya anime/neon di sekitar area anime.
    """
    # Warna cyan-biru neon (BGR)
    color_outer = (int(200 * alpha), int(180 * alpha), int(50 * alpha))
    color_inner = (int(255 * alpha), int(255 * alpha), int(100 * alpha))
    corner_color = (int(100 * alpha), int(255 * alpha), int(255 * alpha))

    # Outer glow
    cv2.rectangle(frame, (x_min - 2, y_min - 2), (x_max + 2, y_max + 2),
                  color_outer, 3, cv2.LINE_AA)
    # Inner line
    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max),
                  color_inner, 1, cv2.LINE_AA)

    # Corner markers (L-shaped corners)
    corner_len = min(25, (x_max - x_min) // 5, (y_max - y_min) // 5)
    corners = [
        (x_min, y_min),  # Top-left
        (x_max, y_min),  # Top-right
        (x_min, y_max),  # Bottom-left
        (x_max, y_max),  # Bottom-right
    ]
    for i, (cx, cy) in enumerate(corners):
        dx = corner_len if (i % 2 == 0) else -corner_len
        dy = corner_len if (i < 2) else -corner_len
        cv2.line(frame, (cx, cy), (cx + dx, cy), corner_color, 2, cv2.LINE_AA)
        cv2.line(frame, (cx, cy), (cx, cy + dy), corner_color, 2, cv2.LINE_AA)

    # Label "ANIME" di atas kotak
    label_y = max(y_min - 8, 15)
    cv2.putText(frame, "ANIME", (x_min, label_y), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, corner_color, 1, cv2.LINE_AA)


# ==========================================
#  Night Vision Effect
# ==========================================

def apply_night_vision(frame, level):
    """Efek night vision hijau ala militer dengan scanline dan noise grain."""
    if level <= 0:
        return frame
    h, w = frame.shape[:2]

    # Convert ke grayscale dan boost brightness
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = np.clip(gray.astype(np.float32) * 1.3 + 10, 0, 255).astype(np.uint8)

    # Buat frame night vision (dominan hijau)
    night = np.zeros_like(frame)
    night[:, :, 0] = (gray * 0.05).astype(np.uint8)   # Blue minimal
    night[:, :, 1] = gray                               # Green dominan
    night[:, :, 2] = (gray * 0.08).astype(np.uint8)    # Red minimal

    # Noise grain
    noise = np.random.randint(0, 25, (h, w), dtype=np.uint8)
    night[:, :, 1] = np.clip(night[:, :, 1].astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # Scanlines (setiap baris genap jadi lebih gelap)
    night[::2, :] = (night[::2, :].astype(np.float32) * 0.75).astype(np.uint8)

    # Vignette gelap di pinggir
    cx_v, cy_v = w // 2, h // 2
    Y, X = np.ogrid[:h, :w]
    max_dist = math.sqrt(cx_v**2 + cy_v**2)
    dist = np.sqrt((X.astype(np.float32) - cx_v)**2 + (Y.astype(np.float32) - cy_v)**2)
    vignette = np.clip(1.0 - (dist / max_dist) * 0.7, 0.3, 1.0).astype(np.float32)
    for c in range(3):
        night[:, :, c] = (night[:, :, c].astype(np.float32) * vignette).astype(np.uint8)

    # Blend
    return cv2.addWeighted(night, level, frame, 1 - level, 0)


def draw_night_vision_hud(frame, level, blink_on=True):
    """Gambar HUD overlay night vision: crosshair, corner brackets, text."""
    if level < 0.3:
        return
    h, w = frame.shape[:2]
    alpha = min(1.0, level)
    color = (0, int(200 * alpha), 0)

    # Corner brackets (detail)
    blen = 40
    cv2.line(frame, (15, 15), (15 + blen, 15), color, 2, cv2.LINE_AA)
    cv2.line(frame, (15, 15), (15, 15 + blen), color, 2, cv2.LINE_AA)
    cv2.line(frame, (w - 15, 15), (w - 15 - blen, 15), color, 2, cv2.LINE_AA)
    cv2.line(frame, (w - 15, 15), (w - 15, 15 + blen), color, 2, cv2.LINE_AA)
    cv2.line(frame, (15, h - 15), (15 + blen, h - 15), color, 2, cv2.LINE_AA)
    cv2.line(frame, (15, h - 15), (15, h - 15 - blen), color, 2, cv2.LINE_AA)
    cv2.line(frame, (w - 15, h - 15), (w - 15 - blen, h - 15), color, 2, cv2.LINE_AA)
    cv2.line(frame, (w - 15, h - 15), (w - 15, h - 15 - blen), color, 2, cv2.LINE_AA)

    # Scope reticle di tengah (lingkaran konsentris + garis silang)
    cx, cy = w // 2, h // 2
    dim = (0, int(100 * alpha), 0)
    for r in [30, 60, 90]:
        cv2.circle(frame, (cx, cy), r, dim, 1, cv2.LINE_AA)
    # Garis silang dengan gap di tengah
    gap = 15
    cv2.line(frame, (cx - 90, cy), (cx - gap, cy), color, 1, cv2.LINE_AA)
    cv2.line(frame, (cx + gap, cy), (cx + 90, cy), color, 1, cv2.LINE_AA)
    cv2.line(frame, (cx, cy - 90), (cx, cy - gap), color, 1, cv2.LINE_AA)
    cv2.line(frame, (cx, cy + gap), (cx, cy + 90), color, 1, cv2.LINE_AA)
    cv2.circle(frame, (cx, cy), 2, color, cv2.FILLED)
    # Tick marks pada crosshair
    for d in [30, 60]:
        tick = 5
        cv2.line(frame, (cx - d, cy - tick), (cx - d, cy + tick), dim, 1, cv2.LINE_AA)
        cv2.line(frame, (cx + d, cy - tick), (cx + d, cy + tick), dim, 1, cv2.LINE_AA)
        cv2.line(frame, (cx - tick, cy - d), (cx + tick, cy - d), dim, 1, cv2.LINE_AA)
        cv2.line(frame, (cx - tick, cy + d), (cx + tick, cy + d), dim, 1, cv2.LINE_AA)

    # Labels kiri bawah
    cv2.putText(frame, "NV MODE", (20, h - 45), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, color, 1, cv2.LINE_AA)
    time_str = time.strftime("%H:%M:%S")
    cv2.putText(frame, time_str, (20, h - 25), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, color, 1, cv2.LINE_AA)

    # "RECORDING" blink indicator (pojok kiri atas)
    if blink_on:
        cv2.circle(frame, (30, 30), 5, (0, 0, int(180 * alpha)), cv2.FILLED)
        cv2.putText(frame, "RECORDING", (42, 35), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, color, 1, cv2.LINE_AA)

    # Mini compass (pojok kanan atas)
    comp_cx, comp_cy = w - 50, 50
    comp_r = 22
    cv2.circle(frame, (comp_cx, comp_cy), comp_r, dim, 1, cv2.LINE_AA)
    cv2.putText(frame, "N", (comp_cx - 4, comp_cy - comp_r - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1, cv2.LINE_AA)
    cv2.putText(frame, "S", (comp_cx - 3, comp_cy + comp_r + 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.3, dim, 1, cv2.LINE_AA)
    cv2.putText(frame, "E", (comp_cx + comp_r + 5, comp_cy + 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.3, dim, 1, cv2.LINE_AA)
    cv2.putText(frame, "W", (comp_cx - comp_r - 15, comp_cy + 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.3, dim, 1, cv2.LINE_AA)
    # Compass needle
    cv2.line(frame, (comp_cx, comp_cy), (comp_cx, comp_cy - comp_r + 5),
             color, 2, cv2.LINE_AA)
    cv2.line(frame, (comp_cx, comp_cy), (comp_cx, comp_cy + comp_r - 8),
             dim, 1, cv2.LINE_AA)

    # Distance meter (kanan bawah) - slowly oscillating
    dist_val = 15.0 + 20.0 * math.sin(time.time() * 0.3)
    cv2.putText(frame, f"DIST: {dist_val:.1f}m", (w - 150, h - 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
    elev_val = 1.5 + 0.5 * math.sin(time.time() * 0.5)
    cv2.putText(frame, f"ELEV: {elev_val:.1f}m", (w - 150, h - 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, dim, 1, cv2.LINE_AA)


# ==========================================
#  Glitch / VHS Effect
# ==========================================

def apply_glitch_effect(frame, level, time_val=0):
    """Efek glitch/VHS retro: RGB split, horizontal bars, scanlines."""
    if level <= 0:
        return frame
    h, w = frame.shape[:2]
    result = frame.copy()

    # Intensity oscillation (bergelombang otomatis)
    osc = 0.6 + 0.4 * math.sin(time_val * 8.0)
    eff_level = level * osc

    # RGB channel split (chromatic aberration)
    shift = max(1, int(eff_level * random.randint(5, 18)))
    if shift < w:
        # Red channel geser kanan
        result[:, shift:, 2] = frame[:, :w - shift, 2]
        result[:, :shift, 2] = 0
        # Blue channel geser kiri
        result[:, :w - shift, 0] = frame[:, shift:, 0]
        result[:, w - shift:, 0] = 0

    # Green-magenta vertical shift
    g_shift = max(1, int(eff_level * random.randint(2, 8)))
    if g_shift < h:
        result[g_shift:, :, 1] = frame[:h - g_shift, :, 1]

    # Random horizontal displacement bars
    num_bars = int(eff_level * random.randint(3, 10))
    for _ in range(num_bars):
        y_start = random.randint(0, h - 1)
        bar_h = random.randint(2, max(3, int(15 * eff_level)))
        y_end = min(y_start + bar_h, h)
        shift_x = random.randint(int(-30 * eff_level), int(30 * eff_level))
        if shift_x > 0 and shift_x < w:
            result[y_start:y_end, shift_x:] = frame[y_start:y_end, :w - shift_x]
        elif shift_x < 0 and abs(shift_x) < w:
            result[y_start:y_end, :w + shift_x] = frame[y_start:y_end, -shift_x:]

    # VHS scanlines
    result[::3, :] = (result[::3, :].astype(np.float32) * 0.85).astype(np.uint8)

    # Random color tint flicker
    if random.random() < 0.3 * eff_level:
        tint = np.zeros_like(result)
        tint[:, :] = (random.randint(0, 20), 0, random.randint(0, 25))
        result = cv2.add(result, tint)

    # Random horizontal white noise bar
    if random.random() < 0.4 * eff_level:
        y_noise = random.randint(0, h - 3)
        noise_h = random.randint(1, 4)
        y_end_n = min(y_noise + noise_h, h)
        noise_bar = np.random.randint(0, 255, (y_end_n - y_noise, w, 3), dtype=np.uint8)
        alpha_noise = 0.3 * eff_level
        result[y_noise:y_end_n] = cv2.addWeighted(
            noise_bar, alpha_noise, result[y_noise:y_end_n], 1 - alpha_noise, 0
        )

    # Screen shake (geser frame sedikit secara acak)
    if eff_level > 0.5:
        shake_x = random.randint(-3, 3)
        shake_y = random.randint(-2, 2)
        M = np.float32([[1, 0, shake_x], [0, 1, shake_y]])
        result = cv2.warpAffine(result, M, (w, h))

    return cv2.addWeighted(result, level, frame, 1 - level, 0)


def draw_glitch_hud(frame, level, blink_on=True):
    """Gambar overlay VHS-style: REC indicator, timestamp."""
    if level < 0.3:
        return
    h, w = frame.shape[:2]
    alpha = min(1.0, level)

    # "REC" indicator dengan titik merah berkedip
    txt_color = (int(200 * alpha), int(200 * alpha), int(200 * alpha))
    if blink_on:
        rec_color = (0, 0, int(255 * alpha))
        cv2.circle(frame, (25, 25), 6, rec_color, cv2.FILLED)
    cv2.putText(frame, "REC", (38, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, txt_color, 1, cv2.LINE_AA)

    # "PLAY" di pojok kanan atas
    cv2.putText(frame, "PLAY >>", (w - 110, 25), cv2.FONT_HERSHEY_SIMPLEX,
                0.4, txt_color, 1, cv2.LINE_AA)

    # VHS tracking line (garis horizontal bergerak)
    tracking_y = int((time.time() * 100) % h)
    track_color = (int(100 * alpha), int(100 * alpha), int(100 * alpha))
    cv2.line(frame, (0, tracking_y), (w, tracking_y), track_color, 1)

    # VHS timestamp di bawah
    time_str = time.strftime("%Y/%m/%d  %H:%M:%S")
    ts_color = (int(200 * alpha), int(200 * alpha), int(80 * alpha))
    cv2.putText(frame, time_str, (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX,
                0.4, ts_color, 1, cv2.LINE_AA)

    # SP mode indicator
    cv2.putText(frame, "SP", (w - 40, h - 15), cv2.FONT_HERSHEY_SIMPLEX,
                0.4, txt_color, 1, cv2.LINE_AA)


# ==========================================
#  Spotlight Effect
# ==========================================

def apply_spotlight(frame, cx, cy, radius, level):
    """Efek spotlight: area terang mengikuti posisi jari, sisanya gelap."""
    if level <= 0:
        return frame
    h, w = frame.shape[:2]

    # Buat frame gelap
    dark = (frame.astype(np.float32) * 0.12).astype(np.uint8)

    # Buat mask gradien lingkaran
    Y, X = np.ogrid[:h, :w]
    dist = np.sqrt((X.astype(np.float32) - cx)**2 + (Y.astype(np.float32) - cy)**2)

    # Smooth falloff
    mask = np.clip(1.0 - (dist / max(1, radius)), 0, 1).astype(np.float32)
    mask = mask ** 1.5  # Smooth edge
    mask_3ch = np.stack([mask, mask, mask], axis=-1)

    # Blend: terang di tengah, gelap di luar
    spotlight = (frame.astype(np.float32) * mask_3ch +
                 dark.astype(np.float32) * (1 - mask_3ch)).astype(np.uint8)

    # Blend dengan original sesuai level
    return cv2.addWeighted(spotlight, level, frame, 1 - level, 0)


def draw_spotlight_ring(frame, cx, cy, radius, level):
    """Gambar ring terang di sekitar spotlight."""
    if level < 0.3:
        return
    alpha = min(1.0, level)
    color = (int(200 * alpha), int(200 * alpha), int(100 * alpha))
    dim = (int(80 * alpha), int(80 * alpha), int(40 * alpha))

    # Multiple concentric rings
    cv2.circle(frame, (cx, cy), radius, color, 1, cv2.LINE_AA)
    cv2.circle(frame, (cx, cy), radius + 4, dim, 1, cv2.LINE_AA)
    cv2.circle(frame, (cx, cy), max(1, radius - 8), dim, 1, cv2.LINE_AA)

    # Crosshair dengan gap
    csize = 10
    gap = 4
    cv2.line(frame, (cx - csize, cy), (cx - gap, cy), color, 1, cv2.LINE_AA)
    cv2.line(frame, (cx + gap, cy), (cx + csize, cy), color, 1, cv2.LINE_AA)
    cv2.line(frame, (cx, cy - csize), (cx, cy - gap), color, 1, cv2.LINE_AA)
    cv2.line(frame, (cx, cy + gap), (cx, cy + csize), color, 1, cv2.LINE_AA)

    # Radial tick marks (8 arah)
    for angle_deg in range(0, 360, 45):
        rad = math.radians(angle_deg)
        inner_r = radius - 5
        outer_r = radius + 2
        x1 = int(cx + inner_r * math.cos(rad))
        y1 = int(cy + inner_r * math.sin(rad))
        x2 = int(cx + outer_r * math.cos(rad))
        y2 = int(cy + outer_r * math.sin(rad))
        cv2.line(frame, (x1, y1), (x2, y2), dim, 1, cv2.LINE_AA)

    # Label
    cv2.putText(frame, "SPOTLIGHT", (cx - 35, cy - radius - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
    # Coordinates display
    coord_str = f"({cx},{cy})"
    cv2.putText(frame, coord_str, (cx - 25, cy + radius + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.3, dim, 1, cv2.LINE_AA)


# ==========================================
#  Color Invert / Negative Effect
# ==========================================

def apply_color_invert(frame, level, frame_count=0):
    """Inversi warna (efek negatif) dengan scanline ala film."""
    if level <= 0:
        return frame

    # Invert warna
    inverted = cv2.bitwise_not(frame)

    # Blend
    result = cv2.addWeighted(inverted, level, frame, 1 - level, 0)

    # Film grain noise
    if level > 0.3:
        h_r, w_r = result.shape[:2]
        grain = np.random.randint(0, int(30 * level), (h_r, w_r), dtype=np.uint8)
        grain_bgr = cv2.cvtColor(grain, cv2.COLOR_GRAY2BGR)
        result = cv2.add(result, grain_bgr)

    # Scanlines tipis ala film negatif
    if level > 0.3:
        result[::4, :] = (result[::4, :].astype(np.float32) *
                          (0.85 + 0.15 * (1 - level))).astype(np.uint8)

    return result


def draw_invert_border(frame, level, frame_count=0):
    """Gambar border film negatif dengan perforasi."""
    if level < 0.3:
        return
    h, w = frame.shape[:2]
    alpha = min(1.0, level)
    color = (int(180 * alpha), int(120 * alpha), int(255 * alpha))

    # Border ganda
    cv2.rectangle(frame, (5, 5), (w - 5, h - 5), color, 1, cv2.LINE_AA)
    cv2.rectangle(frame, (8, 8), (w - 8, h - 8),
                  (int(80*alpha), int(60*alpha), int(120*alpha)), 1)

    # Film perforations bergerak (animated sprocket holes)
    perf_w = 8
    perf_h = 14
    perf_gap = 28
    offset = int(frame_count * 2) % perf_gap  # Animasi bergerak
    for y in range(-perf_gap + offset, h + perf_gap, perf_gap):
        if 0 <= y < h - perf_h:
            # Kiri
            cv2.rectangle(frame, (1, y), (1 + perf_w, y + perf_h), color, 1)
            cv2.rectangle(frame, (3, y + 2), (perf_w - 1, y + perf_h - 2),
                          (int(40*alpha), int(30*alpha), int(60*alpha)), cv2.FILLED)
            # Kanan
            cv2.rectangle(frame, (w - 1 - perf_w, y), (w - 1, y + perf_h), color, 1)
            cv2.rectangle(frame, (w - perf_w + 1, y + 2), (w - 3, y + perf_h - 2),
                          (int(40*alpha), int(30*alpha), int(60*alpha)), cv2.FILLED)

    # Label "NEGATIVE" di atas
    cv2.putText(frame, "NEGATIVE", (w // 2 - 45, 22), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, color, 1, cv2.LINE_AA)

    # Frame counter di pojok kanan bawah
    counter_str = f"F:{frame_count:06d}"
    cv2.putText(frame, counter_str, (w - 100, h - 12), cv2.FONT_HERSHEY_SIMPLEX,
                0.35, color, 1, cv2.LINE_AA)


# ==========================================
#  🔥 Thermal Vision Effect (BARU - Fitur 11)
# ==========================================

def apply_thermal_vision(frame, level):
    """Efek thermal/infrared vision menggunakan colormap JET."""
    if level <= 0:
        return frame
    h, w = frame.shape[:2]

    # Convert ke grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Boost kontras dengan CLAHE (agar detail panas terlihat jelas)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # Apply colormap JET (biru=dingin, merah=panas)
    thermal = cv2.applyColorMap(gray, cv2.COLORMAP_JET)

    # Tambahkan sedikit blur untuk efek "heat diffusion"
    thermal = cv2.GaussianBlur(thermal, (3, 3), 0)

    # Noise grain kecil untuk efek sensor thermal
    noise = np.random.randint(0, int(8 * level), (h, w), dtype=np.uint8)
    noise_bgr = cv2.cvtColor(noise, cv2.COLOR_GRAY2BGR)
    thermal = cv2.add(thermal, noise_bgr)

    # Blend sesuai level
    return cv2.addWeighted(thermal, level, frame, 1 - level, 0)


def draw_thermal_hud(frame, level):
    """Gambar HUD thermal vision: color scale bar, crosshair, suhu palsu."""
    if level < 0.3:
        return
    h, w = frame.shape[:2]
    alpha = min(1.0, level)

    # Warna HUD thermal (kuning-putih)
    color = (int(100 * alpha), int(220 * alpha), int(255 * alpha))
    dim = (int(50 * alpha), int(110 * alpha), int(128 * alpha))

    # Corner brackets thermal style
    blen = 30
    cv2.line(frame, (10, 10), (10 + blen, 10), color, 2, cv2.LINE_AA)
    cv2.line(frame, (10, 10), (10, 10 + blen), color, 2, cv2.LINE_AA)
    cv2.line(frame, (w - 10, 10), (w - 10 - blen, 10), color, 2, cv2.LINE_AA)
    cv2.line(frame, (w - 10, 10), (w - 10, 10 + blen), color, 2, cv2.LINE_AA)
    cv2.line(frame, (10, h - 10), (10 + blen, h - 10), color, 2, cv2.LINE_AA)
    cv2.line(frame, (10, h - 10), (10, h - 10 - blen), color, 2, cv2.LINE_AA)
    cv2.line(frame, (w - 10, h - 10), (w - 10 - blen, h - 10), color, 2, cv2.LINE_AA)
    cv2.line(frame, (w - 10, h - 10), (w - 10, h - 10 - blen), color, 2, cv2.LINE_AA)

    # Crosshair di tengah
    cx, cy = w // 2, h // 2
    gap = 12
    cv2.line(frame, (cx - 40, cy), (cx - gap, cy), color, 1, cv2.LINE_AA)
    cv2.line(frame, (cx + gap, cy), (cx + 40, cy), color, 1, cv2.LINE_AA)
    cv2.line(frame, (cx, cy - 40), (cx, cy - gap), color, 1, cv2.LINE_AA)
    cv2.line(frame, (cx, cy + gap), (cx, cy + 40), color, 1, cv2.LINE_AA)
    cv2.circle(frame, (cx, cy), 3, color, 1, cv2.LINE_AA)

    # Suhu palsu di tengah crosshair (oscillating)
    fake_temp = 32.0 + 6.0 * math.sin(time.time() * 1.5) + random.uniform(-0.5, 0.5)
    cv2.putText(frame, f"{fake_temp:.1f} C", (cx + 15, cy - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

    # Color scale bar di kanan
    bar_x = w - 35
    bar_y_start = 60
    bar_h = 160
    bar_w = 15
    for i in range(bar_h):
        ratio = i / bar_h
        val = int(255 * (1 - ratio))
        bar_color_row = cv2.applyColorMap(
            np.array([[val]], dtype=np.uint8), cv2.COLORMAP_JET
        )[0][0]
        bar_color_tuple = (int(bar_color_row[0] * alpha),
                           int(bar_color_row[1] * alpha),
                           int(bar_color_row[2] * alpha))
        cv2.line(frame, (bar_x, bar_y_start + i),
                 (bar_x + bar_w, bar_y_start + i), bar_color_tuple, 1)
    cv2.rectangle(frame, (bar_x - 1, bar_y_start - 1),
                  (bar_x + bar_w + 1, bar_y_start + bar_h + 1), dim, 1)
    cv2.putText(frame, "HOT", (bar_x - 5, bar_y_start - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, int(255 * alpha)), 1, cv2.LINE_AA)
    cv2.putText(frame, "COLD", (bar_x - 10, bar_y_start + bar_h + 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (int(255 * alpha), 0, 0), 1, cv2.LINE_AA)

    # Label "THERMAL" kiri atas
    cv2.putText(frame, "THERMAL IMAGING", (15, 50), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, color, 1, cv2.LINE_AA)

    # Timestamp kiri bawah
    time_str = time.strftime("%H:%M:%S")
    cv2.putText(frame, time_str, (15, h - 20), cv2.FONT_HERSHEY_SIMPLEX,
                0.4, dim, 1, cv2.LINE_AA)

    # "MAX" dan "MIN" temp labels
    max_temp = 38.0 + random.uniform(-0.3, 0.3)
    min_temp = 22.0 + random.uniform(-0.3, 0.3)
    cv2.putText(frame, f"MAX: {max_temp:.1f}C", (15, h - 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, int(100 * alpha), int(255 * alpha)), 1, cv2.LINE_AA)
    cv2.putText(frame, f"MIN: {min_temp:.1f}C", (15, h - 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (int(255 * alpha), int(100 * alpha), 0), 1, cv2.LINE_AA)


# ==========================================
#  🌊 Underwater / Aqua Effect (BARU - Fitur 12)
# ==========================================

class BubbleParticle:
    """Partikel gelembung air yang naik ke atas."""
    def __init__(self, frame_w, frame_h):
        self.frame_w = frame_w
        self.frame_h = frame_h
        self.reset()

    def reset(self):
        self.x = float(random.randint(0, self.frame_w))
        self.y = float(self.frame_h + random.randint(10, 60))
        self.vx = random.uniform(-0.5, 0.5)
        self.vy = random.uniform(-3.0, -1.0)
        self.size = random.randint(3, 14)
        self.wobble_phase = random.uniform(0, 2 * math.pi)
        self.wobble_speed = random.uniform(2.0, 5.0)
        self.wobble_amp = random.uniform(0.5, 2.0)
        self.life = 1.0
        self.alpha = random.uniform(0.3, 0.8)

    def update(self, time_val):
        self.y += self.vy
        self.x += self.vx + math.sin(time_val * self.wobble_speed + self.wobble_phase) * self.wobble_amp
        self.vy -= 0.01  # sedikit percepatan ke atas
        if self.y < -20:
            self.reset()

    def draw(self, frame, level):
        ix, iy = int(self.x), int(self.y)
        if iy < 0 or iy >= self.frame_h or ix < 0 or ix >= self.frame_w:
            return
        a = self.alpha * level
        s = self.size

        # Gelembung utama (lingkaran semi-transparan)
        overlay = frame.copy()
        bubble_color = (int(200 * a), int(180 * a), int(140 * a))  # Biru muda
        cv2.circle(overlay, (ix, iy), s, bubble_color, 1, cv2.LINE_AA)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        # Highlight (kilatan cahaya di sudut kiri atas gelembung)
        hx = ix - max(1, s // 3)
        hy = iy - max(1, s // 3)
        highlight_size = max(1, s // 4)
        highlight_color = (int(255 * a), int(255 * a), int(255 * a))
        cv2.circle(frame, (hx, hy), highlight_size, highlight_color, cv2.FILLED, cv2.LINE_AA)


def apply_underwater_effect(frame, level, time_val=0):
    """Efek bawah air: distorsi gelombang, tint biru-hijau, caustics."""
    if level <= 0:
        return frame
    h, w = frame.shape[:2]
    result = frame.copy()

    # Distorsi sinusoidal (gelombang air)
    wave_amp = int(3 * level)
    wave_freq = 0.02
    if wave_amp > 0:
        map_x = np.zeros((h, w), dtype=np.float32)
        map_y = np.zeros((h, w), dtype=np.float32)
        for row in range(h):
            for col in range(w):
                map_x[row, col] = col + wave_amp * math.sin(row * wave_freq + time_val * 2.0)
                map_y[row, col] = row + wave_amp * math.sin(col * wave_freq + time_val * 1.5)
        result = cv2.remap(result, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    # Tint biru-hijau (warna bawah air)
    tint_overlay = np.zeros_like(result, dtype=np.uint8)
    tint_overlay[:, :] = (140, 90, 20)  # BGR: biru tua + sedikit hijau
    tint_strength = 0.25 * level
    result = cv2.addWeighted(result, 1 - tint_strength, tint_overlay, tint_strength, 0)

    # God rays / caustics dari atas (garis-garis cahaya yang bergerak)
    caustic_overlay = np.zeros_like(result, dtype=np.uint8)
    num_rays = 5
    for i in range(num_rays):
        ray_x = int((w / (num_rays + 1)) * (i + 1) + 40 * math.sin(time_val * 0.8 + i * 1.2))
        ray_w = random.randint(15, 35)
        ray_alpha_val = int(40 * level * (0.5 + 0.5 * math.sin(time_val * 1.5 + i * 0.7)))
        pts = np.array([
            [ray_x - ray_w // 2, 0],
            [ray_x + ray_w // 2, 0],
            [ray_x + ray_w, h],
            [ray_x - ray_w, h],
        ], dtype=np.int32)
        cv2.fillPoly(caustic_overlay, [pts], (ray_alpha_val, ray_alpha_val, ray_alpha_val // 2))
    result = cv2.add(result, caustic_overlay)

    # Vignette biru gelap di pinggir
    cy_v, cx_v = h // 2, w // 2
    Y, X = np.ogrid[:h, :w]
    max_dist = math.sqrt(cx_v**2 + cy_v**2)
    dist = np.sqrt((X.astype(np.float32) - cx_v)**2 + (Y.astype(np.float32) - cy_v)**2)
    vignette = np.clip(1.0 - (dist / max_dist) * 0.5 * level, 0.4, 1.0).astype(np.float32)
    for c_ch in range(3):
        result[:, :, c_ch] = (result[:, :, c_ch].astype(np.float32) * vignette).astype(np.uint8)

    # Blend
    return cv2.addWeighted(result, level, frame, 1 - level, 0)


def draw_underwater_hud(frame, level, time_val=0):
    """HUD efek bawah air: kedalaman, tekanan, O2 level."""
    if level < 0.3:
        return
    h, w = frame.shape[:2]
    alpha = min(1.0, level)
    color = (int(200 * alpha), int(180 * alpha), int(80 * alpha))  # Cyan
    dim = (int(100 * alpha), int(90 * alpha), int(40 * alpha))

    # Label "UNDERWATER" di atas
    cv2.putText(frame, "UNDERWATER MODE", (15, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, color, 1, cv2.LINE_AA)

    # Depth meter kiri
    depth = 12.5 + 5.0 * math.sin(time_val * 0.3)
    cv2.putText(frame, f"DEPTH: {depth:.1f}m", (15, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)

    # Pressure
    pressure = 2.2 + 0.5 * math.sin(time_val * 0.2)
    cv2.putText(frame, f"PRESS: {pressure:.1f} atm", (15, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, dim, 1, cv2.LINE_AA)

    # O2 level bar di kiri bawah
    o2_level = 0.75 + 0.15 * math.sin(time_val * 0.1)
    bar_x = 15
    bar_y = h - 50
    bar_w = 100
    bar_h = 12
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), dim, 1)
    fill_w = int(bar_w * o2_level)
    bar_fill_color = (int(200 * alpha), int(200 * alpha), 0) if o2_level > 0.3 else (0, 0, int(255 * alpha))
    cv2.rectangle(frame, (bar_x + 1, bar_y + 1), (bar_x + fill_w, bar_y + bar_h - 1),
                  bar_fill_color, cv2.FILLED)
    cv2.putText(frame, f"O2: {int(o2_level * 100)}%", (bar_x, bar_y - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)

    # Temperature air
    water_temp = 18.0 + 3.0 * math.sin(time_val * 0.15)
    cv2.putText(frame, f"WATER: {water_temp:.1f}C", (15, h - 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, dim, 1, cv2.LINE_AA)

    # Compass bearing kanan bawah
    bearing = int((time_val * 10) % 360)
    cv2.putText(frame, f"BRG: {bearing:03d}", (w - 110, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)


# ==========================================
#  🎬 Slow Motion Replay (BARU - Fitur 13)
# ==========================================

def draw_slowmo_overlay(frame, replay_progress, replay_index, total_frames):
    """Gambar overlay sinematik slow-motion: letterbox + label."""
    h, w = frame.shape[:2]

    # Cinematic letterbox bars (atas dan bawah)
    bar_height = int(h * 0.08)
    cv2.rectangle(frame, (0, 0), (w, bar_height), (0, 0, 0), cv2.FILLED)
    cv2.rectangle(frame, (0, h - bar_height), (w, h), (0, 0, 0), cv2.FILLED)

    # Label "SLOW-MO REPLAY" di bar atas
    label_color = (180, 180, 255)
    cv2.putText(frame, "SLOW-MO REPLAY", (15, bar_height - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, label_color, 1, cv2.LINE_AA)

    # Playback icon (triangle ◀◀)
    tri_x = w - 100
    tri_y = bar_height - 15
    pts1 = np.array([[tri_x, tri_y], [tri_x + 12, tri_y + 8], [tri_x, tri_y + 16]], dtype=np.int32)
    pts2 = np.array([[tri_x + 14, tri_y], [tri_x + 26, tri_y + 8], [tri_x + 14, tri_y + 16]], dtype=np.int32)
    cv2.fillPoly(frame, [pts1], label_color)
    cv2.fillPoly(frame, [pts2], label_color)

    # Progress bar di bar bawah
    prog_y = h - bar_height + 8
    prog_x = 15
    prog_w = w - 30
    prog_h = 4
    cv2.rectangle(frame, (prog_x, prog_y), (prog_x + prog_w, prog_y + prog_h),
                  (80, 80, 80), cv2.FILLED)
    filled = int(prog_w * replay_progress)
    cv2.rectangle(frame, (prog_x, prog_y), (prog_x + filled, prog_y + prog_h),
                  label_color, cv2.FILLED)

    # Frame counter
    fc_text = f"{replay_index}/{total_frames}"
    cv2.putText(frame, fc_text, (w - 80, h - bar_height + 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1, cv2.LINE_AA)

    # Blink "▶" indicator
    if int(time.time() * 4) % 2 == 0:
        cv2.putText(frame, "x0.3", (15, h - bar_height + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 200, 255), 1, cv2.LINE_AA)

    # Motion trail effect - subtle blue tint
    overlay = np.zeros_like(frame, dtype=np.uint8)
    overlay[:, :] = (40, 15, 5)
    cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)


# ==========================================
#  🌈 Color Pop / Selective Color (BARU - Fitur 14)
# ==========================================

def apply_color_pop(frame, hand, level):
    """
    Efek Color Pop: seluruh frame grayscale kecuali warna dominan di area tangan.
    Efek dramatis ala Sin City.
    """
    if level <= 0:
        return frame
    h, w = frame.shape[:2]

    lmList = hand["lmList"]

    # Ambil warna area di sekitar telapak tangan (landmark 0 = wrist, 9 = middle base)
    palm_x = lmList[9][0]
    palm_y = lmList[9][1]

    # Sampling area kecil di sekitar telapak untuk mendapatkan warna dominan
    sample_size = 20
    sx1 = max(0, palm_x - sample_size)
    sx2 = min(w, palm_x + sample_size)
    sy1 = max(0, palm_y - sample_size)
    sy2 = min(h, palm_y + sample_size)

    if sx2 - sx1 < 5 or sy2 - sy1 < 5:
        return frame

    sample_region = frame[sy1:sy2, sx1:sx2]
    sample_hsv = cv2.cvtColor(sample_region, cv2.COLOR_BGR2HSV)

    # Hitung hue dominan
    mean_hue = np.mean(sample_hsv[:, :, 0])
    mean_sat = np.mean(sample_hsv[:, :, 1])

    # Buat mask untuk warna yang dekat dengan warna dominan
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hue_range = 20  # Toleransi hue
    sat_min = max(30, int(mean_sat * 0.4))

    lower = np.array([max(0, int(mean_hue) - hue_range), sat_min, 40])
    upper = np.array([min(179, int(mean_hue) + hue_range), 255, 255])

    # Handle wrap-around untuk hue
    if mean_hue - hue_range < 0:
        mask1 = cv2.inRange(hsv_frame, np.array([0, sat_min, 40]), upper)
        mask2 = cv2.inRange(hsv_frame, np.array([180 + int(mean_hue) - hue_range, sat_min, 40]),
                            np.array([179, 255, 255]))
        color_mask = cv2.bitwise_or(mask1, mask2)
    elif mean_hue + hue_range > 179:
        mask1 = cv2.inRange(hsv_frame, lower, np.array([179, 255, 255]))
        mask2 = cv2.inRange(hsv_frame, np.array([0, sat_min, 40]),
                            np.array([int(mean_hue) + hue_range - 180, 255, 255]))
        color_mask = cv2.bitwise_or(mask1, mask2)
    else:
        color_mask = cv2.inRange(hsv_frame, lower, upper)

    # Smooth mask edges
    color_mask = cv2.GaussianBlur(color_mask, (7, 7), 0)
    _, color_mask = cv2.threshold(color_mask, 127, 255, cv2.THRESH_BINARY)

    # Buat frame grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    # Gabungkan: grayscale di background, warna asli di foreground (mask)
    color_mask_3ch = cv2.cvtColor(color_mask, cv2.COLOR_GRAY2BGR) / 255.0
    result = (frame.astype(np.float32) * color_mask_3ch +
              gray_bgr.astype(np.float32) * (1 - color_mask_3ch)).astype(np.uint8)

    # Boost saturasi area berwarna agar lebih "pop"
    result_hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV)
    sat_boost = color_mask.astype(np.float32) / 255.0
    result_hsv[:, :, 1] = np.clip(
        result_hsv[:, :, 1].astype(np.float32) * (1.0 + 0.5 * sat_boost), 0, 255
    ).astype(np.uint8)
    result = cv2.cvtColor(result_hsv, cv2.COLOR_HSV2BGR)

    # Vignette dramatis
    cy_v, cx_v = h // 2, w // 2
    Y, X = np.ogrid[:h, :w]
    max_dist = math.sqrt(cx_v**2 + cy_v**2)
    dist_map = np.sqrt((X.astype(np.float32) - cx_v)**2 + (Y.astype(np.float32) - cy_v)**2)
    vignette = np.clip(1.0 - (dist_map / max_dist) * 0.4 * level, 0.5, 1.0).astype(np.float32)
    for c_ch in range(3):
        result[:, :, c_ch] = (result[:, :, c_ch].astype(np.float32) * vignette).astype(np.uint8)

    # Blend sesuai level
    return cv2.addWeighted(result, level, frame, 1 - level, 0)


def draw_color_pop_overlay(frame, level):
    """Overlay label Color Pop."""
    if level < 0.3:
        return
    h, w = frame.shape[:2]
    alpha = min(1.0, level)
    color = (int(100 * alpha), int(200 * alpha), int(255 * alpha))

    # Label "COLOR POP" di atas
    cv2.putText(frame, "COLOR POP", (w // 2 - 55, 25), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, color, 2, cv2.LINE_AA)

    # Garis bawah label
    cv2.line(frame, (w // 2 - 55, 32), (w // 2 + 60, 32), color, 1, cv2.LINE_AA)

    # Corner accents (kiri atas dan kanan bawah)
    accent_len = 20
    cv2.line(frame, (5, 5), (5 + accent_len, 5), color, 2, cv2.LINE_AA)
    cv2.line(frame, (5, 5), (5, 5 + accent_len), color, 2, cv2.LINE_AA)
    cv2.line(frame, (w - 5, h - 5), (w - 5 - accent_len, h - 5), color, 2, cv2.LINE_AA)
    cv2.line(frame, (w - 5, h - 5), (w - 5, h - 5 - accent_len), color, 2, cv2.LINE_AA)


# ==========================================
#  📸 Screenshot Helper (BARU - Fitur 15)
# ==========================================

def save_screenshot(frame, captures_dir):
    """Simpan screenshot ke folder captures/ dan return path file."""
    os.makedirs(captures_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"capture_{timestamp}.png"
    filepath = os.path.join(captures_dir, filename)
    cv2.imwrite(filepath, frame)
    return filepath, filename


def draw_screenshot_thumbnail(frame, thumbnail, display_time, filepath, position="bottom-right"):
    """Gambar thumbnail screenshot di pojok dengan animasi slide-in."""
    if thumbnail is None:
        return
    h, w = frame.shape[:2]
    th, tw = thumbnail.shape[:2]

    # Animasi slide-in (pertama 0.5 detik)
    slide_progress = min(1.0, display_time / 0.5)
    offset_x = int((1 - slide_progress) * (tw + 20))

    # Posisi thumbnail
    margin = 15
    tx = w - tw - margin - offset_x  # Kanan bawah yang bukan area label gesture
    ty = h - th - margin - 40  # Agak naik agar tidak overlap label gesture

    if tx < 0:
        return

    # Background gelap + border
    pad = 4
    overlay = frame.copy()
    cv2.rectangle(overlay, (tx - pad, ty - pad), (tx + tw + pad, ty + th + pad + 18),
                  (20, 20, 20), cv2.FILLED)
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

    # Border putih
    cv2.rectangle(frame, (tx - pad, ty - pad), (tx + tw + pad, ty + th + pad + 18),
                  (200, 200, 200), 1, cv2.LINE_AA)

    # Thumbnail
    frame[ty:ty+th, tx:tx+tw] = thumbnail

    # Filename label di bawah thumbnail
    cv2.putText(frame, "SAVED", (tx, ty + th + 14), cv2.FONT_HERSHEY_SIMPLEX,
                0.3, (100, 255, 100), 1, cv2.LINE_AA)


def draw_screenshot_flash(frame, flash_level):
    """Flash putih saat screenshot."""
    if flash_level > 0.01:
        white = np.ones_like(frame, dtype=np.uint8) * 255
        return cv2.addWeighted(white, flash_level * 0.8, frame, 1 - flash_level * 0.8, 0)
    return frame


# ==========================================
#  ⏺️ Video Recording Helper (BARU - Fitur 16)
# ==========================================

def draw_recording_indicator(frame, is_recording, rec_duration, blink_on=True):
    """Gambar indikator recording: titik merah berkedip + timer + border."""
    if not is_recording:
        return
    h, w = frame.shape[:2]

    # Titik merah berkedip
    if blink_on:
        cv2.circle(frame, (w - 25, 25), 8, (0, 0, 255), cv2.FILLED)
    cv2.putText(frame, "REC", (w - 65, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0, 0, 255), 2, cv2.LINE_AA)

    # Timer durasi
    minutes = int(rec_duration) // 60
    seconds = int(rec_duration) % 60
    timer_str = f"{minutes:02d}:{seconds:02d}"
    cv2.putText(frame, timer_str, (w - 65, 50), cv2.FONT_HERSHEY_SIMPLEX,
                0.4, (200, 200, 200), 1, cv2.LINE_AA)

    # Border merah tipis di sekeliling frame
    cv2.rectangle(frame, (2, 2), (w - 2, h - 2), (0, 0, 180), 2)


def draw_save_notification(frame, display_time, filepath):
    """Tampilkan notifikasi 'VIDEO SAVED' setelah stop recording."""
    if display_time <= 0:
        return
    h, w = frame.shape[:2]
    alpha = min(1.0, display_time)

    # Box notifikasi di tengah atas
    box_w = 280
    box_h = 50
    bx = (w - box_w) // 2
    by = 10

    overlay = frame.copy()
    cv2.rectangle(overlay, (bx, by), (bx + box_w, by + box_h), (20, 60, 20), cv2.FILLED)
    cv2.addWeighted(overlay, 0.85 * alpha, frame, 1 - 0.85 * alpha, 0, frame)
    cv2.rectangle(frame, (bx, by), (bx + box_w, by + box_h),
                  (0, int(200 * alpha), 0), 1, cv2.LINE_AA)

    color = (0, int(255 * alpha), 0)
    cv2.putText(frame, "VIDEO SAVED!", (bx + 75, by + 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

    filename = os.path.basename(filepath) if filepath else ""
    cv2.putText(frame, filename, (bx + 20, by + 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (int(150 * alpha), int(150 * alpha), int(150 * alpha)),
                1, cv2.LINE_AA)


# ==========================================
#  Common UI Helpers
# ==========================================

def lerp(a, b, t):
    """Linear interpolation antara a dan b dengan faktor t."""
    return a + (b - a) * t


def draw_fps_counter(frame, fps):
    """Gambar FPS counter di pojok kiri atas."""
    overlay = frame.copy()
    cv2.rectangle(overlay, (8, 5), (95, 28), (0, 0, 0), cv2.FILLED)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
    fps_color = (0, 255, 0) if fps >= 20 else (0, 200, 255) if fps >= 10 else (0, 0, 255)
    cv2.putText(frame, f"FPS: {int(fps)}", (12, 22), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, fps_color, 1, cv2.LINE_AA)


def draw_active_gesture_label(frame, gesture_name, gesture_color):
    """Gambar label gesture aktif di pojok kanan bawah."""
    if not gesture_name:
        return
    h, w = frame.shape[:2]
    label = f"[ {gesture_name} ]"
    text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)[0]
    tx = w - text_size[0] - 15
    ty = h - 15
    # Background semi-transparan
    overlay = frame.copy()
    cv2.rectangle(overlay, (tx - 8, ty - text_size[1] - 8),
                  (tx + text_size[0] + 8, ty + 8), (0, 0, 0), cv2.FILLED)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
    # Border
    cv2.rectangle(frame, (tx - 8, ty - text_size[1] - 8),
                  (tx + text_size[0] + 8, ty + 8), gesture_color, 1, cv2.LINE_AA)
    # Text
    cv2.putText(frame, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, gesture_color, 2, cv2.LINE_AA)


def draw_info_panel(frame):
    """Gambar panel info gesture semi-transparan di tengah layar."""
    h, w = frame.shape[:2]
    pw, ph = 360, 440
    px = (w - pw) // 2
    py = (h - ph) // 2

    # Background semi-transparan
    overlay = frame.copy()
    cv2.rectangle(overlay, (px, py), (px + pw, py + ph), (20, 20, 20), cv2.FILLED)
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
    cv2.rectangle(frame, (px, py), (px + pw, py + ph), (100, 200, 255), 2, cv2.LINE_AA)

    # Title
    cv2.putText(frame, "GESTURE GUIDE", (px + 95, py + 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (100, 200, 255), 2, cv2.LINE_AA)
    cv2.line(frame, (px + 10, py + 38), (px + pw - 10, py + 38), (80, 80, 80), 1)

    # Subtitle - 1 Tangan
    cv2.putText(frame, "[1 TANGAN]", (px + 15, py + 56),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (100, 150, 200), 1, cv2.LINE_AA)

    gestures_1h = [
        ("Peace (V)", "Blur Screen", (255, 200, 100)),
        ("Thumb+Index", "Blur Area", (200, 200, 100)),
        ("Rock Sign", "Glitch/VHS", (100, 100, 255)),
        ("Index Only", "Spotlight", (200, 200, 100)),
        ("Pinky Only", "Color Invert", (180, 120, 255)),
        ("Fist", "Night Vision", (0, 200, 0)),
        ("Open Palm", "Freeze Frame", (200, 200, 200)),
        ("Middle Only", "Thermal", (0, 180, 255)),
        ("Ring Only", "Underwater", (200, 150, 50)),
        ("Thumb Only", "Slow-Mo Replay", (180, 180, 255)),
        ("I+M+R Fingers", "Color Pop", (100, 200, 255)),
    ]

    y_start = py + 72
    line_h = 20
    for i, (gesture, effect, c) in enumerate(gestures_1h):
        y_pos = y_start + i * line_h
        cv2.putText(frame, gesture, (px + 15, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.37, c, 1, cv2.LINE_AA)
        cv2.putText(frame, effect, (px + 190, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.37, c, 1, cv2.LINE_AA)

    # Separator
    sep_y = y_start + len(gestures_1h) * line_h + 5
    cv2.line(frame, (px + 10, sep_y), (px + pw - 10, sep_y), (80, 80, 80), 1)

    # Subtitle - 2 Tangan
    cv2.putText(frame, "[2 TANGAN]", (px + 15, sep_y + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (100, 150, 200), 1, cv2.LINE_AA)

    gestures_2h = [
        ("Triangle", "Grayscale", (200, 200, 200)),
        ("Heart", "Love Effect", (100, 80, 255)),
        ("Rectangle", "Anime Filter", (255, 255, 100)),
    ]

    y_start_2h = sep_y + 34
    for i, (gesture, effect, c) in enumerate(gestures_2h):
        y_pos = y_start_2h + i * line_h
        cv2.putText(frame, gesture, (px + 15, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.37, c, 1, cv2.LINE_AA)
        cv2.putText(frame, effect, (px + 190, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.37, c, 1, cv2.LINE_AA)

    # Separator
    sep_y2 = y_start_2h + len(gestures_2h) * line_h + 5
    cv2.line(frame, (px + 10, sep_y2), (px + pw - 10, sep_y2), (80, 80, 80), 1)

    # Subtitle - Keyboard
    cv2.putText(frame, "[KEYBOARD]", (px + 15, sep_y2 + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (100, 150, 200), 1, cv2.LINE_AA)

    keyboard_shortcuts = [
        ("S", "Screenshot", (100, 255, 100)),
        ("R", "Record Video", (100, 100, 255)),
        ("H", "Toggle Help", (150, 150, 150)),
        ("Q", "Quit", (150, 150, 150)),
    ]

    y_start_kb = sep_y2 + 34
    for i, (key, effect, c) in enumerate(keyboard_shortcuts):
        y_pos = y_start_kb + i * line_h
        cv2.putText(frame, f"[{key}]", (px + 15, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.37, c, 1, cv2.LINE_AA)
        cv2.putText(frame, effect, (px + 190, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.37, c, 1, cv2.LINE_AA)

    cv2.putText(frame, "Press H to hide", (px + 110, py + ph - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1, cv2.LINE_AA)


def draw_box_blur_border(frame, x_min, y_min, x_max, y_max):
    """Gambar neon border di sekeliling area box blur."""
    # Outer glow
    cv2.rectangle(frame, (x_min - 2, y_min - 2), (x_max + 2, y_max + 2),
                  (150, 100, 0), 3, cv2.LINE_AA)
    # Inner line
    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max),
                  (255, 200, 50), 1, cv2.LINE_AA)

    # Corner markers (L-shaped)
    corner_len = min(20, (x_max - x_min) // 5, (y_max - y_min) // 5)
    cc = (255, 255, 100)
    for idx, (cx_c, cy_c) in enumerate([(x_min, y_min), (x_max, y_min),
                                         (x_min, y_max), (x_max, y_max)]):
        dx = corner_len if (idx % 2 == 0) else -corner_len
        dy = corner_len if (idx < 2) else -corner_len
        cv2.line(frame, (cx_c, cy_c), (cx_c + dx, cy_c), cc, 2, cv2.LINE_AA)
        cv2.line(frame, (cx_c, cy_c), (cx_c, cy_c + dy), cc, 2, cv2.LINE_AA)

    # Label "BLUR ZONE"
    label_y = max(y_min - 8, 15)
    cv2.putText(frame, "BLUR ZONE", (x_min, label_y), cv2.FONT_HERSHEY_SIMPLEX,
                0.4, cc, 1, cv2.LINE_AA)


def draw_startup_splash(frame, alpha):
    """Gambar startup splash overlay yang fade out."""
    if alpha <= 0.01:
        return frame
    h, w = frame.shape[:2]

    # Dark overlay
    overlay = np.zeros_like(frame, dtype=np.uint8)
    frame = cv2.addWeighted(overlay, alpha * 0.7, frame, 1 - alpha * 0.7, 0)

    # Title text
    title = "GESTURE CAMERA"
    sub = "ULTIMATE EDITION"
    title_alpha = min(1.0, alpha * 2)
    t_color = (int(100 * title_alpha), int(200 * title_alpha), int(255 * title_alpha))
    s_color = (int(150 * title_alpha), int(150 * title_alpha), int(150 * title_alpha))

    t_size = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)[0]
    s_size = cv2.getTextSize(sub, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
    tx = (w - t_size[0]) // 2
    ty = h // 2 - 10
    sx = (w - s_size[0]) // 2
    sy = h // 2 + 30

    cv2.putText(frame, title, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX,
                1.2, t_color, 3, cv2.LINE_AA)
    cv2.putText(frame, sub, (sx, sy), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, s_color, 1, cv2.LINE_AA)

    ver = "v3.0 | 16 Gestures | Press H for help"
    v_size = cv2.getTextSize(ver, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)[0]
    vx = (w - v_size[0]) // 2
    v_color = (int(80*title_alpha), int(80*title_alpha), int(80*title_alpha))
    cv2.putText(frame, ver, (vx, h // 2 + 60), cv2.FONT_HERSHEY_SIMPLEX,
                0.35, v_color, 1, cv2.LINE_AA)

    return frame


def main():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("ERROR: Kamera tidak bisa dibuka!")
        return

    # Detektor tangan: butuh 2 tangan untuk gesture segitiga & love
    detector = HandDetector(detectionCon=0.7, maxHands=2)

    # Folder captures
    script_dir = os.path.dirname(os.path.abspath(__file__))
    captures_dir = os.path.join(script_dir, "captures")
    os.makedirs(captures_dir, exist_ok=True)

    print("=" * 55)
    print("  GESTURE CAMERA - ULTIMATE EDITION (16 Gestures)")
    print("=" * 55)
    print("  Jari & landmark tangan terdeteksi otomatis")
    print("  -----------------------------------------------")
    print("  [1 TANGAN]")
    print("  Peace Sign (V)        : Blur Full Screen")
    print("  Jempol & Telunjuk     : Blur Area Kotak")
    print("  Rock Sign (metal)     : Glitch / VHS Effect")
    print("  Telunjuk Saja         : Spotlight")
    print("  Kelingking Saja       : Color Invert")
    print("  Kepalan Tangan        : Night Vision")
    print("  Telapak Terbuka       : Freeze Frame")
    print("  Jari Tengah Saja      : Thermal Vision")
    print("  Jari Manis Saja       : Underwater Effect")
    print("  Jempol Saja           : Slow Motion Replay")
    print("  Telunjuk+Tengah+Manis : Color Pop")
    print("  -----------------------------------------------")
    print("  [2 TANGAN]")
    print("  Segitiga              : Grayscale")
    print("  Love / Heart          : Efek Hati")
    print("  Kotak (tangan buka)   : Anime Filter")
    print("  -----------------------------------------------")
    print("  [KEYBOARD]")
    print("  S                     : Screenshot")
    print("  R                     : Record / Stop Video")
    print("  H                     : Toggle Help")
    print("  Q                     : Quit")
    print("=" * 55)

    # Smooth transition untuk blur & grayscale
    blur_level = 0.0
    blur_speed = 0.15
    max_blur_kernel = 99
    gray_level = 0.0
    gray_speed = 0.12

    # Heart effect state
    heart_level = 0.0     # 0.0 = off, 1.0 = fully on
    heart_speed = 0.1
    heart_particles = []  # List partikel hati
    heart_pulse_time = 0  # Untuk animasi denyut
    heart_spawn_timer = 0
    heart_center = (0, 0)
    heart_size = 50

    # Delay/konfirmasi deteksi love gesture
    love_hold_start = 0       # Waktu mulai menahan gesture
    love_hold_duration = 1.2  # Harus tahan selama 1.2 detik
    love_confirmed = False    # Apakah sudah dikonfirmasi
    love_detecting = False    # Sedang dalam proses deteksi

    # Anime rectangle state
    anime_level = 0.0   # Smooth transition
    anime_speed = 0.15
    anime_rect = None    # (x_min, y_min, x_max, y_max)

    # Night Vision state
    nv_level = 0.0
    nv_speed = 0.12

    # Freeze Frame state
    freeze_frame = None
    freeze_level = 0.0
    freeze_speed = 0.15
    freeze_flash = 0.0
    freeze_cooldown = 0
    freeze_capture_time = 0
    freeze_display_duration = 2.5

    # Glitch/VHS state
    glitch_level = 0.0
    glitch_speed = 0.15

    # Spotlight state
    spot_level = 0.0
    spot_speed = 0.12
    spot_pos = (0, 0)
    spot_radius = 120

    # Color Invert state
    invert_level = 0.0
    invert_speed = 0.12

    # === BARU: Thermal Vision state ===
    thermal_level = 0.0
    thermal_speed = 0.12

    # === BARU: Underwater state ===
    underwater_level = 0.0
    underwater_speed = 0.10
    bubble_particles = []
    bubbles_initialized = False

    # === BARU: Slow Motion Replay state ===
    frame_buffer = deque(maxlen=60)  # Circular buffer 60 frame
    slowmo_active = False
    slowmo_replay_index = 0
    slowmo_replay_frames = []
    slowmo_sub_frame = 0  # Sub-frame counter untuk kecepatan 1/3
    slowmo_cooldown = 0

    # === BARU: Color Pop state ===
    color_pop_level = 0.0
    color_pop_speed = 0.12
    color_pop_hand = None  # Hand data saat color pop aktif

    # === BARU: Screenshot state ===
    screenshot_flash = 0.0
    screenshot_thumbnail = None
    screenshot_display_time = 0.0
    screenshot_filepath = ""
    screenshot_count = 0

    # === BARU: Video Recording state ===
    is_recording = False
    video_writer = None
    rec_start_time = 0
    rec_save_notification_time = 0.0
    rec_saved_filepath = ""

    # UI state
    show_help = False
    frame_count = 0
    splash_alpha = 1.0
    start_time = time.time()

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        current_time = time.time()
        frame_count += 1

        # Initialize bubble particles once we know frame dimensions
        if not bubbles_initialized:
            bubble_particles = [BubbleParticle(w, h) for _ in range(25)]
            bubbles_initialized = True

        # Buffer frame untuk slow-mo (sebelum efek diterapkan)
        if not slowmo_active:
            frame_buffer.append(frame.copy())

        # Deteksi tangan (draw=False, kita gambar sendiri biar lebih bagus)
        hands, frame = detector.findHands(frame, draw=False)

        peace_detected = False
        box_blur_detected = False
        box_coords = None
        triangle_detected = False
        triangle_pts = None
        love_raw_detected = False   # Gesture terdeteksi frame ini (belum konfirmasi)
        love_center = None
        love_size = 0
        rect_detected = False
        rect_coords = None
        fist_detected = False
        palm_detected = False
        rock_detected = False
        spotlight_detected = False
        spotlight_pos = None
        pinky_detected = False
        # === BARU ===
        middle_detected = False     # Jari tengah saja → Thermal
        ring_detected = False       # Jari manis saja → Underwater
        thumb_only_detected = False # Jempol saja → Slow-mo
        three_fingers_detected = False  # Telunjuk+Tengah+Manis → Color Pop
        three_fingers_hand = None

        if hands:
            # Gambar landmark untuk setiap tangan yang terdeteksi
            hand_colors = [(0, 255, 150), (255, 150, 0)]  # Hijau, Biru-oranye
            for i, hand in enumerate(hands):
                color = hand_colors[i % len(hand_colors)]
                draw_hand_landmarks(frame, hand, color)

            # --- Deteksi gesture 2 tangan ---
            if len(hands) == 2:
                # Cek rectangle/anime dulu (kedua tangan buka semua jari)
                rect_detected, rect_coords = is_rectangle_gesture(hands, detector)

                # Kalau bukan rectangle, cek love
                if not rect_detected:
                    love_raw_detected, love_center, love_size = is_heart_gesture(hands, detector)

            # --- Gesture 1 tangan (ambil tangan pertama) ---
            if not love_raw_detected and not rect_detected:
                hand1 = hands[0]
                fingers = detector.fingersUp(hand1)

                # LOGIKA 1: Peace Sign (Telunjuk & Tengah UP) → Blur full screen
                if fingers == [0, 1, 1, 0, 0] or fingers == [1, 1, 1, 0, 0]:
                    peace_detected = True

                # LOGIKA 2: Jempol & Telunjuk saja UP → Blur area kotak
                elif fingers == [1, 1, 0, 0, 0]:
                    box_blur_detected = True
                    lmList = hand1["lmList"]

                    # Ujung jempol (id:4) dan ujung telunjuk (id:8)
                    x1, y1 = lmList[4][0], lmList[4][1]
                    x2, y2 = lmList[8][0], lmList[8][1]

                    # Bounding box
                    x_min, x_max = max(0, min(x1, x2)), min(w, max(x1, x2))
                    y_min, y_max = max(0, min(y1, y2)), min(h, max(y1, y2))

                    if x_max - x_min > 20 and y_max - y_min > 20:
                        box_coords = (x_min, y_min, x_max, y_max)

                # LOGIKA 3: Rock Sign (Telunjuk + Kelingking UP) → Glitch/VHS
                elif fingers == [0, 1, 0, 0, 1] or fingers == [1, 1, 0, 0, 1]:
                    rock_detected = True

                # LOGIKA 14 (BARU): Telunjuk + Tengah + Manis → Color Pop
                elif fingers == [0, 1, 1, 1, 0]:
                    three_fingers_detected = True
                    three_fingers_hand = hand1

                # LOGIKA 4: Telunjuk Saja → Spotlight
                elif fingers == [0, 1, 0, 0, 0]:
                    spotlight_detected = True
                    lmList = hand1["lmList"]
                    spotlight_pos = (lmList[8][0], lmList[8][1])

                # LOGIKA 11 (BARU): Jari Tengah Saja → Thermal Vision
                elif fingers == [0, 0, 1, 0, 0]:
                    middle_detected = True

                # LOGIKA 12 (BARU): Jari Manis Saja → Underwater
                elif fingers == [0, 0, 0, 1, 0]:
                    ring_detected = True

                # LOGIKA 5: Kelingking Saja → Color Invert
                elif fingers == [0, 0, 0, 0, 1]:
                    pinky_detected = True

                # LOGIKA 13 (BARU): Jempol Saja → Slow Motion Replay
                elif fingers == [1, 0, 0, 0, 0]:
                    thumb_only_detected = True

                # LOGIKA 6: Kepalan Tangan → Night Vision
                elif fingers == [0, 0, 0, 0, 0]:
                    fist_detected = True

                # LOGIKA 7: Telapak Terbuka (1 tangan saja) → Freeze Frame
                elif sum(fingers) >= 5 and len(hands) == 1:
                    palm_detected = True

                # Deteksi segitiga: butuh 2 tangan
                if len(hands) == 2 and not love_raw_detected and not rect_detected:
                    lm1 = hands[0]["lmList"]
                    lm2 = hands[1]["lmList"]

                    # Ujung jempol (id:4) dan ujung telunjuk (id:8) dari masing-masing tangan
                    thumb1 = (lm1[4][0], lm1[4][1])
                    index1 = (lm1[8][0], lm1[8][1])
                    thumb2 = (lm2[4][0], lm2[4][1])
                    index2 = (lm2[8][0], lm2[8][1])

                    # Cek apakah ujung jempol kedua tangan saling dekat (membentuk puncak)
                    # dan ujung telunjuk kedua tangan saling dekat (membentuk puncak)
                    # Skenario: tangan kiri & kanan membentuk segitiga
                    #   - Jempol1 dekat Jempol2 → satu titik sudut
                    #   - Telunjuk1 & Telunjuk2 → dua titik sudut lainnya

                    thumb_dist = distance(thumb1, thumb2)
                    thresh = 60  # Jarak maksimal ujung jempol berdekatan

                    if thumb_dist < thresh:
                        # Puncak segitiga = titik tengah kedua jempol
                        top = ((thumb1[0] + thumb2[0]) // 2, (thumb1[1] + thumb2[1]) // 2)
                        # Dua titik bawah = ujung telunjuk masing-masing tangan
                        bottom_left = index1
                        bottom_right = index2

                        if is_triangle(top, bottom_left, bottom_right, min_side=50):
                            triangle_detected = True
                            triangle_pts = (top, bottom_left, bottom_right)

                    # Alternatif: Telunjuk1 dekat Telunjuk2 → puncak, Jempol = basis
                    index_dist = distance(index1, index2)
                    if not triangle_detected and index_dist < thresh:
                        top = ((index1[0] + index2[0]) // 2, (index1[1] + index2[1]) // 2)
                        bottom_left = thumb1
                        bottom_right = thumb2

                        if is_triangle(top, bottom_left, bottom_right, min_side=50):
                            triangle_detected = True
                            triangle_pts = (top, bottom_left, bottom_right)

        # ====================================================
        # EKSEKUSI EFEK VISUAL
        # ====================================================

        # 1. Animasi Blur Full Screen (Peace Sign)
        if peace_detected:
            blur_level = min(1.0, blur_level + blur_speed)
        else:
            blur_level = max(0.0, blur_level - blur_speed)

        if blur_level > 0.01:
            kernel_size = int(blur_level * max_blur_kernel)
            kernel_size = max(1, kernel_size)
            if kernel_size % 2 == 0:
                kernel_size += 1
            blurred_frame = cv2.GaussianBlur(frame, (kernel_size, kernel_size), 0)
            frame = cv2.addWeighted(blurred_frame, blur_level, frame, 1 - blur_level, 0)

        # 2. Blur Area Kotak (Jempol & Telunjuk)
        if box_blur_detected and box_coords:
            x_min, y_min, x_max, y_max = box_coords
            roi = frame[y_min:y_max, x_min:x_max]
            roi_blurred = cv2.GaussianBlur(roi, (71, 71), 0)
            frame[y_min:y_max, x_min:x_max] = roi_blurred

        # 3. Smooth transition grayscale (Segitiga)
        if triangle_detected:
            gray_level = min(1.0, gray_level + gray_speed)
        else:
            gray_level = max(0.0, gray_level - gray_speed)

        if gray_level > 0.01:
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_bgr = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2BGR)
            frame = cv2.addWeighted(gray_bgr, gray_level, frame, 1 - gray_level, 0)

        # 4. ❤️ Efek Love/Heart — dengan delay konfirmasi
        # Fase 1: Deteksi pola → tahan gesture → loading ring
        if love_raw_detected:
            if not love_detecting:
                # Baru mulai mendeteksi
                love_detecting = True
                love_hold_start = current_time
            # Hitung progress hold
            hold_elapsed = current_time - love_hold_start
            hold_progress = min(1.0, hold_elapsed / love_hold_duration)

            # Update posisi center
            heart_center = love_center
            heart_size = love_size

            # Gambar loading ring (lingkaran yang terisi sesuai progress)
            if not love_confirmed:
                ring_cx, ring_cy = love_center
                ring_radius = 35
                # Background ring (abu-abu transparan)
                cv2.circle(frame, (ring_cx, ring_cy), ring_radius, (80, 80, 80), 2, cv2.LINE_AA)
                # Progress arc (merah-pink, terisi sesuai progress)
                end_angle = int(360 * hold_progress)
                if end_angle > 0:
                    cv2.ellipse(frame, (ring_cx, ring_cy), (ring_radius, ring_radius),
                               -90, 0, end_angle, (130, 100, 255), 3, cv2.LINE_AA)
                # Mini heart icon di tengah ring
                mini_pts = generate_heart_points(ring_cx, ring_cy, 10, num_points=30)
                if len(mini_pts) >= 3:
                    mini_arr = np.array(mini_pts, dtype=np.int32)
                    # Warna makin terang sesuai progress
                    r_val = int(150 + 105 * hold_progress)
                    cv2.fillPoly(frame, [mini_arr], (100, 80, r_val))
                    cv2.polylines(frame, [mini_arr], True, (180, 140, 255), 1, cv2.LINE_AA)

            # Cek apakah sudah cukup lama di-hold
            if hold_progress >= 1.0 and not love_confirmed:
                love_confirmed = True
                # Burst partikel saat pertama kali dikonfirmasi!
                for _ in range(20):
                    p = HeartParticle(heart_center[0], heart_center[1], w, h)
                    p.vx = random.uniform(-3.0, 3.0)
                    p.vy = random.uniform(-5.0, -1.0)
                    p.size = random.randint(6, 15)
                    heart_particles.append(p)
                for _ in range(10):
                    angle = random.uniform(0, 2 * math.pi)
                    sr = random.randint(10, 40)
                    sx = int(heart_center[0] + sr * math.cos(angle))
                    sy = int(heart_center[1] + sr * math.sin(angle))
                    p = HeartParticle(sx, sy, w, h)
                    p.is_sparkle = True
                    p.size = random.randint(4, 10)
                    heart_particles.append(p)
        else:
            # Gesture dilepas → reset deteksi
            love_detecting = False
            love_hold_start = 0
            love_confirmed = False

        # Fase 2: Efek aktif setelah dikonfirmasi
        if love_confirmed:
            heart_level = min(1.0, heart_level + heart_speed)
        else:
            heart_level = max(0.0, heart_level - heart_speed * 0.5)

        if heart_level > 0.01:
            # Efek vignette merah/pink di pinggir layar
            draw_heart_vignette(frame, alpha=heart_level * 0.6)

            # Spawn partikel hati kecil-kecil terus menerus
            heart_spawn_timer += 1
            if heart_spawn_timer >= 3:  # Setiap 3 frame
                heart_spawn_timer = 0
                num_new = random.randint(2, 5)
                for _ in range(num_new):
                    # Spawn dari area sekitar center, tersebar
                    spread = max(30, heart_size)
                    sx = heart_center[0] + random.randint(-spread, spread)
                    sy = heart_center[1] + random.randint(-spread // 2, spread // 2)
                    p = HeartParticle(sx, sy, w, h)
                    p.size = random.randint(4, 10)  # Kecil-kecil
                    heart_particles.append(p)

            # Tambah sparkles tersebar di layar
            if random.random() < 0.5:
                sx = random.randint(0, w)
                sy = random.randint(0, h)
                p = HeartParticle(sx, sy, w, h)
                p.is_sparkle = True
                p.size = random.randint(3, 7)
                p.decay = random.uniform(0.02, 0.04)
                p.vy = random.uniform(-1.5, -0.3)
                heart_particles.append(p)

        # Update dan gambar semua partikel
        alive_particles = []
        for p in heart_particles:
            p.update()
            if p.is_alive():
                p.draw(frame)
                alive_particles.append(p)
        heart_particles = alive_particles

        # Batasi jumlah partikel agar performa tetap oke
        if len(heart_particles) > 200:
            heart_particles = heart_particles[-200:]

        # 5. 🎌 Efek Anime (Rectangle - kedua tangan terbuka)
        if rect_detected and rect_coords:
            anime_level = min(1.0, anime_level + anime_speed)
            anime_rect = rect_coords
        else:
            anime_level = max(0.0, anime_level - anime_speed * 0.5)

        if anime_level > 0.01 and anime_rect:
            ax_min, ay_min, ax_max, ay_max = anime_rect
            # Clamp ke batas frame
            ax_min = max(0, ax_min)
            ay_min = max(0, ay_min)
            ax_max = min(w, ax_max)
            ay_max = min(h, ay_max)

            if ax_max - ax_min > 10 and ay_max - ay_min > 10:
                # Ambil ROI dan terapkan anime filter
                roi = frame[ay_min:ay_max, ax_min:ax_max].copy()
                anime_roi = apply_anime_filter(roi)

                # Blend anime dengan original sesuai anime_level
                blended = cv2.addWeighted(anime_roi, anime_level, roi, 1 - anime_level, 0)
                frame[ay_min:ay_max, ax_min:ax_max] = blended

                # Gambar frame kotak bergaya anime
                draw_anime_frame(frame, ax_min, ay_min, ax_max, ay_max, anime_level)

        # 6. Night Vision (Kepalan Tangan)
        if fist_detected:
            nv_level = min(1.0, nv_level + nv_speed)
        else:
            nv_level = max(0.0, nv_level - nv_speed)

        if nv_level > 0.01:
            frame = apply_night_vision(frame, nv_level)
            draw_night_vision_hud(frame, nv_level)

        # 7. Freeze Frame (Telapak Terbuka)
        if palm_detected and freeze_frame is None and (current_time - freeze_cooldown) > 1.0:
            freeze_frame = frame.copy()
            freeze_capture_time = current_time
            freeze_flash = 1.0
            freeze_level = 0.0

        if freeze_frame is not None:
            elapsed_freeze = current_time - freeze_capture_time
            if elapsed_freeze < freeze_display_duration:
                freeze_level = min(1.0, freeze_level + freeze_speed)
                # Tampilkan frozen frame dengan border Polaroid
                fh, fw = frame.shape[:2]
                border = 15
                display = cv2.addWeighted(freeze_frame, freeze_level, frame, 1 - freeze_level, 0)
                # Border Polaroid
                cv2.rectangle(display, (border, border), (fw - border, fh - border),
                              (240, 240, 240), 2, cv2.LINE_AA)
                cv2.rectangle(display, (border + 3, border + 3),
                              (fw - border - 3, fh - border - 3),
                              (200, 200, 200), 1, cv2.LINE_AA)
                # Label "CAPTURED"
                label_color = (200, 200, 200)
                cv2.putText(display, "CAPTURED", (fw // 2 - 50, fh - border - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, label_color, 1, cv2.LINE_AA)
                # Timestamp
                cap_time_str = time.strftime("%H:%M:%S",
                                             time.localtime(freeze_capture_time))
                cv2.putText(display, cap_time_str, (border + 10, border + 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, label_color, 1, cv2.LINE_AA)
                frame = display
            else:
                # Fade out
                freeze_level = max(0.0, freeze_level - freeze_speed)
                if freeze_level > 0.01:
                    frame = cv2.addWeighted(freeze_frame, freeze_level,
                                            frame, 1 - freeze_level, 0)
                else:
                    freeze_frame = None
                    freeze_cooldown = current_time

        # Flash effect saat capture
        if freeze_flash > 0.01:
            white = np.ones_like(frame, dtype=np.uint8) * 255
            frame = cv2.addWeighted(white, freeze_flash * 0.7,
                                    frame, 1 - freeze_flash * 0.7, 0)
            freeze_flash *= 0.75  # Decay cepat

        # 8. Glitch / VHS (Rock Sign)
        if rock_detected:
            glitch_level = min(1.0, glitch_level + glitch_speed)
        else:
            glitch_level = max(0.0, glitch_level - glitch_speed)

        if glitch_level > 0.01:
            frame = apply_glitch_effect(frame, glitch_level)
            draw_glitch_hud(frame, glitch_level)

        # 9. Spotlight (Telunjuk Saja)
        if spotlight_detected and spotlight_pos:
            spot_level = min(1.0, spot_level + spot_speed)
            spot_pos = spotlight_pos
        else:
            spot_level = max(0.0, spot_level - spot_speed)

        if spot_level > 0.01:
            frame = apply_spotlight(frame, spot_pos[0], spot_pos[1],
                                    spot_radius, spot_level)
            draw_spotlight_ring(frame, spot_pos[0], spot_pos[1],
                                spot_radius, spot_level)

        # 10. Color Invert (Kelingking Saja)
        if pinky_detected:
            invert_level = min(1.0, invert_level + invert_speed)
        else:
            invert_level = max(0.0, invert_level - invert_speed)

        if invert_level > 0.01:
            frame = apply_color_invert(frame, invert_level)
            draw_invert_border(frame, invert_level)

        # ====================================================
        # FITUR BARU (11-14)
        # ====================================================

        # 11. 🔥 Thermal Vision (Jari Tengah Saja)
        if middle_detected:
            thermal_level = min(1.0, thermal_level + thermal_speed)
        else:
            thermal_level = max(0.0, thermal_level - thermal_speed)

        if thermal_level > 0.01:
            frame = apply_thermal_vision(frame, thermal_level)
            draw_thermal_hud(frame, thermal_level)

        # 12. 🌊 Underwater (Jari Manis Saja)
        if ring_detected:
            underwater_level = min(1.0, underwater_level + underwater_speed)
        else:
            underwater_level = max(0.0, underwater_level - underwater_speed)

        if underwater_level > 0.01:
            frame = apply_underwater_effect(frame, underwater_level, current_time)
            draw_underwater_hud(frame, underwater_level, current_time)
            # Update dan gambar gelembung
            for bubble in bubble_particles:
                bubble.update(current_time)
                bubble.draw(frame, underwater_level)

        # 13. 🎬 Slow Motion Replay (Jempol Saja)
        if thumb_only_detected and not slowmo_active and len(frame_buffer) >= 10 and \
                (current_time - slowmo_cooldown) > 2.0:
            # Mulai replay
            slowmo_active = True
            slowmo_replay_frames = list(frame_buffer)
            slowmo_replay_index = 0
            slowmo_sub_frame = 0

        if slowmo_active:
            if slowmo_replay_index < len(slowmo_replay_frames):
                # Tampilkan frame replay
                replay_frame = slowmo_replay_frames[slowmo_replay_index].copy()
                total = len(slowmo_replay_frames)
                progress = slowmo_replay_index / max(1, total - 1)

                # Motion trail: blend dengan frame sebelumnya
                if slowmo_replay_index > 0:
                    prev_frame = slowmo_replay_frames[slowmo_replay_index - 1]
                    replay_frame = cv2.addWeighted(replay_frame, 0.7, prev_frame, 0.3, 0)

                draw_slowmo_overlay(replay_frame, progress, slowmo_replay_index, total)
                frame = replay_frame

                # Kecepatan 1/3 (setiap frame ditampilkan 3x)
                slowmo_sub_frame += 1
                if slowmo_sub_frame >= 3:
                    slowmo_sub_frame = 0
                    slowmo_replay_index += 1
            else:
                # Replay selesai
                slowmo_active = False
                slowmo_cooldown = current_time

        # 14. 🌈 Color Pop (Telunjuk + Tengah + Manis)
        if three_fingers_detected and three_fingers_hand is not None:
            color_pop_level = min(1.0, color_pop_level + color_pop_speed)
            color_pop_hand = three_fingers_hand
        else:
            color_pop_level = max(0.0, color_pop_level - color_pop_speed)

        if color_pop_level > 0.01 and color_pop_hand is not None:
            frame = apply_color_pop(frame, color_pop_hand, color_pop_level)
            draw_color_pop_overlay(frame, color_pop_level)

        # ====================================================
        # OVERLAY VISUAL (segitiga, border, dll)
        # ====================================================

        # Gambar segitiga jika terdeteksi
        if triangle_detected and triangle_pts:
            pts = triangle_pts
            # Garis segitiga dengan efek glow
            for i in range(3):
                p1 = pts[i]
                p2 = pts[(i + 1) % 3]
                # Outer glow
                cv2.line(frame, p1, p2, (0, 100, 255), 6, cv2.LINE_AA)
                # Inner line
                cv2.line(frame, p1, p2, (0, 200, 255), 2, cv2.LINE_AA)

            # Titik sudut segitiga
            for pt in pts:
                cv2.circle(frame, pt, 10, (0, 255, 255), cv2.FILLED)
                cv2.circle(frame, pt, 12, (255, 255, 255), 2, cv2.LINE_AA)

        # Box blur border
        if box_blur_detected and box_coords:
            draw_box_blur_border(frame, *box_coords)

        # ====================================================
        # SCREENSHOT & RECORDING
        # ====================================================

        # 15. Screenshot flash
        if screenshot_flash > 0.01:
            frame = draw_screenshot_flash(frame, screenshot_flash)
            screenshot_flash *= 0.7

        # Screenshot thumbnail display
        if screenshot_thumbnail is not None:
            elapsed_ss = current_time - screenshot_display_time
            if elapsed_ss < 3.0:
                draw_screenshot_thumbnail(frame, screenshot_thumbnail,
                                          elapsed_ss, screenshot_filepath)
            else:
                screenshot_thumbnail = None

        # 16. Recording indicator
        if is_recording:
            rec_duration = current_time - rec_start_time
            blink = int(current_time * 2) % 2 == 0
            draw_recording_indicator(frame, True, rec_duration, blink)
            # Write frame to video
            if video_writer is not None:
                video_writer.write(frame)

        # Save notification setelah stop recording
        if rec_save_notification_time > 0:
            remaining = 3.0 - (current_time - rec_save_notification_time)
            if remaining > 0:
                draw_save_notification(frame, remaining, rec_saved_filepath)
            else:
                rec_save_notification_time = 0

        # Screenshot counter (kecil di pojok)
        if screenshot_count > 0:
            cv2.putText(frame, f"Photos: {screenshot_count}", (w - 100, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (150, 150, 150), 1, cv2.LINE_AA)

        # ====================================================
        # UI OVERLAY
        # ====================================================

        # Startup splash (fade out selama 3 detik pertama)
        elapsed_start = current_time - start_time
        if elapsed_start < 3.0:
            splash_alpha = max(0.0, 1.0 - elapsed_start / 3.0)
            frame = draw_startup_splash(frame, splash_alpha)

        # Help panel
        if show_help:
            draw_info_panel(frame)

        # Active gesture label
        gesture_name = ""
        gesture_color = (200, 200, 200)
        if peace_detected:
            gesture_name = "BLUR"
            gesture_color = (255, 200, 100)
        elif box_blur_detected:
            gesture_name = "BOX BLUR"
            gesture_color = (200, 200, 100)
        elif rock_detected:
            gesture_name = "GLITCH"
            gesture_color = (100, 100, 255)
        elif spotlight_detected:
            gesture_name = "SPOTLIGHT"
            gesture_color = (200, 200, 100)
        elif pinky_detected:
            gesture_name = "NEGATIVE"
            gesture_color = (180, 120, 255)
        elif fist_detected:
            gesture_name = "NIGHT VISION"
            gesture_color = (0, 200, 0)
        elif palm_detected:
            gesture_name = "FREEZE"
            gesture_color = (200, 200, 200)
        elif triangle_detected:
            gesture_name = "GRAYSCALE"
            gesture_color = (200, 200, 200)
        elif love_confirmed:
            gesture_name = "LOVE"
            gesture_color = (100, 80, 255)
        elif rect_detected:
            gesture_name = "ANIME"
            gesture_color = (255, 255, 100)
        elif middle_detected:
            gesture_name = "THERMAL"
            gesture_color = (0, 180, 255)
        elif ring_detected:
            gesture_name = "UNDERWATER"
            gesture_color = (200, 150, 50)
        elif thumb_only_detected or slowmo_active:
            gesture_name = "SLOW-MO"
            gesture_color = (180, 180, 255)
        elif three_fingers_detected:
            gesture_name = "COLOR POP"
            gesture_color = (100, 200, 255)

        if gesture_name:
            draw_active_gesture_label(frame, gesture_name, gesture_color)

        cv2.imshow('Gesture Camera - Ultimate Edition', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == ord('Q'):
            break
        elif key == ord('h') or key == ord('H'):
            show_help = not show_help
        elif key == ord('s') or key == ord('S'):
            # Screenshot!
            filepath, filename = save_screenshot(frame, captures_dir)
            screenshot_flash = 1.0
            screenshot_count += 1
            # Buat thumbnail (resize kecil)
            thumb_h = 90
            thumb_w = int(w * (thumb_h / h))
            screenshot_thumbnail = cv2.resize(frame, (thumb_w, thumb_h))
            screenshot_display_time = current_time
            screenshot_filepath = filepath
            print(f"  📸 Screenshot saved: {filepath}")
        elif key == ord('r') or key == ord('R'):
            if not is_recording:
                # Start recording
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                rec_filename = f"recording_{timestamp}.avi"
                rec_filepath = os.path.join(captures_dir, rec_filename)
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                video_writer = cv2.VideoWriter(rec_filepath, fourcc, 20.0, (w, h))
                is_recording = True
                rec_start_time = current_time
                rec_saved_filepath = rec_filepath
                print(f"  ⏺️  Recording started: {rec_filepath}")
            else:
                # Stop recording
                is_recording = False
                if video_writer is not None:
                    video_writer.release()
                    video_writer = None
                rec_save_notification_time = current_time
                print(f"  ⏹️  Recording saved: {rec_saved_filepath}")

    # Cleanup
    if video_writer is not None:
        video_writer.release()
    cap.release()
    cv2.destroyAllWindows()
    print("\nKamera ditutup. Sampai jumpa!")

if __name__ == "__main__":
    main()