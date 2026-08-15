import cv2
import math
import time
import random
import numpy as np
from cvzone.HandTrackingModule import HandDetector

# ==========================================
#  Gesture Camera - Ultimate Edition (10 Gestures)
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

    # Gambar garis koneksi
    for c in connections:
        x1, y1 = lmList[c[0]][0], lmList[c[0]][1]
        x2, y2 = lmList[c[1]][0], lmList[c[1]][1]
        cv2.line(frame, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)

    # Gambar titik landmark
    for i, lm in enumerate(lmList):
        x, y = lm[0], lm[1]
        # Ujung jari (4, 8, 12, 16, 20) diberi warna dan ukuran berbeda
        if i in [4, 8, 12, 16, 20]:
            cv2.circle(frame, (x, y), 7, (0, 200, 255), cv2.FILLED)
            cv2.circle(frame, (x, y), 9, (255, 255, 255), 2, cv2.LINE_AA)
        else:
            cv2.circle(frame, (x, y), 4, color, cv2.FILLED)


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


def draw_night_vision_hud(frame, level):
    """Gambar HUD overlay night vision: crosshair, corner brackets, text."""
    if level < 0.3:
        return
    h, w = frame.shape[:2]
    alpha = min(1.0, level)
    color = (0, int(200 * alpha), 0)

    # Corner brackets
    blen = 30
    corners = [
        ((10, 10), (10 + blen, 10), (10, 10 + blen)),
        ((w - 10, 10), (w - 10 - blen, 10), (w - 10, 10 + blen)),
        ((10, h - 10), (10 + blen, h - 10), (10, h - 10 - blen)),
        ((w - 10, h - 10), (w - 10 - blen, h - 10), (w - 10, h - 10 - blen)),
    ]
    for corner, h_end, v_end in corners:
        cv2.line(frame, corner, h_end, color, 1, cv2.LINE_AA)
        cv2.line(frame, corner, v_end, color, 1, cv2.LINE_AA)

    # Crosshair di tengah
    cx, cy = w // 2, h // 2
    csize = 15
    cv2.line(frame, (cx - csize, cy), (cx + csize, cy), color, 1, cv2.LINE_AA)
    cv2.line(frame, (cx, cy - csize), (cx, cy + csize), color, 1, cv2.LINE_AA)
    cv2.circle(frame, (cx, cy), csize + 5, color, 1, cv2.LINE_AA)

    # Labels
    cv2.putText(frame, "NV MODE", (15, h - 20), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, color, 1, cv2.LINE_AA)
    time_str = time.strftime("%H:%M:%S")
    cv2.putText(frame, time_str, (w - 110, h - 20), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, color, 1, cv2.LINE_AA)


# ==========================================
#  Glitch / VHS Effect
# ==========================================

def apply_glitch_effect(frame, level):
    """Efek glitch/VHS retro: RGB split, horizontal bars, scanlines."""
    if level <= 0:
        return frame
    h, w = frame.shape[:2]
    result = frame.copy()

    # RGB channel split (chromatic aberration)
    shift = max(1, int(level * random.randint(5, 15)))
    if shift < w:
        # Red channel geser kanan
        result[:, shift:, 2] = frame[:, :w - shift, 2]
        result[:, :shift, 2] = 0
        # Blue channel geser kiri
        result[:, :w - shift, 0] = frame[:, shift:, 0]
        result[:, w - shift:, 0] = 0

    # Random horizontal displacement bars
    num_bars = int(level * random.randint(3, 8))
    for _ in range(num_bars):
        y_start = random.randint(0, h - 1)
        bar_h = random.randint(2, max(3, int(15 * level)))
        y_end = min(y_start + bar_h, h)
        shift_x = random.randint(int(-25 * level), int(25 * level))
        if shift_x > 0 and shift_x < w:
            result[y_start:y_end, shift_x:] = frame[y_start:y_end, :w - shift_x]
        elif shift_x < 0 and abs(shift_x) < w:
            result[y_start:y_end, :w + shift_x] = frame[y_start:y_end, -shift_x:]

    # VHS scanlines
    result[::3, :] = (result[::3, :].astype(np.float32) * 0.85).astype(np.uint8)

    # Random color tint flicker
    if random.random() < 0.3 * level:
        tint = np.zeros_like(result)
        tint[:, :] = (random.randint(0, 20), 0, random.randint(0, 25))
        result = cv2.add(result, tint)

    # Random horizontal white noise bar
    if random.random() < 0.4 * level:
        y_noise = random.randint(0, h - 3)
        noise_h = random.randint(1, 4)
        y_end_n = min(y_noise + noise_h, h)
        noise_bar = np.random.randint(0, 255, (y_end_n - y_noise, w, 3), dtype=np.uint8)
        alpha_noise = 0.3 * level
        result[y_noise:y_end_n] = cv2.addWeighted(
            noise_bar, alpha_noise, result[y_noise:y_end_n], 1 - alpha_noise, 0
        )

    return cv2.addWeighted(result, level, frame, 1 - level, 0)


def draw_glitch_hud(frame, level):
    """Gambar overlay VHS-style: REC indicator, timestamp."""
    if level < 0.3:
        return
    h, w = frame.shape[:2]
    alpha = min(1.0, level)

    # "REC" indicator dengan titik merah berkedip
    rec_color = (0, 0, int(255 * alpha))
    txt_color = (int(200 * alpha), int(200 * alpha), int(200 * alpha))
    cv2.circle(frame, (25, 25), 6, rec_color, cv2.FILLED)
    cv2.putText(frame, "REC", (38, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, txt_color, 1, cv2.LINE_AA)

    # "PLAY" di pojok kanan atas
    cv2.putText(frame, "PLAY >>", (w - 110, 25), cv2.FONT_HERSHEY_SIMPLEX,
                0.4, txt_color, 1, cv2.LINE_AA)

    # VHS timestamp di bawah
    time_str = time.strftime("%Y/%m/%d  %H:%M:%S")
    ts_color = (int(200 * alpha), int(200 * alpha), int(80 * alpha))
    cv2.putText(frame, time_str, (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX,
                0.4, ts_color, 1, cv2.LINE_AA)


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
    dim_color = (int(100 * alpha), int(100 * alpha), int(50 * alpha))
    cv2.circle(frame, (cx, cy), radius, color, 1, cv2.LINE_AA)
    cv2.circle(frame, (cx, cy), radius + 3, dim_color, 1, cv2.LINE_AA)
    # Crosshair kecil
    csize = 8
    cv2.line(frame, (cx - csize, cy), (cx + csize, cy), color, 1, cv2.LINE_AA)
    cv2.line(frame, (cx, cy - csize), (cx, cy + csize), color, 1, cv2.LINE_AA)
    # Label
    cv2.putText(frame, "SPOTLIGHT", (cx - 35, cy - radius - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)


# ==========================================
#  Color Invert / Negative Effect
# ==========================================

def apply_color_invert(frame, level):
    """Inversi warna (efek negatif) dengan scanline ala film."""
    if level <= 0:
        return frame

    # Invert warna
    inverted = cv2.bitwise_not(frame)

    # Blend
    result = cv2.addWeighted(inverted, level, frame, 1 - level, 0)

    # Scanlines tipis ala film negatif
    if level > 0.3:
        result[::4, :] = (result[::4, :].astype(np.float32) *
                          (0.85 + 0.15 * (1 - level))).astype(np.uint8)

    return result


def draw_invert_border(frame, level):
    """Gambar border film negatif dengan perforasi."""
    if level < 0.3:
        return
    h, w = frame.shape[:2]
    alpha = min(1.0, level)
    color = (int(180 * alpha), int(120 * alpha), int(255 * alpha))

    # Border tipis
    cv2.rectangle(frame, (5, 5), (w - 5, h - 5), color, 1, cv2.LINE_AA)

    # Film perforations (lubang film) di kiri dan kanan
    perf_size = 6
    perf_gap = 25
    for y in range(15, h - 15, perf_gap):
        cv2.rectangle(frame, (2, y), (2 + perf_size, y + perf_size * 2), color, 1)
        cv2.rectangle(frame, (w - 2 - perf_size, y), (w - 2, y + perf_size * 2), color, 1)

    # Label "NEGATIVE" di atas
    cv2.putText(frame, "NEGATIVE", (w // 2 - 40, 20), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, color, 1, cv2.LINE_AA)


def main():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("ERROR: Kamera tidak bisa dibuka!")
        return

    # Detektor tangan: butuh 2 tangan untuk gesture segitiga & love
    detector = HandDetector(detectionCon=0.7, maxHands=2)

    print("=" * 55)
    print("  GESTURE CAMERA - ULTIMATE EDITION (10 Gestures)")
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
    print("  -----------------------------------------------")
    print("  [2 TANGAN]")
    print("  Segitiga              : Grayscale")
    print("  Love / Heart          : Efek Hati")
    print("  Kotak (tangan buka)   : Anime Filter")
    print("  -----------------------------------------------")
    print("  Tekan 'Q' untuk keluar")
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

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        current_time = time.time()

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

                # LOGIKA 6: Rock Sign (Telunjuk + Kelingking UP) → Glitch/VHS
                elif fingers == [0, 1, 0, 0, 1] or fingers == [1, 1, 0, 0, 1]:
                    rock_detected = True

                # LOGIKA 7: Telunjuk Saja → Spotlight
                elif fingers == [0, 1, 0, 0, 0]:
                    spotlight_detected = True
                    lmList = hand1["lmList"]
                    spotlight_pos = (lmList[8][0], lmList[8][1])

                # LOGIKA 8: Kelingking Saja → Color Invert
                elif fingers == [0, 0, 0, 0, 1]:
                    pinky_detected = True

                # LOGIKA 9: Kepalan Tangan → Night Vision
                elif fingers == [0, 0, 0, 0, 0]:
                    fist_detected = True

                # LOGIKA 10: Telapak Terbuka (1 tangan saja) → Freeze Frame
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

        cv2.imshow('Gesture Camera - Ultimate Edition', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == ord('Q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("\nKamera ditutup. Sampai jumpa!")

if __name__ == "__main__":
    main()