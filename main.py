import cv2
import math
import time
import random
import numpy as np
from cvzone.HandTrackingModule import HandDetector

# ==========================================
#  Gesture Camera - Full Edition
#  1. Peace Sign (✌️) = Blur Seluruh Layar
#  2. Jempol & Telunjuk = Blur dalam kotak
#  3. Segitiga △ (dua tangan) = Grayscale
#  4. Love/Heart ❤️ (dua tangan) = Efek Hati
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


def main():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("ERROR: Kamera tidak bisa dibuka!")
        return

    # Detektor tangan: butuh 2 tangan untuk gesture segitiga & love
    detector = HandDetector(detectionCon=0.7, maxHands=2)

    print("=" * 50)
    print("  GESTURE CAMERA - FULL EDITION")
    print("  - Jari & landmark tangan terdeteksi otomatis")
    print("  - Peace Sign (V) : Blur Full Screen")
    print("  - Jempol & Telunjuk : Blur Area Kotak")
    print("  - Segitiga △ (2 tangan) : Grayscale")
    print("  - Love ❤️ (2 tangan)   : Efek Hati")
    print("  Tekan 'Q' untuk keluar")
    print("=" * 50)

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

        if hands:
            # Gambar landmark untuk setiap tangan yang terdeteksi
            hand_colors = [(0, 255, 150), (255, 150, 0)]  # Hijau, Biru-oranye
            for i, hand in enumerate(hands):
                color = hand_colors[i % len(hand_colors)]
                draw_hand_landmarks(frame, hand, color)

            # --- Deteksi Love/Heart Gesture (prioritas tinggi, 2 tangan) ---
            if len(hands) == 2:
                love_raw_detected, love_center, love_size = is_heart_gesture(hands, detector)

            # --- Gesture 1 tangan (ambil tangan pertama) ---
            if not love_raw_detected:
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

                # Deteksi segitiga: butuh 2 tangan
                if len(hands) == 2 and not love_raw_detected:
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



        cv2.imshow('Gesture Camera - Full Edition', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == ord('Q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("\nKamera ditutup. Sampai jumpa!")

if __name__ == "__main__":
    main()