import cv2
import math
import numpy as np
from cvzone.HandTrackingModule import HandDetector

# ==========================================
#  Gesture Blur Camera - Triangle Edition
#  Deteksi jari + Bentuk segitiga = Grayscale
#  Gunakan kedua tangan, pertemukan ujung
#  jempol & telunjuk membentuk segitiga △
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


def main():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("ERROR: Kamera tidak bisa dibuka!")
        return

    # Detektor tangan: butuh 2 tangan untuk gesture segitiga
    detector = HandDetector(detectionCon=0.7, maxHands=2)

    print("=" * 50)
    print("  GESTURE CAMERA - TRIANGLE GRAYSCALE EDITION")
    print("  - Jari & landmark tangan terdeteksi otomatis")
    print("  - Bentuk SEGITIGA dengan kedua tangan:")
    print("    Pertemukan ujung jempol & telunjuk △")
    print("    → Kamera jadi ABU-ABU (grayscale)")
    print("  Tekan 'Q' untuk keluar")
    print("=" * 50)

    # Smooth transition untuk grayscale
    gray_level = 0.0
    gray_speed = 0.12

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        # Deteksi tangan (draw=False, kita gambar sendiri biar lebih bagus)
        hands, frame = detector.findHands(frame, draw=False)

        triangle_detected = False
        triangle_pts = None

        if hands:
            # Gambar landmark untuk setiap tangan yang terdeteksi
            hand_colors = [(0, 255, 150), (255, 150, 0)]  # Hijau, Biru-oranye
            for i, hand in enumerate(hands):
                color = hand_colors[i % len(hand_colors)]
                draw_hand_landmarks(frame, hand, color)

            # Deteksi segitiga: butuh 2 tangan
            if len(hands) == 2:
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

        # Smooth transition grayscale
        if triangle_detected:
            gray_level = min(1.0, gray_level + gray_speed)
        else:
            gray_level = max(0.0, gray_level - gray_speed)

        # Terapkan grayscale dengan transisi halus
        if gray_level > 0.01:
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_bgr = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2BGR)
            frame = cv2.addWeighted(gray_bgr, gray_level, frame, 1 - gray_level, 0)

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

        # Status overlay di pojok kiri atas
        status = "TRIANGLE DETECTED!" if triangle_detected else "Detecting hands..."
        status_color = (0, 255, 255) if triangle_detected else (200, 200, 200)

        # Background kotak untuk teks status
        cv2.rectangle(frame, (10, 10), (350, 45), (0, 0, 0), cv2.FILLED)
        cv2.rectangle(frame, (10, 10), (350, 45), status_color, 1)
        cv2.putText(frame, status, (20, 35), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, status_color, 2, cv2.LINE_AA)

        # Tampilkan grayscale level jika aktif
        if gray_level > 0.01:
            pct = int(gray_level * 100)
            bar_w = int(gray_level * 200)
            cv2.rectangle(frame, (10, 55), (210, 80), (0, 0, 0), cv2.FILLED)
            cv2.rectangle(frame, (10, 55), (10 + bar_w, 80), (0, 180, 255), cv2.FILLED)
            cv2.rectangle(frame, (10, 55), (210, 80), (200, 200, 200), 1)
            cv2.putText(frame, f"Gray: {pct}%", (220, 75), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (200, 200, 200), 1, cv2.LINE_AA)

        cv2.imshow('Gesture Camera - Triangle Grayscale', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == ord('Q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("\nKamera ditutup. Sampai jumpa!")

if __name__ == "__main__":
    main()