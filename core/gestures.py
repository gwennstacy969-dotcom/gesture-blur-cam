import math

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

def is_heart_gesture(hands, detector):
    """
    Deteksi gesture hati/love dari dua tangan.
    Returns: (detected: bool, center: tuple, size: int)
    """
    if len(hands) != 2:
        return False, None, 0

    lm1 = hands[0]["lmList"]
    lm2 = hands[1]["lmList"]

    thumb1 = (lm1[4][0], lm1[4][1])
    index1 = (lm1[8][0], lm1[8][1])
    thumb2 = (lm2[4][0], lm2[4][1])
    index2 = (lm2[8][0], lm2[8][1])

    f1 = detector.fingersUp(hands[0])
    f2 = detector.fingersUp(hands[1])

    other_fingers_down_1 = (f1[2] == 0 and f1[3] == 0 and f1[4] == 0)
    other_fingers_down_2 = (f2[2] == 0 and f2[3] == 0 and f2[4] == 0)

    if not (other_fingers_down_1 and other_fingers_down_2):
        return False, None, 0

    thresh = 80 
    index_dist = distance(index1, index2)
    thumb_dist = distance(thumb1, thumb2)

    if index_dist < thresh and thumb_dist < thresh:
        top = ((index1[0] + index2[0]) // 2, (index1[1] + index2[1]) // 2)
        bottom = ((thumb1[0] + thumb2[0]) // 2, (thumb1[1] + thumb2[1]) // 2)
        cx = (top[0] + bottom[0]) // 2
        cy = (top[1] + bottom[1]) // 2
        heart_size = max(30, int(distance(top, bottom) * 0.7))
        return True, (cx, cy), heart_size

    if thumb_dist < thresh and index_dist < thresh * 1.5:
        top = ((thumb1[0] + thumb2[0]) // 2, (thumb1[1] + thumb2[1]) // 2)
        bottom = ((index1[0] + index2[0]) // 2, (index1[1] + index2[1]) // 2)
        cx = (top[0] + bottom[0]) // 2
        cy = (top[1] + bottom[1]) // 2
        heart_size = max(30, int(distance(top, bottom) * 0.7))
        return True, (cx, cy), heart_size

    cross_dist_1 = distance(index1, thumb2)
    cross_dist_2 = distance(index2, thumb1)

    if cross_dist_1 < thresh and cross_dist_2 < thresh:
        top1 = ((index1[0] + thumb2[0]) // 2, (index1[1] + thumb2[1]) // 2)
        top2 = ((index2[0] + thumb1[0]) // 2, (index2[1] + thumb1[1]) // 2)
        cx = (top1[0] + top2[0]) // 2
        cy = (top1[1] + top2[1]) // 2
        heart_size = max(30, int(distance(top1, top2) * 0.8))
        return True, (cx, cy), heart_size

    return False, None, 0

def is_rectangle_gesture(hands, detector):
    """
    Deteksi gesture kotak/rectangle dari dua tangan.
    Returns: (detected: bool, rect_coords: tuple(x_min, y_min, x_max, y_max))
    """
    if len(hands) != 2:
        return False, None

    f1 = detector.fingersUp(hands[0])
    f2 = detector.fingersUp(hands[1])

    if sum(f1) < 4 or sum(f2) < 4:
        return False, None

    lm1 = hands[0]["lmList"]
    lm2 = hands[1]["lmList"]

    tips_1 = [(lm1[i][0], lm1[i][1]) for i in [4, 8, 12, 16, 20]]
    tips_2 = [(lm2[i][0], lm2[i][1]) for i in [4, 8, 12, 16, 20]]

    all_points = tips_1 + tips_2 + [(lm1[0][0], lm1[0][1]), (lm2[0][0], lm2[0][1])]

    all_x = [p[0] for p in all_points]
    all_y = [p[1] for p in all_points]

    x_min = min(all_x)
    x_max = max(all_x)
    y_min = min(all_y)
    y_max = max(all_y)

    rect_w = x_max - x_min
    rect_h = y_max - y_min
    if rect_w < 80 or rect_h < 80:
        return False, None

    cx1 = sum(p[0] for p in tips_1) / len(tips_1)
    cx2 = sum(p[0] for p in tips_2) / len(tips_2)
    hand_separation = abs(cx1 - cx2)
    if hand_separation < 60:
        return False, None

    return True, (x_min, y_min, x_max, y_max)

def is_spiderman_gesture(hand, detector):
    """
    Deteksi pose Spider-Man (web shooter):
    - Jempol: UP (1)
    - Telunjuk: UP (1)
    - Jari Tengah: DOWN (0)
    - Jari Manis: DOWN (0)
    - Kelingking: UP (1)
    Returns: (detected: bool, origin: tuple(x, y), dir_vec: tuple(dx, dy), angle: float)
    """
    fingers = detector.fingersUp(hand)
    if fingers != [1, 1, 0, 0, 1]:
        return False, (0, 0), (0, 0), 0.0

    lm = hand["lmList"]
    # Titik keluar jaring: area tengah telapak tangan / dekat pergelangan tangan
    w_x, w_y = lm[0][0], lm[0][1]
    m_x, m_y = lm[9][0], lm[9][1]
    origin = (int(w_x * 0.45 + m_x * 0.55), int(w_y * 0.45 + m_y * 0.55))

    # Arah jaring: dari pergelangan tangan (lm[0]) menuju ujung telunjuk (lm[8]) dan kelingking (lm[20])
    tip_x = (lm[8][0] + lm[20][0]) / 2.0
    tip_y = (lm[8][1] + lm[20][1]) / 2.0
    vx = tip_x - w_x
    vy = tip_y - w_y
    length = math.hypot(vx, vy)
    if length < 1.0:
        dir_vec = (0.0, -1.0)
        angle = -math.pi / 2
    else:
        dir_vec = (vx / length, vy / length)
        angle = math.atan2(dir_vec[1], dir_vec[0])

    return True, origin, dir_vec, angle

def analyze_face_expression(face):
    """
    Menganalisa ekspresi wajah dari 468 landmarks:
    - 'senyum' : bibir melebar & sudut mulut naik (senyum ceria)
    - 'sedih'  : sudut mulut turun / raut murung & sedih
    - 'mangap' : mulut terbuka lebar (tertawa / mangap)
    - 'neutral': ekspresi biasa
    Returns: (expr_type: str, intensity: float, info: dict)
    """
    if not face or len(face) < 468:
        return 'neutral', 0.0, {}

    x33, y33 = face[33][0], face[33][1]
    x263, y263 = face[263][0], face[263][1]
    dx = x263 - x33
    dy = y263 - y33
    eye_dist = math.hypot(dx, dy)
    if eye_dist < 15:
        return 'neutral', 0.0, {}

    # Normalisasi rotasi kepala (sudut garis mata)
    angle = math.atan2(dy, dx)
    cos_a = math.cos(-angle)
    sin_a = math.sin(-angle)
    mid_x = (x33 + x263) / 2.0
    mid_y = (y33 + y263) / 2.0

    def rot(pt):
        rx = pt[0] - mid_x
        ry = pt[1] - mid_y
        return (rx * cos_a - ry * sin_a, rx * sin_a + ry * cos_a)

    p61 = rot(face[61])    # Sudut mulut kiri
    p291 = rot(face[291])  # Sudut mulut kanan
    p13 = rot(face[13])    # Bibir atas dalam
    p14 = rot(face[14])    # Bibir bawah dalam
    p0 = rot(face[0])      # Bibir atas luar
    p17 = rot(face[17])    # Bibir bawah luar

    mouth_w = math.hypot(p291[0] - p61[0], p291[1] - p61[1])
    mouth_h = abs(p14[1] - p13[1])
    outer_mouth_h = abs(p17[1] - p0[1])

    mouth_open_ratio = mouth_h / eye_dist
    mouth_width_ratio = mouth_w / eye_dist

    corners_y = (p61[1] + p291[1]) / 2.0
    center_y = (p13[1] + p14[1]) / 2.0
    # Positif jika sudut mulut terangkat ke atas (senyum)
    corner_lift = (center_y - corners_y) / eye_dist

    # Alis untuk indikator sedih
    p107 = rot(face[107])
    p70 = rot(face[70])
    p336 = rot(face[336])
    p300 = rot(face[300])
    brow_inner_y = (p107[1] + p336[1]) / 2.0
    brow_outer_y = (p70[1] + p300[1]) / 2.0
    brow_sad_ratio = (brow_outer_y - brow_inner_y) / eye_dist

    mouth_center_px = (int((face[13][0] + face[14][0]) // 2), int((face[13][1] + face[14][1]) // 2))
    eyes_center_px = (int(mid_x), int(mid_y))
    left_eye_px = (face[159][0], face[159][1])
    right_eye_px = (face[386][0], face[386][1])
    left_cheek_px = (face[234][0], face[234][1])
    right_cheek_px = (face[454][0], face[454][1])
    forehead_px = (face[10][0], face[10][1])

    info = {
        "mouth_center": mouth_center_px,
        "eyes_center": eyes_center_px,
        "left_eye": left_eye_px,
        "right_eye": right_eye_px,
        "left_cheek": left_cheek_px,
        "right_cheek": right_cheek_px,
        "forehead": forehead_px,
        "eye_dist": eye_dist,
        "mouth_w": mouth_w,
        "mouth_h": mouth_h,
        "mouth_open_ratio": mouth_open_ratio,
        "corner_lift": corner_lift
    }

    # 1. Prioritas MANGAP (tertawa / mulut terbuka lebar)
    if mouth_open_ratio > 0.19 or (outer_mouth_h / max(1, mouth_w) > 0.46):
        intensity = min(1.0, (mouth_open_ratio - 0.16) / 0.18)
        return 'mangap', max(0.3, intensity), info

    # 2. SENYUM (sudut mulut naik & mulut melebar ceria)
    if (corner_lift > 0.020 and mouth_width_ratio > 0.74) or (mouth_width_ratio > 0.82 and corner_lift > 0.005):
        intensity = min(1.0, max((mouth_width_ratio - 0.74) / 0.15, (corner_lift - 0.015) / 0.04))
        return 'senyum', max(0.3, intensity), info

    # 3. SEDIH (sudut bibir turun ke bawah & raut murung/alis sedih)
    if (corner_lift < -0.035) or (corner_lift < -0.018 and brow_sad_ratio > 0.030):
        intensity = min(1.0, max((-corner_lift) / 0.045, brow_sad_ratio / 0.05))
        return 'sedih', max(0.3, intensity), info

    return 'neutral', 0.0, info
