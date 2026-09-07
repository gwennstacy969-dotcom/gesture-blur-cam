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
