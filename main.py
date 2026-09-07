import cv2
import time
import random
import os
import numpy as np
from cvzone.HandTrackingModule import HandDetector
import math

from core.gestures import is_heart_gesture, is_rectangle_gesture, is_triangle, distance
from core.effects import (
    apply_anime_filter, apply_night_vision, apply_glitch_effect, apply_spotlight,
    apply_color_invert, apply_thermal_vision, apply_underwater_effect, apply_color_pop
)
from core.ui import (
    draw_hand_landmarks, HeartParticle, draw_heart_vignette, draw_anime_frame,
    draw_night_vision_hud, draw_glitch_hud, draw_spotlight_ring, draw_invert_border,
    draw_thermal_hud, BubbleParticle, draw_underwater_hud, draw_color_pop_overlay,
    save_screenshot, draw_screenshot_thumbnail, draw_screenshot_flash, draw_recording_indicator,
    draw_save_notification, draw_fps_counter, draw_active_gesture_label, draw_info_panel,
    draw_box_blur_border, draw_startup_splash, generate_heart_points
)

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
    consecutive_failures = 0
    max_consecutive_failures = 30  # Toleransi frame drop berturut-turut

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            consecutive_failures += 1
            if consecutive_failures >= max_consecutive_failures:
                print("ERROR: Kamera tidak merespon setelah banyak percobaan. Keluar...")
                break
            # Frame drop biasa, skip dan coba lagi
            time.sleep(0.01)
            continue
        consecutive_failures = 0  # Reset counter kalau berhasil baca frame

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        current_time = time.time()
        frame_count += 1
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0 or math.isnan(fps):
            fps = 30 # default

        # Initialize bubble particles once we know frame dimensions
        if not bubbles_initialized:
            bubble_particles = [BubbleParticle(w, h) for _ in range(25)]
            bubbles_initialized = True

        # Deteksi tangan (draw=False, kita gambar sendiri biar lebih bagus)
        try:
            hands, frame = detector.findHands(frame, draw=False)
        except Exception:
            hands = []

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

                    thumb1 = (lm1[4][0], lm1[4][1])
                    index1 = (lm1[8][0], lm1[8][1])
                    thumb2 = (lm2[4][0], lm2[4][1])
                    index2 = (lm2[8][0], lm2[8][1])

                    thumb_dist = distance(thumb1, thumb2)
                    thresh = 60  # Jarak maksimal ujung jempol berdekatan

                    if thumb_dist < thresh:
                        top = ((thumb1[0] + thumb2[0]) // 2, (thumb1[1] + thumb2[1]) // 2)
                        bottom_left = index1
                        bottom_right = index2

                        if is_triangle(top, bottom_left, bottom_right, min_side=50):
                            triangle_detected = True
                            triangle_pts = (top, bottom_left, bottom_right)

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
            try:
                x_min, y_min, x_max, y_max = box_coords
                roi = frame[y_min:y_max, x_min:x_max]
                if roi.size > 0:
                    roi_blurred = cv2.GaussianBlur(roi, (71, 71), 0)
                    frame[y_min:y_max, x_min:x_max] = roi_blurred
            except Exception:
                pass

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
        if love_raw_detected:
            if not love_detecting:
                love_detecting = True
                love_hold_start = current_time
            hold_elapsed = current_time - love_hold_start
            hold_progress = min(1.0, hold_elapsed / love_hold_duration)

            heart_center = love_center
            heart_size = love_size

            if not love_confirmed:
                ring_cx, ring_cy = love_center
                ring_radius = 35
                cv2.circle(frame, (ring_cx, ring_cy), ring_radius, (80, 80, 80), 2, cv2.LINE_AA)
                end_angle = int(360 * hold_progress)
                if end_angle > 0:
                    cv2.ellipse(frame, (ring_cx, ring_cy), (ring_radius, ring_radius),
                               -90, 0, end_angle, (130, 100, 255), 3, cv2.LINE_AA)
                mini_pts = generate_heart_points(ring_cx, ring_cy, 10, num_points=30)
                if len(mini_pts) >= 3:
                    mini_arr = np.array(mini_pts, dtype=np.int32)
                    r_val = int(150 + 105 * hold_progress)
                    cv2.fillPoly(frame, [mini_arr], (100, 80, r_val))
                    cv2.polylines(frame, [mini_arr], True, (180, 140, 255), 1, cv2.LINE_AA)

            if hold_progress >= 1.0 and not love_confirmed:
                love_confirmed = True
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
            love_detecting = False
            love_hold_start = 0
            love_confirmed = False

        if love_confirmed:
            heart_level = min(1.0, heart_level + heart_speed)
        else:
            heart_level = max(0.0, heart_level - heart_speed * 0.5)

        if heart_level > 0.01:
            draw_heart_vignette(frame, alpha=heart_level * 0.6)
            heart_spawn_timer += 1
            if heart_spawn_timer >= 3: 
                heart_spawn_timer = 0
                num_new = random.randint(2, 5)
                for _ in range(num_new):
                    spread = max(30, heart_size)
                    sx = heart_center[0] + random.randint(-spread, spread)
                    sy = heart_center[1] + random.randint(-spread // 2, spread // 2)
                    p = HeartParticle(sx, sy, w, h)
                    p.size = random.randint(4, 10) 
                    heart_particles.append(p)

            if random.random() < 0.5:
                sx = random.randint(0, w)
                sy = random.randint(0, h)
                p = HeartParticle(sx, sy, w, h)
                p.is_sparkle = True
                p.size = random.randint(3, 7)
                p.decay = random.uniform(0.02, 0.04)
                p.vy = random.uniform(-1.5, -0.3)
                heart_particles.append(p)

        alive_particles = []
        for p in heart_particles:
            p.update()
            if p.is_alive():
                p.draw(frame)
                alive_particles.append(p)
        heart_particles = alive_particles

        if len(heart_particles) > 150:
            heart_particles = heart_particles[-100:]

        # 5. 🎌 Efek Anime (Rectangle - kedua tangan terbuka)
        if rect_detected and rect_coords:
            anime_level = min(1.0, anime_level + anime_speed)
            anime_rect = rect_coords
        else:
            anime_level = max(0.0, anime_level - anime_speed * 0.5)

        if anime_level > 0.01 and anime_rect:
            try:
                ax_min, ay_min, ax_max, ay_max = anime_rect
                ax_min = max(0, ax_min)
                ay_min = max(0, ay_min)
                ax_max = min(w, ax_max)
                ay_max = min(h, ay_max)

                if ax_max - ax_min > 10 and ay_max - ay_min > 10:
                    roi = frame[ay_min:ay_max, ax_min:ax_max].copy()
                    anime_roi = apply_anime_filter(roi)
                    blended = cv2.addWeighted(anime_roi, anime_level, roi, 1 - anime_level, 0)
                    frame[ay_min:ay_max, ax_min:ax_max] = blended
                    draw_anime_frame(frame, ax_min, ay_min, ax_max, ay_max, anime_level)
            except Exception:
                pass

        # 6. Night Vision (Kepalan Tangan)
        if fist_detected:
            nv_level = min(1.0, nv_level + nv_speed)
        else:
            nv_level = max(0.0, nv_level - nv_speed)

        if nv_level > 0.01:
            try:
                frame = apply_night_vision(frame, nv_level)
                draw_night_vision_hud(frame, nv_level)
            except Exception:
                pass

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
                fh, fw = frame.shape[:2]
                border = 15
                display = cv2.addWeighted(freeze_frame, freeze_level, frame, 1 - freeze_level, 0)
                cv2.rectangle(display, (border, border), (fw - border, fh - border),
                              (240, 240, 240), 2, cv2.LINE_AA)
                cv2.rectangle(display, (border + 3, border + 3),
                              (fw - border - 3, fh - border - 3),
                              (200, 200, 200), 1, cv2.LINE_AA)
                label_color = (200, 200, 200)
                cv2.putText(display, "CAPTURED", (fw // 2 - 50, fh - border - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, label_color, 1, cv2.LINE_AA)
                cap_time_str = time.strftime("%H:%M:%S",
                                             time.localtime(freeze_capture_time))
                cv2.putText(display, cap_time_str, (border + 10, border + 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, label_color, 1, cv2.LINE_AA)
                frame = display
            else:
                freeze_level = max(0.0, freeze_level - freeze_speed)
                if freeze_level > 0.01:
                    frame = cv2.addWeighted(freeze_frame, freeze_level,
                                            frame, 1 - freeze_level, 0)
                else:
                    freeze_frame = None
                    freeze_cooldown = current_time

        if freeze_flash > 0.01:
            white = np.ones_like(frame, dtype=np.uint8) * 255
            frame = cv2.addWeighted(white, freeze_flash * 0.7,
                                    frame, 1 - freeze_flash * 0.7, 0)
            freeze_flash *= 0.75  

        # 8. Glitch / VHS (Rock Sign)
        if rock_detected:
            glitch_level = min(1.0, glitch_level + glitch_speed)
        else:
            glitch_level = max(0.0, glitch_level - glitch_speed)

        if glitch_level > 0.01:
            try:
                frame = apply_glitch_effect(frame, glitch_level)
                draw_glitch_hud(frame, glitch_level)
            except Exception:
                pass

        # 9. Spotlight (Telunjuk Saja)
        if spotlight_detected and spotlight_pos:
            spot_level = min(1.0, spot_level + spot_speed)
            spot_pos = spotlight_pos
        else:
            spot_level = max(0.0, spot_level - spot_speed)

        if spot_level > 0.01:
            try:
                frame = apply_spotlight(frame, spot_pos[0], spot_pos[1],
                                        spot_radius, spot_level)
                draw_spotlight_ring(frame, spot_pos[0], spot_pos[1],
                                    spot_radius, spot_level)
            except Exception:
                pass

        # 10. Color Invert (Kelingking Saja)
        if pinky_detected:
            invert_level = min(1.0, invert_level + invert_speed)
        else:
            invert_level = max(0.0, invert_level - invert_speed)

        if invert_level > 0.01:
            try:
                frame = apply_color_invert(frame, invert_level)
                draw_invert_border(frame, invert_level)
            except Exception:
                pass

        # ====================================================
        # FITUR BARU (11-14)
        # ====================================================

        # 11. 🔥 Thermal Vision (Jari Tengah Saja)
        if middle_detected:
            thermal_level = min(1.0, thermal_level + thermal_speed)
        else:
            thermal_level = max(0.0, thermal_level - thermal_speed)

        if thermal_level > 0.01:
            try:
                frame = apply_thermal_vision(frame, thermal_level)
                draw_thermal_hud(frame, thermal_level)
            except Exception:
                pass

        # 12. 🌊 Underwater (Jari Manis Saja)
        if ring_detected:
            underwater_level = min(1.0, underwater_level + underwater_speed)
        else:
            underwater_level = max(0.0, underwater_level - underwater_speed)

        if underwater_level > 0.01:
            try:
                frame = apply_underwater_effect(frame, underwater_level, current_time)
                draw_underwater_hud(frame, underwater_level, current_time)
                for bubble in bubble_particles:
                    bubble.update(current_time)
                    bubble.draw(frame, underwater_level)
            except Exception:
                pass


        # 14. 🌈 Color Pop (Telunjuk + Tengah + Manis)
        if three_fingers_detected and three_fingers_hand is not None:
            color_pop_level = min(1.0, color_pop_level + color_pop_speed)
            color_pop_hand = three_fingers_hand
        else:
            color_pop_level = max(0.0, color_pop_level - color_pop_speed)

        if color_pop_level > 0.01 and color_pop_hand is not None:
            try:
                frame = apply_color_pop(frame, color_pop_hand, color_pop_level)
                draw_color_pop_overlay(frame, color_pop_level)
            except Exception:
                pass

        # ====================================================
        # OVERLAY VISUAL (segitiga, border, dll)
        # ====================================================

        if triangle_detected and triangle_pts:
            pts = triangle_pts
            for i in range(3):
                p1 = pts[i]
                p2 = pts[(i + 1) % 3]
                cv2.line(frame, p1, p2, (0, 100, 255), 6, cv2.LINE_AA)
                cv2.line(frame, p1, p2, (0, 200, 255), 2, cv2.LINE_AA)
            for pt in pts:
                cv2.circle(frame, pt, 10, (0, 255, 255), cv2.FILLED)
                cv2.circle(frame, pt, 12, (255, 255, 255), 2, cv2.LINE_AA)

        if box_blur_detected and box_coords:
            draw_box_blur_border(frame, *box_coords)

        # ====================================================
        # SCREENSHOT & RECORDING
        # ====================================================

        if screenshot_flash > 0.01:
            frame = draw_screenshot_flash(frame, screenshot_flash)
            screenshot_flash *= 0.7

        if screenshot_thumbnail is not None:
            elapsed_ss = current_time - screenshot_display_time
            if elapsed_ss < 3.0:
                draw_screenshot_thumbnail(frame, screenshot_thumbnail,
                                          elapsed_ss, screenshot_filepath)
            else:
                screenshot_thumbnail = None

        if is_recording:
            rec_duration = current_time - rec_start_time
            blink = int(current_time * 2) % 2 == 0
            draw_recording_indicator(frame, True, rec_duration, blink)
            if video_writer is not None:
                video_writer.write(frame)

        if rec_save_notification_time > 0:
            remaining = 3.0 - (current_time - rec_save_notification_time)
            if remaining > 0:
                draw_save_notification(frame, remaining, rec_saved_filepath)
            else:
                rec_save_notification_time = 0

        if screenshot_count > 0:
            cv2.putText(frame, f"Photos: {screenshot_count}", (w - 100, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (150, 150, 150), 1, cv2.LINE_AA)

        # ====================================================
        # UI OVERLAY
        # ====================================================

        elapsed_start = current_time - start_time
        if elapsed_start < 3.0:
            splash_alpha = max(0.0, 1.0 - elapsed_start / 3.0)
            frame = draw_startup_splash(frame, splash_alpha)

        if show_help:
            draw_info_panel(frame)

        # draw fps
        draw_fps_counter(frame, fps)

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
            filepath, filename = save_screenshot(frame, captures_dir)
            screenshot_flash = 1.0
            screenshot_count += 1
            thumb_h = 90
            thumb_w = int(w * (thumb_h / h))
            screenshot_thumbnail = cv2.resize(frame, (thumb_w, thumb_h))
            screenshot_display_time = current_time
            screenshot_filepath = filepath
            print(f"  📸 Screenshot saved: {filepath}")
        elif key == ord('r') or key == ord('R'):
            if not is_recording:
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
                is_recording = False
                if video_writer is not None:
                    video_writer.release()
                    video_writer = None
                rec_save_notification_time = current_time
                print(f"  ⏹️  Recording saved: {rec_saved_filepath}")

    if video_writer is not None:
        video_writer.release()
    cap.release()
    cv2.destroyAllWindows()
    print("\nKamera ditutup. Sampai jumpa!")

if __name__ == "__main__":
    main()