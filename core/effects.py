import cv2
import numpy as np
import math
import random

def apply_anime_filter(roi):
    if roi.size == 0:
        return roi
    h_roi, w_roi = roi.shape[:2]
    if h_roi < 10 or w_roi < 10:
        return roi
    smooth = cv2.bilateralFilter(roi, 9, 75, 75)
    smooth = cv2.bilateralFilter(smooth, 9, 75, 75)
    div = 24
    smooth = (smooth // div) * div + div // 2
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 7)
    edges = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, blockSize=9, C=2
    )
    edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    cartoon = cv2.bitwise_and(smooth, edges_bgr)
    hsv = cv2.cvtColor(cartoon, cv2.COLOR_BGR2HSV)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1].astype(np.float32) * 1.5, 0, 255).astype(np.uint8)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2].astype(np.float32) * 1.1, 0, 255).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def apply_night_vision(frame, level):
    if level <= 0:
        return frame
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = np.clip(gray.astype(np.float32) * 1.3 + 10, 0, 255).astype(np.uint8)
    night = np.zeros_like(frame)
    night[:, :, 0] = (gray * 0.05).astype(np.uint8)
    night[:, :, 1] = gray
    night[:, :, 2] = (gray * 0.08).astype(np.uint8)
    noise = np.random.randint(0, 25, (h, w), dtype=np.uint8)
    night[:, :, 1] = np.clip(night[:, :, 1].astype(np.int16) + noise, 0, 255).astype(np.uint8)
    night[::2, :] = (night[::2, :].astype(np.float32) * 0.75).astype(np.uint8)
    cx_v, cy_v = w // 2, h // 2
    Y, X = np.ogrid[:h, :w]
    max_dist = math.sqrt(cx_v**2 + cy_v**2)
    dist = np.sqrt((X.astype(np.float32) - cx_v)**2 + (Y.astype(np.float32) - cy_v)**2)
    vignette = np.clip(1.0 - (dist / max_dist) * 0.7, 0.3, 1.0).astype(np.float32)
    for c in range(3):
        night[:, :, c] = (night[:, :, c].astype(np.float32) * vignette).astype(np.uint8)
    return cv2.addWeighted(night, level, frame, 1 - level, 0)

def apply_glitch_effect(frame, level, time_val=0):
    if level <= 0:
        return frame
    h, w = frame.shape[:2]
    result = frame.copy()
    osc = 0.6 + 0.4 * math.sin(time_val * 8.0)
    eff_level = level * osc
    shift = max(1, int(eff_level * random.randint(5, 18)))
    if shift < w:
        result[:, shift:, 2] = frame[:, :w - shift, 2]
        result[:, :shift, 2] = 0
        result[:, :w - shift, 0] = frame[:, shift:, 0]
        result[:, w - shift:, 0] = 0
    g_shift = max(1, int(eff_level * random.randint(2, 8)))
    if g_shift < h:
        result[g_shift:, :, 1] = frame[:h - g_shift, :, 1]
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
    result[::3, :] = (result[::3, :].astype(np.float32) * 0.85).astype(np.uint8)
    if random.random() < 0.3 * eff_level:
        tint = np.zeros_like(result)
        tint[:, :] = (random.randint(0, 20), 0, random.randint(0, 25))
        result = cv2.add(result, tint)
    if random.random() < 0.4 * eff_level:
        y_noise = random.randint(0, h - 3)
        noise_h = random.randint(1, 4)
        y_end_n = min(y_noise + noise_h, h)
        noise_bar = np.random.randint(0, 255, (y_end_n - y_noise, w, 3), dtype=np.uint8)
        alpha_noise = 0.3 * eff_level
        result[y_noise:y_end_n] = cv2.addWeighted(
            noise_bar, alpha_noise, result[y_noise:y_end_n], 1 - alpha_noise, 0
        )
    if eff_level > 0.5:
        shake_x = random.randint(-3, 3)
        shake_y = random.randint(-2, 2)
        M = np.float32([[1, 0, shake_x], [0, 1, shake_y]])
        result = cv2.warpAffine(result, M, (w, h))
    return cv2.addWeighted(result, level, frame, 1 - level, 0)

def apply_spotlight(frame, cx, cy, radius, level):
    if level <= 0:
        return frame
    h, w = frame.shape[:2]
    dark = (frame.astype(np.float32) * 0.12).astype(np.uint8)
    Y, X = np.ogrid[:h, :w]
    dist = np.sqrt((X.astype(np.float32) - cx)**2 + (Y.astype(np.float32) - cy)**2)
    mask = np.clip(1.0 - (dist / max(1, radius)), 0, 1).astype(np.float32)
    mask = mask ** 1.5 
    mask_3ch = np.stack([mask, mask, mask], axis=-1)
    spotlight = (frame.astype(np.float32) * mask_3ch +
                 dark.astype(np.float32) * (1 - mask_3ch)).astype(np.uint8)
    return cv2.addWeighted(spotlight, level, frame, 1 - level, 0)

def apply_color_invert(frame, level, frame_count=0):
    if level <= 0:
        return frame
    inverted = cv2.bitwise_not(frame)
    result = cv2.addWeighted(inverted, level, frame, 1 - level, 0)
    if level > 0.3:
        h_r, w_r = result.shape[:2]
        grain = np.random.randint(0, int(30 * level), (h_r, w_r), dtype=np.uint8)
        grain_bgr = cv2.cvtColor(grain, cv2.COLOR_GRAY2BGR)
        result = cv2.add(result, grain_bgr)
    if level > 0.3:
        result[::4, :] = (result[::4, :].astype(np.float32) *
                          (0.85 + 0.15 * (1 - level))).astype(np.uint8)
    return result

def apply_thermal_vision(frame, level):
    if level <= 0:
        return frame
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    thermal = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
    thermal = cv2.GaussianBlur(thermal, (3, 3), 0)
    noise = np.random.randint(0, int(8 * level), (h, w), dtype=np.uint8)
    noise_bgr = cv2.cvtColor(noise, cv2.COLOR_GRAY2BGR)
    thermal = cv2.add(thermal, noise_bgr)
    return cv2.addWeighted(thermal, level, frame, 1 - level, 0)

def apply_underwater_effect(frame, level, time_val=0):
    if level <= 0:
        return frame
    h, w = frame.shape[:2]
    result = frame.copy()
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
    tint_overlay = np.zeros_like(result, dtype=np.uint8)
    tint_overlay[:, :] = (140, 90, 20)
    tint_strength = 0.25 * level
    result = cv2.addWeighted(result, 1 - tint_strength, tint_overlay, tint_strength, 0)
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
    cy_v, cx_v = h // 2, w // 2
    Y, X = np.ogrid[:h, :w]
    max_dist = math.sqrt(cx_v**2 + cy_v**2)
    dist = np.sqrt((X.astype(np.float32) - cx_v)**2 + (Y.astype(np.float32) - cy_v)**2)
    vignette = np.clip(1.0 - (dist / max_dist) * 0.5 * level, 0.4, 1.0).astype(np.float32)
    for c_ch in range(3):
        result[:, :, c_ch] = (result[:, :, c_ch].astype(np.float32) * vignette).astype(np.uint8)
    return cv2.addWeighted(result, level, frame, 1 - level, 0)

def apply_color_pop(frame, hand, level):
    if level <= 0:
        return frame
    h, w = frame.shape[:2]
    lmList = hand["lmList"]
    palm_x = lmList[9][0]
    palm_y = lmList[9][1]
    sample_size = 20
    sx1 = max(0, palm_x - sample_size)
    sx2 = min(w, palm_x + sample_size)
    sy1 = max(0, palm_y - sample_size)
    sy2 = min(h, palm_y + sample_size)
    if sx2 - sx1 < 5 or sy2 - sy1 < 5:
        return frame
    sample_region = frame[sy1:sy2, sx1:sx2]
    sample_hsv = cv2.cvtColor(sample_region, cv2.COLOR_BGR2HSV)
    mean_hue = np.mean(sample_hsv[:, :, 0])
    mean_sat = np.mean(sample_hsv[:, :, 1])
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hue_range = 20
    sat_min = max(30, int(mean_sat * 0.4))
    lower = np.array([max(0, int(mean_hue) - hue_range), sat_min, 40])
    upper = np.array([min(179, int(mean_hue) + hue_range), 255, 255])
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
    color_mask = cv2.GaussianBlur(color_mask, (7, 7), 0)
    _, color_mask = cv2.threshold(color_mask, 127, 255, cv2.THRESH_BINARY)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    color_mask_3ch = cv2.cvtColor(color_mask, cv2.COLOR_GRAY2BGR) / 255.0
    result = (frame.astype(np.float32) * color_mask_3ch +
              gray_bgr.astype(np.float32) * (1 - color_mask_3ch)).astype(np.uint8)
    result_hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV)
    sat_boost = color_mask.astype(np.float32) / 255.0
    result_hsv[:, :, 1] = np.clip(
        result_hsv[:, :, 1].astype(np.float32) * (1.0 + 0.5 * sat_boost), 0, 255
    ).astype(np.uint8)
    result = cv2.cvtColor(result_hsv, cv2.COLOR_HSV2BGR)
    cy_v, cx_v = h // 2, w // 2
    Y, X = np.ogrid[:h, :w]
    max_dist = math.sqrt(cx_v**2 + cy_v**2)
    dist_map = np.sqrt((X.astype(np.float32) - cx_v)**2 + (Y.astype(np.float32) - cy_v)**2)
    vignette = np.clip(1.0 - (dist_map / max_dist) * 0.4 * level, 0.5, 1.0).astype(np.float32)
    for c_ch in range(3):
        result[:, :, c_ch] = (result[:, :, c_ch].astype(np.float32) * vignette).astype(np.uint8)
    return cv2.addWeighted(result, level, frame, 1 - level, 0)
