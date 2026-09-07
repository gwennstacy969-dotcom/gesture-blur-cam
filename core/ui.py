import cv2
import numpy as np
import math
import random
import os
import time

def lerp(a, b, t):
    """Linear interpolation antara a dan b dengan faktor t."""
    return a + (b - a) * t

def draw_hand_landmarks(frame, hand, color=(0, 255, 150)):
    lmList = hand["lmList"]
    connections = [
        (0, 1), (1, 2), (2, 3), (3, 4),
        (0, 5), (5, 6), (6, 7), (7, 8),
        (0, 9), (9, 10), (10, 11), (11, 12),
        (0, 13), (13, 14), (14, 15), (15, 16),
        (0, 17), (17, 18), (18, 19), (19, 20),
        (5, 9), (9, 13), (13, 17),
    ]
    finger_tip_colors = {
        4:  (0, 230, 255),
        8:  (255, 200, 0),
        12: (255, 0, 220),
        16: (0, 255, 100),
        20: (0, 150, 255),
    }
    for c in connections:
        x1, y1 = lmList[c[0]][0], lmList[c[0]][1]
        x2, y2 = lmList[c[1]][0], lmList[c[1]][1]
        glow = (color[0] // 3, color[1] // 3, color[2] // 3)
        cv2.line(frame, (x1, y1), (x2, y2), glow, 5, cv2.LINE_AA)
        cv2.line(frame, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
    for i, lm in enumerate(lmList):
        x, y = lm[0], lm[1]
        if i in finger_tip_colors:
            fc = finger_tip_colors[i]
            cv2.circle(frame, (x, y), 11, (fc[0]//3, fc[1]//3, fc[2]//3), 2, cv2.LINE_AA)
            cv2.circle(frame, (x, y), 7, fc, cv2.FILLED)
            cv2.circle(frame, (x, y), 9, (255, 255, 255), 2, cv2.LINE_AA)
        else:
            cv2.circle(frame, (x, y), 4, color, cv2.FILLED)
            cv2.circle(frame, (x, y), 5, (color[0]//2, color[1]//2, color[2]//2), 1, cv2.LINE_AA)

def generate_heart_points(cx, cy, size, num_points=100):
    points = []
    for i in range(num_points):
        t = 2 * math.pi * i / num_points
        x = 16 * (math.sin(t) ** 3)
        y = -(13 * math.cos(t) - 5 * math.cos(2*t) - 2 * math.cos(3*t) - math.cos(4*t))
        px = int(cx + x * size / 17)
        py = int(cy + y * size / 17)
        points.append((px, py))
    return points

class HeartParticle:
    def __init__(self, x, y, frame_w, frame_h):
        self.x = float(x + random.randint(-40, 40))
        self.y = float(y + random.randint(-20, 20))
        self.vx = random.uniform(-1.5, 1.5)
        self.vy = random.uniform(-3.0, -1.0)
        self.life = 1.0
        self.decay = random.uniform(0.008, 0.02)
        self.size = random.randint(4, 12)
        self.frame_w = frame_w
        self.frame_h = frame_h
        self.color = (random.randint(80, 180), random.randint(50, 120), random.randint(200, 255))
        self.is_sparkle = random.random() < 0.3
    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.vy -= 0.02
        self.vx *= 0.99
        self.life -= self.decay
        self.size = max(1, int(self.size * (0.98 + self.life * 0.01)))
    def is_alive(self):
        return self.life > 0 and 0 <= self.x < self.frame_w and 0 <= self.y < self.frame_h
    def draw(self, frame):
        alpha = max(0.0, min(1.0, self.life))
        ix, iy = int(self.x), int(self.y)
        if self.is_sparkle:
            spark_size = max(1, int(self.size * alpha))
            c = (int(self.color[0] * alpha + 255 * (1 - alpha)),
                 int(self.color[1] * alpha + 255 * (1 - alpha)),
                 int(self.color[2] * alpha))
            cv2.line(frame, (ix - spark_size, iy), (ix + spark_size, iy), c, 1, cv2.LINE_AA)
            cv2.line(frame, (ix, iy - spark_size), (ix, iy + spark_size), c, 1, cv2.LINE_AA)
            d = max(1, spark_size // 2)
            cv2.line(frame, (ix - d, iy - d), (ix + d, iy + d), c, 1, cv2.LINE_AA)
            cv2.line(frame, (ix - d, iy + d), (ix + d, iy - d), c, 1, cv2.LINE_AA)
            cv2.circle(frame, (ix, iy), max(1, spark_size // 3), (255, 255, 255), cv2.FILLED)
        else:
            s = max(2, int(self.size * alpha))
            pts = generate_heart_points(ix, iy, s, num_points=30)
            if len(pts) >= 3:
                pts_array = np.array(pts, dtype=np.int32)
                fill_color = (int(self.color[0] * alpha), int(self.color[1] * alpha), int(self.color[2] * alpha))
                cv2.fillPoly(frame, [pts_array], fill_color)
                outline_color = (min(255, int(fill_color[0] + 60)), min(255, int(fill_color[1] + 60)), min(255, int(fill_color[2] + 30)))
                cv2.polylines(frame, [pts_array], True, outline_color, 1, cv2.LINE_AA)

def draw_heart_vignette(frame, alpha=0.3):
    h, w = frame.shape[:2]
    overlay = np.zeros_like(frame, dtype=np.uint8)
    for i in range(min(80, h // 4)):
        intensity = int(60 * (1 - i / 80) * alpha)
        color = (max(0, intensity // 3), 0, max(0, intensity))
        cv2.line(overlay, (0, i), (w, i), color, 1)
        cv2.line(overlay, (0, h - 1 - i), (w, h - 1 - i), color, 1)
        cv2.line(overlay, (i, 0), (i, h), color, 1)
        cv2.line(overlay, (w - 1 - i, 0), (w - 1 - i, h), color, 1)
    cv2.add(frame, overlay, frame)

class BubbleParticle:
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
        self.vy -= 0.01
        if self.y < -20:
            self.reset()
    def draw(self, frame, level):
        ix, iy = int(self.x), int(self.y)
        if iy < 0 or iy >= self.frame_h or ix < 0 or ix >= self.frame_w:
            return
        a = self.alpha * level
        s = self.size
        overlay = frame.copy()
        bubble_color = (int(200 * a), int(180 * a), int(140 * a))
        cv2.circle(overlay, (ix, iy), s, bubble_color, 1, cv2.LINE_AA)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        hx = ix - max(1, s // 3)
        hy = iy - max(1, s // 3)
        highlight_size = max(1, s // 4)
        highlight_color = (int(255 * a), int(255 * a), int(255 * a))
        cv2.circle(frame, (hx, hy), highlight_size, highlight_color, cv2.FILLED, cv2.LINE_AA)

def draw_anime_frame(frame, x_min, y_min, x_max, y_max, alpha=1.0):
    color_outer = (int(200 * alpha), int(180 * alpha), int(50 * alpha))
    color_inner = (int(255 * alpha), int(255 * alpha), int(100 * alpha))
    corner_color = (int(100 * alpha), int(255 * alpha), int(255 * alpha))
    cv2.rectangle(frame, (x_min - 2, y_min - 2), (x_max + 2, y_max + 2), color_outer, 3, cv2.LINE_AA)
    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color_inner, 1, cv2.LINE_AA)
    corner_len = min(25, (x_max - x_min) // 5, (y_max - y_min) // 5)
    corners = [(x_min, y_min), (x_max, y_min), (x_min, y_max), (x_max, y_max)]
    for i, (cx, cy) in enumerate(corners):
        dx = corner_len if (i % 2 == 0) else -corner_len
        dy = corner_len if (i < 2) else -corner_len
        cv2.line(frame, (cx, cy), (cx + dx, cy), corner_color, 2, cv2.LINE_AA)
        cv2.line(frame, (cx, cy), (cx, cy + dy), corner_color, 2, cv2.LINE_AA)
    label_y = max(y_min - 8, 15)
    cv2.putText(frame, "ANIME", (x_min, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, corner_color, 1, cv2.LINE_AA)

def draw_night_vision_hud(frame, level, blink_on=True):
    if level < 0.3:
        return
    h, w = frame.shape[:2]
    alpha = min(1.0, level)
    color = (0, int(200 * alpha), 0)
    blen = 40
    cv2.line(frame, (15, 15), (15 + blen, 15), color, 2, cv2.LINE_AA)
    cv2.line(frame, (15, 15), (15, 15 + blen), color, 2, cv2.LINE_AA)
    cv2.line(frame, (w - 15, 15), (w - 15 - blen, 15), color, 2, cv2.LINE_AA)
    cv2.line(frame, (w - 15, 15), (w - 15, 15 + blen), color, 2, cv2.LINE_AA)
    cv2.line(frame, (15, h - 15), (15 + blen, h - 15), color, 2, cv2.LINE_AA)
    cv2.line(frame, (15, h - 15), (15, h - 15 - blen), color, 2, cv2.LINE_AA)
    cv2.line(frame, (w - 15, h - 15), (w - 15 - blen, h - 15), color, 2, cv2.LINE_AA)
    cv2.line(frame, (w - 15, h - 15), (w - 15, h - 15 - blen), color, 2, cv2.LINE_AA)
    cx, cy = w // 2, h // 2
    dim = (0, int(100 * alpha), 0)
    for r in [30, 60, 90]:
        cv2.circle(frame, (cx, cy), r, dim, 1, cv2.LINE_AA)
    gap = 15
    cv2.line(frame, (cx - 90, cy), (cx - gap, cy), color, 1, cv2.LINE_AA)
    cv2.line(frame, (cx + gap, cy), (cx + 90, cy), color, 1, cv2.LINE_AA)
    cv2.line(frame, (cx, cy - 90), (cx, cy - gap), color, 1, cv2.LINE_AA)
    cv2.line(frame, (cx, cy + gap), (cx, cy + 90), color, 1, cv2.LINE_AA)
    cv2.circle(frame, (cx, cy), 2, color, cv2.FILLED)
    for d in [30, 60]:
        tick = 5
        cv2.line(frame, (cx - d, cy - tick), (cx - d, cy + tick), dim, 1, cv2.LINE_AA)
        cv2.line(frame, (cx + d, cy - tick), (cx + d, cy + tick), dim, 1, cv2.LINE_AA)
        cv2.line(frame, (cx - tick, cy - d), (cx + tick, cy - d), dim, 1, cv2.LINE_AA)
        cv2.line(frame, (cx - tick, cy + d), (cx + tick, cy + d), dim, 1, cv2.LINE_AA)
    cv2.putText(frame, "NV MODE", (20, h - 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    time_str = time.strftime("%H:%M:%S")
    cv2.putText(frame, time_str, (20, h - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    if blink_on:
        cv2.circle(frame, (30, 30), 5, (0, 0, int(180 * alpha)), cv2.FILLED)
        cv2.putText(frame, "RECORDING", (42, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
    comp_cx, comp_cy = w - 50, 50
    comp_r = 22
    cv2.circle(frame, (comp_cx, comp_cy), comp_r, dim, 1, cv2.LINE_AA)
    cv2.putText(frame, "N", (comp_cx - 4, comp_cy - comp_r - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1, cv2.LINE_AA)
    cv2.putText(frame, "S", (comp_cx - 3, comp_cy + comp_r + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.3, dim, 1, cv2.LINE_AA)
    cv2.putText(frame, "E", (comp_cx + comp_r + 5, comp_cy + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.3, dim, 1, cv2.LINE_AA)
    cv2.putText(frame, "W", (comp_cx - comp_r - 15, comp_cy + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.3, dim, 1, cv2.LINE_AA)
    cv2.line(frame, (comp_cx, comp_cy), (comp_cx, comp_cy - comp_r + 5), color, 2, cv2.LINE_AA)
    cv2.line(frame, (comp_cx, comp_cy), (comp_cx, comp_cy + comp_r - 8), dim, 1, cv2.LINE_AA)
    dist_val = 15.0 + 20.0 * math.sin(time.time() * 0.3)
    cv2.putText(frame, f"DIST: {dist_val:.1f}m", (w - 150, h - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
    elev_val = 1.5 + 0.5 * math.sin(time.time() * 0.5)
    cv2.putText(frame, f"ELEV: {elev_val:.1f}m", (w - 150, h - 45), cv2.FONT_HERSHEY_SIMPLEX, 0.4, dim, 1, cv2.LINE_AA)

def draw_glitch_hud(frame, level, blink_on=True):
    if level < 0.3:
        return
    h, w = frame.shape[:2]
    alpha = min(1.0, level)
    txt_color = (int(200 * alpha), int(200 * alpha), int(200 * alpha))
    if blink_on:
        rec_color = (0, 0, int(255 * alpha))
        cv2.circle(frame, (25, 25), 6, rec_color, cv2.FILLED)
    cv2.putText(frame, "REC", (38, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, txt_color, 1, cv2.LINE_AA)
    cv2.putText(frame, "PLAY >>", (w - 110, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, txt_color, 1, cv2.LINE_AA)
    tracking_y = int((time.time() * 100) % h)
    track_color = (int(100 * alpha), int(100 * alpha), int(100 * alpha))
    cv2.line(frame, (0, tracking_y), (w, tracking_y), track_color, 1)
    time_str = time.strftime("%Y/%m/%d  %H:%M:%S")
    ts_color = (int(200 * alpha), int(200 * alpha), int(80 * alpha))
    cv2.putText(frame, time_str, (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, ts_color, 1, cv2.LINE_AA)
    cv2.putText(frame, "SP", (w - 40, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, txt_color, 1, cv2.LINE_AA)

def draw_spotlight_ring(frame, cx, cy, radius, level):
    if level < 0.3:
        return
    alpha = min(1.0, level)
    color = (int(200 * alpha), int(200 * alpha), int(100 * alpha))
    dim = (int(80 * alpha), int(80 * alpha), int(40 * alpha))
    cv2.circle(frame, (cx, cy), radius, color, 1, cv2.LINE_AA)
    cv2.circle(frame, (cx, cy), radius + 4, dim, 1, cv2.LINE_AA)
    cv2.circle(frame, (cx, cy), max(1, radius - 8), dim, 1, cv2.LINE_AA)
    csize = 10
    gap = 4
    cv2.line(frame, (cx - csize, cy), (cx - gap, cy), color, 1, cv2.LINE_AA)
    cv2.line(frame, (cx + gap, cy), (cx + csize, cy), color, 1, cv2.LINE_AA)
    cv2.line(frame, (cx, cy - csize), (cx, cy - gap), color, 1, cv2.LINE_AA)
    cv2.line(frame, (cx, cy + gap), (cx, cy + csize), color, 1, cv2.LINE_AA)
    for angle_deg in range(0, 360, 45):
        rad = math.radians(angle_deg)
        inner_r = radius - 5
        outer_r = radius + 2
        x1 = int(cx + inner_r * math.cos(rad))
        y1 = int(cy + inner_r * math.sin(rad))
        x2 = int(cx + outer_r * math.cos(rad))
        y2 = int(cy + outer_r * math.sin(rad))
        cv2.line(frame, (x1, y1), (x2, y2), dim, 1, cv2.LINE_AA)
    cv2.putText(frame, "SPOTLIGHT", (cx - 35, cy - radius - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
    coord_str = f"({cx},{cy})"
    cv2.putText(frame, coord_str, (cx - 25, cy + radius + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.3, dim, 1, cv2.LINE_AA)

def draw_invert_border(frame, level, frame_count=0):
    if level < 0.3:
        return
    h, w = frame.shape[:2]
    alpha = min(1.0, level)
    color = (int(180 * alpha), int(120 * alpha), int(255 * alpha))
    cv2.rectangle(frame, (5, 5), (w - 5, h - 5), color, 1, cv2.LINE_AA)
    cv2.rectangle(frame, (8, 8), (w - 8, h - 8), (int(80*alpha), int(60*alpha), int(120*alpha)), 1)
    perf_w, perf_h, perf_gap = 8, 14, 28
    offset = int(frame_count * 2) % perf_gap
    for y in range(-perf_gap + offset, h + perf_gap, perf_gap):
        if 0 <= y < h - perf_h:
            cv2.rectangle(frame, (1, y), (1 + perf_w, y + perf_h), color, 1)
            cv2.rectangle(frame, (3, y + 2), (perf_w - 1, y + perf_h - 2), (int(40*alpha), int(30*alpha), int(60*alpha)), cv2.FILLED)
            cv2.rectangle(frame, (w - 1 - perf_w, y), (w - 1, y + perf_h), color, 1)
            cv2.rectangle(frame, (w - perf_w + 1, y + 2), (w - 3, y + perf_h - 2), (int(40*alpha), int(30*alpha), int(60*alpha)), cv2.FILLED)
    cv2.putText(frame, "NEGATIVE", (w // 2 - 45, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    cv2.putText(frame, f"F:{frame_count:06d}", (w - 100, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)

def draw_thermal_hud(frame, level):
    if level < 0.3:
        return
    h, w = frame.shape[:2]
    alpha = min(1.0, level)
    color = (int(100 * alpha), int(220 * alpha), int(255 * alpha))
    dim = (int(50 * alpha), int(110 * alpha), int(128 * alpha))
    blen = 30
    cv2.line(frame, (10, 10), (10 + blen, 10), color, 2, cv2.LINE_AA)
    cv2.line(frame, (10, 10), (10, 10 + blen), color, 2, cv2.LINE_AA)
    cv2.line(frame, (w - 10, 10), (w - 10 - blen, 10), color, 2, cv2.LINE_AA)
    cv2.line(frame, (w - 10, 10), (w - 10, 10 + blen), color, 2, cv2.LINE_AA)
    cv2.line(frame, (10, h - 10), (10 + blen, h - 10), color, 2, cv2.LINE_AA)
    cv2.line(frame, (10, h - 10), (10, h - 10 - blen), color, 2, cv2.LINE_AA)
    cv2.line(frame, (w - 10, h - 10), (w - 10 - blen, h - 10), color, 2, cv2.LINE_AA)
    cv2.line(frame, (w - 10, h - 10), (w - 10, h - 10 - blen), color, 2, cv2.LINE_AA)
    cx, cy = w // 2, h // 2
    gap = 12
    cv2.line(frame, (cx - 40, cy), (cx - gap, cy), color, 1, cv2.LINE_AA)
    cv2.line(frame, (cx + gap, cy), (cx + 40, cy), color, 1, cv2.LINE_AA)
    cv2.line(frame, (cx, cy - 40), (cx, cy - gap), color, 1, cv2.LINE_AA)
    cv2.line(frame, (cx, cy + gap), (cx, cy + 40), color, 1, cv2.LINE_AA)
    cv2.circle(frame, (cx, cy), 3, color, 1, cv2.LINE_AA)
    fake_temp = 32.0 + 6.0 * math.sin(time.time() * 1.5) + random.uniform(-0.5, 0.5)
    cv2.putText(frame, f"{fake_temp:.1f} C", (cx + 15, cy - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    bar_x, bar_y_start, bar_h, bar_w = w - 35, 60, 160, 15
    for i in range(bar_h):
        ratio = i / bar_h
        val = int(255 * (1 - ratio))
        bar_color_row = cv2.applyColorMap(np.array([[val]], dtype=np.uint8), cv2.COLORMAP_JET)[0][0]
        bar_color_tuple = (int(bar_color_row[0] * alpha), int(bar_color_row[1] * alpha), int(bar_color_row[2] * alpha))
        cv2.line(frame, (bar_x, bar_y_start + i), (bar_x + bar_w, bar_y_start + i), bar_color_tuple, 1)
    cv2.rectangle(frame, (bar_x - 1, bar_y_start - 1), (bar_x + bar_w + 1, bar_y_start + bar_h + 1), dim, 1)
    cv2.putText(frame, "HOT", (bar_x - 5, bar_y_start - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, int(255 * alpha)), 1, cv2.LINE_AA)
    cv2.putText(frame, "COLD", (bar_x - 10, bar_y_start + bar_h + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (int(255 * alpha), 0, 0), 1, cv2.LINE_AA)
    cv2.putText(frame, "THERMAL IMAGING", (15, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    time_str = time.strftime("%H:%M:%S")
    cv2.putText(frame, time_str, (15, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, dim, 1, cv2.LINE_AA)
    max_temp = 38.0 + random.uniform(-0.3, 0.3)
    min_temp = 22.0 + random.uniform(-0.3, 0.3)
    cv2.putText(frame, f"MAX: {max_temp:.1f}C", (15, h - 55), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, int(100 * alpha), int(255 * alpha)), 1, cv2.LINE_AA)
    cv2.putText(frame, f"MIN: {min_temp:.1f}C", (15, h - 38), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (int(255 * alpha), int(100 * alpha), 0), 1, cv2.LINE_AA)

def draw_underwater_hud(frame, level, time_val=0):
    if level < 0.3:
        return
    h, w = frame.shape[:2]
    alpha = min(1.0, level)
    color = (int(200 * alpha), int(180 * alpha), int(80 * alpha))
    dim = (int(100 * alpha), int(90 * alpha), int(40 * alpha))
    cv2.putText(frame, "UNDERWATER MODE", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    depth = 12.5 + 5.0 * math.sin(time_val * 0.3)
    cv2.putText(frame, f"DEPTH: {depth:.1f}m", (15, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
    pressure = 2.2 + 0.5 * math.sin(time_val * 0.2)
    cv2.putText(frame, f"PRESS: {pressure:.1f} atm", (15, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.4, dim, 1, cv2.LINE_AA)
    o2_level = 0.75 + 0.15 * math.sin(time_val * 0.1)
    bar_x, bar_y, bar_w, bar_h = 15, h - 50, 100, 12
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), dim, 1)
    fill_w = int(bar_w * o2_level)
    bar_fill_color = (int(200 * alpha), int(200 * alpha), 0) if o2_level > 0.3 else (0, 0, int(255 * alpha))
    cv2.rectangle(frame, (bar_x + 1, bar_y + 1), (bar_x + fill_w, bar_y + bar_h - 1), bar_fill_color, cv2.FILLED)
    cv2.putText(frame, f"O2: {int(o2_level * 100)}%", (bar_x, bar_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)
    water_temp = 18.0 + 3.0 * math.sin(time_val * 0.15)
    cv2.putText(frame, f"WATER: {water_temp:.1f}C", (15, h - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.35, dim, 1, cv2.LINE_AA)
    bearing = int((time_val * 10) % 360)
    cv2.putText(frame, f"BRG: {bearing:03d}", (w - 110, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)

def draw_color_pop_overlay(frame, level):
    if level < 0.3:
        return
    h, w = frame.shape[:2]
    alpha = min(1.0, level)
    color = (int(100 * alpha), int(200 * alpha), int(255 * alpha))
    cv2.putText(frame, "COLOR POP", (w // 2 - 55, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
    cv2.line(frame, (w // 2 - 55, 32), (w // 2 + 60, 32), color, 1, cv2.LINE_AA)
    accent_len = 20
    cv2.line(frame, (5, 5), (5 + accent_len, 5), color, 2, cv2.LINE_AA)
    cv2.line(frame, (5, 5), (5, 5 + accent_len), color, 2, cv2.LINE_AA)
    cv2.line(frame, (w - 5, h - 5), (w - 5 - accent_len, h - 5), color, 2, cv2.LINE_AA)
    cv2.line(frame, (w - 5, h - 5), (w - 5, h - 5 - accent_len), color, 2, cv2.LINE_AA)

def save_screenshot(frame, captures_dir):
    os.makedirs(captures_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"capture_{timestamp}.png"
    filepath = os.path.join(captures_dir, filename)
    cv2.imwrite(filepath, frame)
    return filepath, filename

def draw_screenshot_thumbnail(frame, thumbnail, display_time, filepath, position="bottom-right"):
    if thumbnail is None:
        return
    h, w = frame.shape[:2]
    th, tw = thumbnail.shape[:2]
    slide_progress = min(1.0, display_time / 0.5)
    offset_x = int((1 - slide_progress) * (tw + 20))
    margin = 15
    tx = w - tw - margin - offset_x
    ty = h - th - margin - 40
    if tx < 0:
        return
    pad = 4
    overlay = frame.copy()
    cv2.rectangle(overlay, (tx - pad, ty - pad), (tx + tw + pad, ty + th + pad + 18), (20, 20, 20), cv2.FILLED)
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
    cv2.rectangle(frame, (tx - pad, ty - pad), (tx + tw + pad, ty + th + pad + 18), (200, 200, 200), 1, cv2.LINE_AA)
    frame[ty:ty+th, tx:tx+tw] = thumbnail
    cv2.putText(frame, "SAVED", (tx, ty + th + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (100, 255, 100), 1, cv2.LINE_AA)

def draw_screenshot_flash(frame, flash_level):
    if flash_level > 0.01:
        white = np.ones_like(frame, dtype=np.uint8) * 255
        return cv2.addWeighted(white, flash_level * 0.8, frame, 1 - flash_level * 0.8, 0)
    return frame

def draw_recording_indicator(frame, is_recording, rec_duration, blink_on=True):
    if not is_recording:
        return
    h, w = frame.shape[:2]
    if blink_on:
        cv2.circle(frame, (w - 25, 25), 8, (0, 0, 255), cv2.FILLED)
    cv2.putText(frame, "REC", (w - 65, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)
    minutes = int(rec_duration) // 60
    seconds = int(rec_duration) % 60
    timer_str = f"{minutes:02d}:{seconds:02d}"
    cv2.putText(frame, timer_str, (w - 65, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.rectangle(frame, (2, 2), (w - 2, h - 2), (0, 0, 180), 2)

def draw_save_notification(frame, display_time, filepath):
    if display_time <= 0:
        return
    h, w = frame.shape[:2]
    alpha = min(1.0, display_time)
    box_w, box_h = 280, 50
    bx, by = (w - box_w) // 2, 10
    overlay = frame.copy()
    cv2.rectangle(overlay, (bx, by), (bx + box_w, by + box_h), (20, 60, 20), cv2.FILLED)
    cv2.addWeighted(overlay, 0.85 * alpha, frame, 1 - 0.85 * alpha, 0, frame)
    cv2.rectangle(frame, (bx, by), (bx + box_w, by + box_h), (0, int(200 * alpha), 0), 1, cv2.LINE_AA)
    color = (0, int(255 * alpha), 0)
    cv2.putText(frame, "VIDEO SAVED!", (bx + 75, by + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
    filename = os.path.basename(filepath) if filepath else ""
    cv2.putText(frame, filename, (bx + 20, by + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (int(150 * alpha), int(150 * alpha), int(150 * alpha)), 1, cv2.LINE_AA)

def draw_fps_counter(frame, fps):
    overlay = frame.copy()
    cv2.rectangle(overlay, (8, 5), (95, 28), (0, 0, 0), cv2.FILLED)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
    fps_color = (0, 255, 0) if fps >= 20 else (0, 200, 255) if fps >= 10 else (0, 0, 255)
    cv2.putText(frame, f"FPS: {int(fps)}", (12, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, fps_color, 1, cv2.LINE_AA)

def draw_active_gesture_label(frame, gesture_name, gesture_color):
    if not gesture_name:
        return
    h, w = frame.shape[:2]
    label = f"[ {gesture_name} ]"
    text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)[0]
    tx = w - text_size[0] - 15
    ty = h - 15
    overlay = frame.copy()
    cv2.rectangle(overlay, (tx - 8, ty - text_size[1] - 8), (tx + text_size[0] + 8, ty + 8), (0, 0, 0), cv2.FILLED)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
    cv2.rectangle(frame, (tx - 8, ty - text_size[1] - 8), (tx + text_size[0] + 8, ty + 8), gesture_color, 1, cv2.LINE_AA)
    cv2.putText(frame, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.55, gesture_color, 2, cv2.LINE_AA)

def draw_info_panel(frame):
    h, w = frame.shape[:2]
    pw, ph = 360, 440
    px = (w - pw) // 2
    py = (h - ph) // 2
    overlay = frame.copy()
    cv2.rectangle(overlay, (px, py), (px + pw, py + ph), (20, 20, 20), cv2.FILLED)
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
    cv2.rectangle(frame, (px, py), (px + pw, py + ph), (100, 200, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "GESTURE GUIDE", (px + 95, py + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (100, 200, 255), 2, cv2.LINE_AA)
    cv2.line(frame, (px + 10, py + 38), (px + pw - 10, py + 38), (80, 80, 80), 1)
    cv2.putText(frame, "[1 TANGAN]", (px + 15, py + 56), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (100, 150, 200), 1, cv2.LINE_AA)
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
        ("I+M+R Fingers", "Color Pop", (100, 200, 255)),
    ]
    y_start = py + 72
    line_h = 20
    for i, (gesture, effect, c) in enumerate(gestures_1h):
        y_pos = y_start + i * line_h
        cv2.putText(frame, gesture, (px + 15, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.37, c, 1, cv2.LINE_AA)
        cv2.putText(frame, effect, (px + 190, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.37, c, 1, cv2.LINE_AA)
    sep_y = y_start + len(gestures_1h) * line_h + 5
    cv2.line(frame, (px + 10, sep_y), (px + pw - 10, sep_y), (80, 80, 80), 1)
    cv2.putText(frame, "[2 TANGAN]", (px + 15, sep_y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (100, 150, 200), 1, cv2.LINE_AA)
    gestures_2h = [
        ("Triangle", "Grayscale", (200, 200, 200)),
        ("Heart", "Love Effect", (100, 80, 255)),
        ("Rectangle", "Anime Filter", (255, 255, 100)),
    ]
    y_start_2h = sep_y + 34
    for i, (gesture, effect, c) in enumerate(gestures_2h):
        y_pos = y_start_2h + i * line_h
        cv2.putText(frame, gesture, (px + 15, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.37, c, 1, cv2.LINE_AA)
        cv2.putText(frame, effect, (px + 190, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.37, c, 1, cv2.LINE_AA)
    sep_y2 = y_start_2h + len(gestures_2h) * line_h + 5
    cv2.line(frame, (px + 10, sep_y2), (px + pw - 10, sep_y2), (80, 80, 80), 1)
    cv2.putText(frame, "[KEYBOARD]", (px + 15, sep_y2 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (100, 150, 200), 1, cv2.LINE_AA)
    keyboard_shortcuts = [
        ("S", "Screenshot", (100, 255, 100)),
        ("R", "Record Video", (100, 100, 255)),
        ("H", "Toggle Help", (150, 150, 150)),
        ("Q", "Quit", (150, 150, 150)),
    ]
    y_start_kb = sep_y2 + 34
    for i, (key, effect, c) in enumerate(keyboard_shortcuts):
        y_pos = y_start_kb + i * line_h
        cv2.putText(frame, f"[{key}]", (px + 15, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.37, c, 1, cv2.LINE_AA)
        cv2.putText(frame, effect, (px + 190, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.37, c, 1, cv2.LINE_AA)
    cv2.putText(frame, "Press H to hide", (px + 110, py + ph - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1, cv2.LINE_AA)

def draw_box_blur_border(frame, x_min, y_min, x_max, y_max):
    cv2.rectangle(frame, (x_min - 2, y_min - 2), (x_max + 2, y_max + 2), (150, 100, 0), 3, cv2.LINE_AA)
    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 200, 50), 1, cv2.LINE_AA)
    corner_len = min(20, (x_max - x_min) // 5, (y_max - y_min) // 5)
    cc = (255, 255, 100)
    for idx, (cx_c, cy_c) in enumerate([(x_min, y_min), (x_max, y_min), (x_min, y_max), (x_max, y_max)]):
        dx = corner_len if (idx % 2 == 0) else -corner_len
        dy = corner_len if (idx < 2) else -corner_len
        cv2.line(frame, (cx_c, cy_c), (cx_c + dx, cy_c), cc, 2, cv2.LINE_AA)
        cv2.line(frame, (cx_c, cy_c), (cx_c, cy_c + dy), cc, 2, cv2.LINE_AA)
    label_y = max(y_min - 8, 15)
    cv2.putText(frame, "BLUR ZONE", (x_min, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, cc, 1, cv2.LINE_AA)

def draw_startup_splash(frame, alpha):
    if alpha <= 0.01:
        return frame
    h, w = frame.shape[:2]
    overlay = np.zeros_like(frame, dtype=np.uint8)
    frame = cv2.addWeighted(overlay, alpha * 0.7, frame, 1 - alpha * 0.7, 0)
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
    cv2.putText(frame, title, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 1.2, t_color, 3, cv2.LINE_AA)
    cv2.putText(frame, sub, (sx, sy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, s_color, 1, cv2.LINE_AA)
    ver = "v3.0 | 16 Gestures | Press H for help"
    v_size = cv2.getTextSize(ver, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)[0]
    vx = (w - v_size[0]) // 2
    v_color = (int(80*title_alpha), int(80*title_alpha), int(80*title_alpha))
    cv2.putText(frame, ver, (vx, h // 2 + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.35, v_color, 1, cv2.LINE_AA)
    return frame
