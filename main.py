import cv2
from cvzone.HandTrackingModule import HandDetector

# ==========================================
#  Gesture Blur Camera - Clean Version
#  1. Pose Peace (✌️) = Blur Seluruh Layar
#  2. Pose L / Kotak (Jempol & Telunjuk 👆) = Blur dalam kotak
# ==========================================

def main():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("ERROR: Kamera tidak bisa dibuka!")
        return

    # Inisialisasi Detektor Tangan dari cvzone
    detector = HandDetector(detectionCon=0.7, maxHands=1)

    print("=" * 45)
    print("  GESTURE BLUR CAMERA (CLEAN VERSION)")
    print("  - Pose Peace Sign (V) : Blur Full Screen")
    print("  - Pose Jempol & Telunjuk : Blur Area Kotak")
    print("  Tekan 'Q' pada keyboard untuk keluar")
    print("=" * 45)

    blur_level = 0.0
    blur_speed = 0.15
    max_blur_kernel = 99

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # Balikkan gambar agar seperti cermin
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        
        # Deteksi tangan (draw=False agar garis hijau kerangka tangan HILANG/Bersih)
        hands, frame = detector.findHands(frame, draw=False)

        peace_detected = False
        box_blur_detected = False
        box_coords = None

        if hands:
            hand = hands[0]
            # Output biner: [Jempol, Telunjuk, Tengah, Manis, Kelingking]
            fingers = detector.fingersUp(hand)

            # ----------------------------------------------------
            # LOGIKA 1: Peace Sign (Telunjuk & Tengah UP)
            # ----------------------------------------------------
            if fingers == [0, 1, 1, 0, 0] or fingers == [1, 1, 1, 0, 0]:
                peace_detected = True

            # ----------------------------------------------------
            # LOGIKA 2: Kotak Jempol & Telunjuk (Hanya dua jari ini UP)
            # ----------------------------------------------------
            elif fingers == [1, 1, 0, 0, 0]:
                box_blur_detected = True
                lmList = hand["lmList"]
                
                # Ambil koordinat ujung jempol (id: 4) dan ujung telunjuk (id: 8)
                x1, y1 = lmList[4][0], lmList[4][1]
                x2, y2 = lmList[8][0], lmList[8][1]
                
                # Buat batasan kotak pembatas (Bounding Box)
                x_min, x_max = max(0, min(x1, x2)), min(w, max(x1, x2))
                y_min, y_max = max(0, min(y1, y2)), min(h, max(y1, y2))
                
                # Pastikan area kotaknya tidak terlalu kecil
                if x_max - x_min > 20 and y_max - y_min > 20:
                    box_coords = (x_min, y_min, x_max, y_max)

        # ====================================================
        # EKSEKUSI EFEK VISUAL
        # ====================================================

        # 1. Terapkan Animasi Blur Full Screen (Jika Peace Sign)
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
        
        # 2. Terapkan Blur Area Kotak (Jika Jempol & Telunjuk)
        # Efek ini menimpa/muncul di atas frame normal
        if box_blur_detected and box_coords:
            x_min, y_min, x_max, y_max = box_coords
            
            # Potong (Crop) area kotak tersebut
            roi = frame[y_min:y_max, x_min:x_max]
            
            # Terapkan blur ekstrem pada area potongan
            roi_blurred = cv2.GaussianBlur(roi, (71, 71), 0)
            
            # Tempelkan kembali area yang sudah diblur ke frame utama
            frame[y_min:y_max, x_min:x_max] = roi_blurred

            # Opsional: Jika kamu ingin melihat garis pinggir kotaknya secara tipis,
            # hapus tanda '#' pada baris di bawah ini.
            # cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 255, 255), 1)

        # Tampilkan hasil akhir murni tanpa UI
        cv2.imshow('Gesture Blur Camera', frame)

        # Tekan 'Q' untuk keluar
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == ord('Q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("\nKamera ditutup. Sampai jumpa!")

if __name__ == "__main__":
    main()