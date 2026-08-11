import cv2

print("Mencari kamera...")
for i in range(4):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f"✅ Kamera BERHASIL ditemukan pada indeks: {i}")
        cap.release()
    else:
        print(f"❌ Kamera GAGAL dibuka pada indeks: {i}")