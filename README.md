# Sistem Deteksi Hilal Otomatis

Ringkasan
Aplikasi web berbasis Streamlit untuk mendeteksi hilal pada citra dan video menggunakan model YOLOv5 (best.pt) serta analisis visibilitas sederhana (kriteria Yallop dan MABIMS). Juga menyediakan generator data historis menggunakan Skyfield.

## 🚀 Aplikasi Live

Aplikasi ini telah dideploy ke Streamlit Cloud dan dapat diakses di: [https://new-hilal.streamlit.app/](https://new-hilal.streamlit.app/)

Fitur
- Deteksi objek (hilal) pada gambar (upload / sample).
- Deteksi objek pada video (upload video → proses frame-by-frame → keluaran video dengan bounding box).
- Pilihan kriteria visibilitas: Yallop dan MABIMS.
- Statistik deteksi (jumlah frame terdeteksi, rasio).
- Generator data historis hilal (Skyfield).
- Download hasil (gambar / video / CSV).

Persyaratan (macOS/Linux)
- Python 3.8+
- Homebrew (opsional, untuk ffmpeg)

Instalasi singkat
1. Buat virtual environment:
   python -m venv .venv
   source .venv/bin/activate
2. Install dependensi utama:
   pip install streamlit torch torchvision torchaudio opencv-python pillow numpy pandas matplotlib skyfield
3. (Direkomendasikan) Install ffmpeg untuk kompatibilitas video:
   brew install ffmpeg

Penempatan file penting
- Letakkan model YOLOv5 terlatih di: `models/best.pt`
- Sample gambar: `data/sample.jpg`
- Hasil akan disimpan di folder `results/`

Menjalankan
streamlit run app.py

Catatan terkait deteksi video & pemutaran di browser
- Jika video hasil proses tidak dapat diputar di preview browser, penyebab umum:
  - codec yang digunakan OpenCV (mp4v) mungkin tidak kompatibel atau moov atom berada di akhir file.
- Solusi praktis:
  - Re-encode ke H.264 dengan moov atom di depan:
    ffmpeg -i input.mp4 -c:v libx264 -preset veryfast -crf 23 -movflags +faststart -pix_fmt yuv420p -an output_h264.mp4
  - Pastikan `out.release()` sudah dipanggil sebelum membuka file untuk preview/download.
- Disarankan menginstal ffmpeg agar hasil video dapat dipratinjau di Streamlit.

Struktur folder
```
/ (project root)
├─ app.py
├─ models/
│  └─ best.pt
├─ data/
│  └─ sample.jpg
├─ results/
└─ hilalpy/
   └─ criteria.py
```

Tips & troubleshooting singkat
- Jika model gagal dimuat: pastikan `models/best.pt` ada dan PyTorch kompatibel.
- Jika Skyfield mau unduh ephemeris pertama kali, diperlukan koneksi internet.
- Untuk performa video, gunakan mesin dengan GPU / CUDA saat memproses banyak frame.

Kontak
kholidnacunk@gmail.com

Lisensi
Gunakan sesuai kebutuhan; sesuaikan lisensi proyek jika
