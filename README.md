# Sistem Deteksi Hilal Otomatis

## Ringkasan
Aplikasi web berbasis Streamlit untuk mendeteksi hilal pada citra dan video menggunakan model YOLOv5 (best.pt). Aplikasi ini menyediakan antarmuka interaktif untuk upload gambar atau video, melakukan deteksi objek hilal, dan menampilkan hasil deteksi dengan bounding box. Juga terintegrasi dengan modul hilalpy untuk perhitungan kriteria visibilitas hilal (Yallop dan MABIMS).

## 🚀 Aplikasi Live
Aplikasi ini telah dideploy ke Streamlit Cloud dan dapat diakses di: [https://newhilal.streamlit.app/](https://new-hilal.streamlit.app/)

## Fitur
- **Deteksi Hilal dari Gambar**: Upload gambar (JPG, PNG) dan deteksi objek hilal menggunakan YOLOv5.
- **Deteksi Hilal dari Video**: Upload video (MP4, AVI, MOV), proses frame-by-frame, dan hasilkan video dengan bounding box.
- **Menu Informasi**: Informasi tentang aplikasi, cara penggunaan, struktur folder, dan kontak.
- **Download Hasil**: Unduh gambar atau video hasil deteksi.
- **Statistik Deteksi**: Jumlah objek terdeteksi, detail bounding box, dan rasio deteksi pada video.

## Persyaratan Sistem
- Python 3.8+
- Dependensi: Streamlit, PyTorch, OpenCV, Pillow, NumPy
- Homebrew (opsional, untuk ffmpeg pada macOS/Linux)

## Instalasi
1. **Buat Virtual Environment**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Pada Windows: .venv\Scripts\activate
   ```

2. **Install Dependensi**:
   ```bash
   pip install -r requirements.txt
   ```
   Atau install manual:
   ```bash
   pip install streamlit torch torchvision torchaudio opencv-python pillow numpy
   ```

3. **Install ffmpeg (Direkomendasikan untuk kompatibilitas video)**:
   - macOS/Linux: `brew install ffmpeg`
   - Windows: Download dari situs resmi ffmpeg.

## Penempatan File Penting
- Letakkan model YOLOv5 terlatih di: `models/best.pt`
- Folder `data/` untuk data input tambahan
- Hasil deteksi akan disimpan di folder `results/`

## Menjalankan Aplikasi
```bash
streamlit run app.py
```

## Cara Penggunaan
1. Jalankan aplikasi dengan perintah di atas.
2. Pilih menu "Deteksi Hilal" di sidebar.
3. Pilih mode: "Deteksi Gambar" atau "Deteksi Video".
4. Upload file dan klik tombol deteksi.
5. Lihat hasil dan unduh jika diperlukan.

## Struktur Folder
```
(project root)
├── app.py                 # Aplikasi utama Streamlit
├── requirements.txt       # Dependensi Python
├── packages.txt           # Paket sistem (untuk deployment)
├── README.md              # Dokumentasi ini
├── models/                # Folder model YOLOv5
│   └── best.pt
├── data/                  # Folder data input
├── results/               # Folder hasil deteksi
└── hilalpy/               # Modul perhitungan hilal
    ├── __init__.py
    ├── cond.py
    ├── criteria.py        # Kriteria Yallop dan MABIMS
    ├── divide.py
    ├── equa.py
    ├── multiply.py
    ├── subtract.py
    └── thres.py
```

## Tips & Troubleshooting
- **Model Gagal Dimuat**: Pastikan file `models/best.pt` ada dan PyTorch terinstall dengan benar.
- **OpenCV Error**: Jika terjadi error import OpenCV, pastikan dependensi sistem seperti libGL terinstall. Untuk deployment, tambahkan `Aptfile` dengan paket: libgl1-mesa-glx, libglib2.0-0, libsm6, libxrender1, libxext6.
- **Video Tidak Dapat Diputar**: Jika video hasil tidak dapat diputar di browser, re-encode menggunakan ffmpeg:
  ```bash
  ffmpeg -i input.mp4 -c:v libx264 -preset veryfast -crf 23 -movflags +faststart -pix_fmt yuv420p -an output_h264.mp4
  ```
- **Performa Video**: Gunakan mesin dengan GPU/CUDA untuk pemrosesan video yang lebih cepat.

## Kontak
Untuk pertanyaan atau dukungan, silakan hubungi: kholidnacunk@gmail.com

## Lisensi
Gunakan sesuai kebutuhan; sesuaikan lisensi proyek jika diperlukan.
