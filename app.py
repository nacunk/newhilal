import streamlit as st
import torch
import cv2
import numpy as np
from PIL import Image
import os
import tempfile
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys
from skyfield.api import load, Topos

# Import modul hilalpy
sys.path.append(str(Path(__file__).parent))
from hilalpy import cond, divide, equa, multiply, subtract, thres
from hilalpy.criteria import mabims, yallop

# Konfigurasi halaman
st.set_page_config(
    page_title="Deteksi Hilal Otomatis",
    page_icon="🌙",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #4B5563;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stAlert {
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Inisialisasi direktori
BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"

# Buat direktori jika belum ada
for dir_path in [MODELS_DIR, DATA_DIR, RESULTS_DIR]:
    dir_path.mkdir(exist_ok=True)


ICON_URL = "https://cdn-icons-png.flaticon.com/128/6419/6419754.png"

# Fungsi untuk memuat model YOLOv5
@st.cache_resource
def load_model():
    """Memuat model YOLOv5 yang sudah dilatih"""
    model_path = MODELS_DIR / "best.pt"
    
    if not model_path.exists():
        st.error(f"Model tidak ditemukan di {model_path}")
        st.info("Silakan letakkan file 'best.pt' di folder 'models/'")
        return None
    
    try:
        # Load model menggunakan torch.hub atau langsung dari file
        model = torch.hub.load('ultralytics/yolov5', 'custom', 
                               path=str(model_path), force_reload=False)
        model.conf = 0.25  # confidence threshold
        model.iou = 0.45   # IoU threshold
        return model
    except Exception as e:
        st.error(f"Error memuat model: {str(e)}")
        return None

# Fungsi deteksi hilal
def detect_hilal(image, model):
    """Mendeteksi hilal pada gambar menggunakan YOLOv5"""
    if model is None:
        return None, None, []
    
    # Konversi PIL Image ke numpy array
    img_array = np.array(image)
    
    # Deteksi
    results = model(img_array)
    
    # Ambil deteksi
    detections = results.pandas().xyxy[0]
    
    # Gambar hasil deteksi
    img_result = np.squeeze(results.render())
    
    return img_result, detections, results

# Fungsi perhitungan visibilitas hilal menggunakan hilalpy
def calculate_hilal_visibility(alt, elongation, width):
    """
    Menghitung visibilitas hilal menggunakan kriteria Yallop (implementasi plain Python)

    Parameters:
    - alt: Altitude hilal (derajat)
    - elongation: Elongasi bulan-matahari (derajat)  # masih disediakan untuk kemungkinan pemakaian
    - width: Lebar hilal (arcminute)

    Returns:
    - q_value: Nilai q Yallop (float)
    - visibility_status: Status visibilitas (str)
    """
    try:
        # Hitung komponen kriteria Yallop secara langsung (tanpa hilalpy)
        w = float(width)
        w2 = w ** 2
        w3 = w ** 3

        # threshold = 11.8371 - 6.3226*W + 0.7319*W^2 - 0.1018*W^3
        threshold = 11.8371 - 6.3226 * w + 0.7319 * w2 - 0.1018 * w3

        q_value = (float(alt) - threshold) / 10.0

        # Tentukan status visibilitas berdasarkan nilai q (kriteria Yallop)
        if q_value >= 0.216:
            visibility_status = "Mudah dilihat"
        elif q_value >= -0.014:
            visibility_status = "Dapat dilihat dengan kondisi ideal"
        elif q_value >= -0.160:
            visibility_status = "Memerlukan alat optik"
        elif q_value >= -0.232:
            visibility_status = "Hanya dapat dilihat dengan teleskop"
        else:
            visibility_status = "Tidak dapat dilihat"

        return q_value, visibility_status

    except Exception as e:
        return None, f"Error perhitungan: {str(e)}"

# Fungsi untuk simulasi data historis
def get_historical_data(start='2024-01-01', end='2024-12-31',
                        location_lat=-6.2, location_lon=106.8, local_hour=18):
    """
    Generate data historis hilal menggunakan Skyfield.
    Default lokasi: Jakarta (-6.2, 106.8). Default waktu pengamatan: local_hour (WIB).
    Mengembalikan DataFrame dengan kolom:
    'Tanggal', 'Altitude (°)', 'Elongasi (°)', 'Lebar (arcmin)', 'Terdeteksi'
    """
    try:
        ts = load.timescale()
        planets = load('de421.bsp')  # butuh internet sekali untuk download, akan di-cache
        earth, moon, sun = planets['earth'], planets['moon'], planets['sun']

        dates = pd.date_range(start=start, end=end, freq='29D')
        rows = []
        for d in dates:
            # waktu UTC untuk jam local_hour (mis. WIB = UTC+7)
            # jika lokal jam = 18 (WIB), konversi ke UTC: 18 - timezone
            # di sini asumsikan timezone offset 7 jam (WIB). Jika lain, panggil fungsi dengan jam UTC langsung.
            utc_hour = local_hour - 7
            t = ts.utc(d.year, d.month, d.day, utc_hour, 0, 0)

            observer = earth + Topos(latitude_degrees=location_lat, longitude_degrees=location_lon)
            astrometric_moon = observer.at(t).observe(moon).apparent()
            astrometric_sun = observer.at(t).observe(sun).apparent()

            alt_moon, az_moon, dist_moon = astrometric_moon.altaz()
            alt_sun, az_sun, dist_sun = astrometric_sun.altaz()

            elong = astrometric_moon.separation_from(astrometric_sun).degrees
            phase_angle = elong  # aproksimasi sudut fase
            illumination = (1 + np.cos(np.radians(phase_angle))) / 2 * 100  # %

            # Aproksimasi lebar hilal (arcmin) dari iluminasi: gunakan diameter sudut rata-rata ~30 arcmin
            width_arcmin = max(0.1, (illumination / 100.0) * 30.0)

            # Gunakan fungsi Yallop lokal untuk menentukan apakah terdeteksi (via calculate_hilal_visibility)
            q_value, vis_status = calculate_hilal_visibility(float(alt_moon.degrees), float(elong), width_arcmin)
            if q_value is None:
                detected = "Tidak"
            else:
                # anggap terdeteksi jika status bukan "Tidak dapat dilihat" atau kategori sangat buruk
                detected = "Ya" if "Tidak dapat dilihat" not in vis_status else "Tidak"

            rows.append({
                'Tanggal': pd.Timestamp(d),
                'Altitude (°)': round(float(alt_moon.degrees), 3),
                'Elongasi (°)': round(float(elong), 3),
                'Lebar (arcmin)': round(width_arcmin, 3),
                'Terdeteksi': detected
            })

        return pd.DataFrame(rows)

    except Exception as e:
        # fallback ke data dummy lama jika Skyfield gagal
        dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='29D')
        data = {
            'Tanggal': dates,
            'Altitude (°)': np.random.uniform(3, 15, len(dates)),
            'Elongasi (°)': np.random.uniform(8, 20, len(dates)),
            'Lebar (arcmin)': np.random.uniform(0.5, 2.5, len(dates)),
            'Terdeteksi': np.random.choice(['Ya', 'Tidak'], len(dates), p=[0.7, 0.3])
        }
        st.warning(f"Skyfield gagal menghasilkan data historis: {e}. Menggunakan data dummy sebagai fallback.")
        return pd.DataFrame(data)

# Header aplikasi
st.markdown('<div class="main-header">🌙 Sistem Deteksi Hilal Otomatis</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Menggunakan YOLOv5 dan Analisis Visibilitas HilalPy</div>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image(ICON_URL, width=100)

    st.header("⚙️ Pengaturan")
    
    menu = st.radio(
        "Pilih Menu:",
        ["🔍 Deteksi Hilal", "📊 Data Historis", "ℹ️ Informasi"]
    )
    
    st.markdown("---")
    st.caption("Dikembangkan dengan Streamlit & YOLOv5")

# Menu 1: Deteksi Hilal
if menu == "🔍 Deteksi Hilal":
    st.header("🔍 Deteksi Hilal dari Citra / Video")

    # Load model
    with st.spinner("Memuat model YOLOv5..."):
        model = load_model()

    if model is not None:
        st.success("✅ Model berhasil dimuat!")

    # Pilih mode deteksi: gambar atau video
    mode = st.radio("Pilih mode deteksi:", ["Deteksi Gambar", "Deteksi Video"], horizontal=True)

    if mode == "Deteksi Gambar":
        # ======== UI Upload Gambar (sama seperti sebelumnya) ========
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📤 Upload Citra")
            uploaded_file = st.file_uploader(
                "Pilih gambar hilal (JPG, PNG)",
                type=['jpg', 'jpeg', 'png']
            )

            if uploaded_file is not None:
                image = Image.open(uploaded_file)
                st.image(image, caption="Gambar yang Diupload", use_column_width=True)
            else:
                image = None

        with col2:
            st.subheader("📊 Parameter Visibilitas")
            altitude = st.slider("Altitude Hilal (°)", 0.0, 20.0, 7.0, 0.1)
            elongation = st.slider("Elongasi (°)", 0.0, 30.0, 12.0, 0.1)
            width = st.slider("Lebar Hilal (arcmin)", 0.0, 5.0, 1.5, 0.1)

            # Pilihan kriteria visibilitas
            criteria = st.selectbox(
                "Pilih Kriteria Visibilitas Hilal:",
                ("Yallop", "MABIMS")
            )

            if st.button("🔬 Analisis Visibilitas", type="primary"):
                q_value, visibility = calculate_hilal_visibility(altitude, elongation, width)

                if criteria == "MABIMS":
                    status = mabims(altitude, elongation)
                    st.info(f"Kriteria: MABIMS — Altitude={altitude}°, Elongasi={elongation}°")
                    if "Tidak" in status:
                        st.error(f"❌ {status}")
                    else:
                        st.success(f"✅ {status}")
                else:  # Yallop
                    if q_value is None:
                        st.error(visibility)
                    else:
                        st.metric("Nilai q Yallop", f"{q_value:.3f}")
                        status = yallop(q_value)
                        if "Terlihat" in status and "Mungkin" not in status:
                            st.success(f"✅ {status}")
                        elif "Mungkin" in status:
                            st.info(f"ℹ️ {status}")
                        elif "optik" in status:
                            st.warning(f"⚠️ {status}")
                        else:
                            st.error(f"❌ {status}")

        # Deteksi gambar dengan model
        if image is not None and model is not None:
            if st.button("🚀 Deteksi Hilal (Gambar)", type="primary"):
                with st.spinner("Mendeteksi hilal pada gambar..."):
                    result_img, detections, results = detect_hilal(image, model)

                    if result_img is not None:
                        st.success(f"✅ Deteksi selesai! Ditemukan {len(detections)} objek")

                        # Simpan hasil
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        result_path = RESULTS_DIR / f"output_{timestamp}.jpg"
                        cv2.imwrite(str(result_path), cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR))

                        # Tampilkan hasil
                        col1, col2 = st.columns(2)

                        with col1:
                            st.subheader("Gambar Asli")
                            st.image(image, use_column_width=True)

                        with col2:
                            st.subheader("Hasil Deteksi")
                            st.image(result_img, use_column_width=True)

                        # Tabel deteksi
                        if len(detections) > 0:
                            st.subheader("📋 Detail Deteksi")
                            st.dataframe(detections[['name', 'confidence', 'xmin', 'ymin', 'xmax', 'ymax']], use_container_width=True)

                            # Download hasil
                            with open(result_path, 'rb') as f:
                                st.download_button(
                                    "💾 Download Hasil Deteksi",
                                    f,
                                    file_name=f"hilal_detection_{timestamp}.jpg",
                                    mime="image/jpeg"
                                )

    else:  # mode == "Deteksi Video"
        st.subheader("🎥 Deteksi Hilal dari Video")
        uploaded_video = st.file_uploader("Unggah video (.mp4, .avi, .mov)", type=["mp4", "avi", "mov"])

        if uploaded_video is not None and model is not None:
            process_button = st.button("🚀 Proses Video", type="primary")
            if process_button:
                tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                tfile.write(uploaded_video.read())
                tfile.flush()
                vid_cap = cv2.VideoCapture(tfile.name)

                fps = vid_cap.get(cv2.CAP_PROP_FPS)
                if fps is None or fps <= 0 or np.isnan(fps):
                    fps = 20.0
                width = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = RESULTS_DIR / f"detected_video_{timestamp}.mp4"
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(str(output_path), fourcc, float(fps), (width, height))

                stframe = st.empty()
                frame_count = 0
                detected_frames = 0

                with st.spinner("Memproses video..."):
                    while True:
                        ret, frame = vid_cap.read()
                        if not ret:
                            break
                        frame_count += 1

                        # frame dari OpenCV = BGR -> model expects RGB numpy
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                        # Deteksi menggunakan model (langsung dari array)
                        results = model(frame_rgb)

                        # Annotated frame (render() mengembalikan RGB)
                        annotated = np.squeeze(results.render())

                        # Tulis ke file video (convert ke BGR untuk OpenCV writer)
                        annotated_bgr = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
                        out.write(annotated_bgr)

                        # Hitung jika ada deteksi
                        try:
                            # results.xyxy[0] umumnya berisi tensor / daftar box
                            has_detection = len(results.xyxy[0]) > 0
                        except Exception:
                            # fallback: gunakan pandas result
                            has_detection = len(results.pandas().xyxy[0]) > 0

                        if has_detection:
                            detected_frames += 1

                        # Tampilkan preview terakhir di Streamlit (RGB)
                        stframe.image(annotated, caption=f'Frame {frame_count}', use_column_width=True)

                vid_cap.release()
                out.release()

                st.success("✅ Proses deteksi video selesai!")
                st.video(str(output_path))

                detection_rate = (detected_frames / frame_count) * 100 if frame_count > 0 else 0.0
                st.info(f"Hilal terdeteksi di {detected_frames} dari {frame_count} frame ({detection_rate:.2f}%).")

                # Download hasil video
                with open(output_path, 'rb') as vf:
                    st.download_button(
                        "💾 Download Video Hasil Deteksi",
                        vf,
                        file_name=f"hilal_detection_video_{timestamp}.mp4",
                        mime="video/mp4"
                    )
        elif uploaded_video is not None and model is None:
            st.error("Model belum dimuat. Pastikan model berada di folder models/ dan coba lagi.")

# Menu 2: Data Historis
elif menu == "📊 Data Historis":
    st.header("📊 Data Historis Observasi Hilal")
    
    # Pengaturan input rentang tanggal & lokasi
    with st.expander("⚙️ Pengaturan Rentang & Lokasi", expanded=True):
        col_a, col_b = st.columns(2)
        with col_a:
            start_date = st.date_input("Dari Tanggal", value=datetime(2024, 1, 1).date())
            end_date = st.date_input("Sampai Tanggal", value=datetime(2024, 12, 31).date())
        with col_b:
            latitude = st.number_input("Latitude (°)", value=-6.2, format="%.6f")
            longitude = st.number_input("Longitude (°)", value=106.8, format="%.6f")
            local_hour = st.number_input("Jam Lokal Pengamatan (0-23)", min_value=0, max_value=23, value=18)
        
        generate = st.button("🔁 Generate Data Historis")
    
    # Hasilkan data historis sesuai input (tombol atau otomatis saat panel dibuka)
    try:
        if generate:
            with st.spinner("Menghasilkan data historis..."):
                df_historical = get_historical_data(
                    start=start_date.strftime("%Y-%m-%d"),
                    end=end_date.strftime("%Y-%m-%d"),
                    location_lat=latitude,
                    location_lon=longitude,
                    local_hour=int(local_hour)
                )
        else:
            # default load singkat (gunakan rentang input jika ingin otomatis selalu regenerate)
            df_historical = get_historical_data(
                start=start_date.strftime("%Y-%m-%d"),
                end=end_date.strftime("%Y-%m-%d"),
                location_lat=latitude,
                location_lon=longitude,
                local_hour=int(local_hour)
            )
    except Exception as e:
        st.error(f"Gagal menghasilkan data historis: {e}")
        df_historical = pd.DataFrame(columns=['Tanggal','Altitude (°)','Elongasi (°)','Lebar (arcmin)','Terdeteksi'])
    
    # Pastikan kolom Tanggal bertipe datetime
    if not df_historical.empty:
        df_historical['Tanggal'] = pd.to_datetime(df_historical['Tanggal'])
    
    # Filter tampilan berdasarkan tanggal yang dipilih
    col1, col2 = st.columns(2)
    with col1:
        ui_start = st.date_input("Filter Dari", value=df_historical['Tanggal'].min().date() if not df_historical.empty else start_date)
    with col2:
        ui_end = st.date_input("Filter Sampai", value=df_historical['Tanggal'].max().date() if not df_historical.empty else end_date)
    
    mask = True
    if not df_historical.empty:
        mask = (df_historical['Tanggal'].dt.date >= ui_start) & (df_historical['Tanggal'].dt.date <= ui_end)
    df_filtered = df_historical[mask] if not df_historical.empty else df_historical.copy()
    
    # Statistik
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Observasi", len(df_filtered))
    with col2:
        detected = len(df_filtered[df_filtered['Terdeteksi'] == 'Ya']) if not df_filtered.empty else 0
        st.metric("Terdeteksi", detected)
    with col3:
        avg_alt = df_filtered['Altitude (°)'].mean() if not df_filtered.empty else 0.0
        st.metric("Rata-rata Altitude", f"{avg_alt:.2f}°")
    with col4:
        avg_elong = df_filtered['Elongasi (°)'].mean() if not df_filtered.empty else 0.0
        st.metric("Rata-rata Elongasi", f"{avg_elong:.2f}°")
    
    # Tabel data
    st.subheader("📋 Tabel Data")
    st.dataframe(df_filtered, use_container_width=True)
    
    # Visualisasi
    st.subheader("📈 Visualisasi Data")
    
    tab1, tab2 = st.tabs(["Grafik Altitude", "Distribusi Deteksi"])
    
    with tab1:
        if df_filtered.empty:
            st.info("Tidak ada data untuk diplot.")
        else:
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(df_filtered['Tanggal'], df_filtered['Altitude (°)'], marker='o', linewidth=2)
            ax.set_xlabel('Tanggal')
            ax.set_ylabel('Altitude (°)')
            ax.set_title('Altitude Hilal Sepanjang Waktu')
            ax.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig)
    
    with tab2:
        if df_filtered.empty:
            st.info("Tidak ada data untuk diplot.")
        else:
            fig, ax = plt.subplots(figsize=(6, 4))
            detection_counts = df_filtered['Terdeteksi'].value_counts()
            ax.pie(detection_counts.values, labels=detection_counts.index, autopct='%1.1f%%', startangle=90)
            ax.set_title('Distribusi Status Deteksi')
            st.pyplot(fig)
    
    # Download data
    csv = df_filtered.to_csv(index=False)
    st.download_button(
        "💾 Download Data CSV",
        csv,
        file_name=f"data_historis_hilal_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv"
    )

# Menu 3: Informasi
else:
    st.header("ℹ️ Informasi Aplikasi")
    
    st.markdown("""
    ### 🌙 Tentang Aplikasi
    
    Aplikasi **Deteksi Hilal Otomatis** ini menggunakan teknologi:
    - **YOLOv5**: Model deep learning untuk deteksi objek hilal pada citra
    - **HilalPy**: Library untuk perhitungan visibilitas hilal
    - **Streamlit**: Framework untuk antarmuka web interaktif
    
    ### 📖 Cara Penggunaan
    
    1. **Deteksi Hilal**:
       - Upload gambar hilal atau gunakan sample
       - Masukkan parameter visibilitas (altitude, elongasi, lebar)
       - Klik tombol "Deteksi Hilal" untuk memulai analisis
       - Lihat hasil deteksi dan analisis visibilitas
    
    2. **Data Historis**:
       - Lihat data observasi hilal masa lalu
       - Filter berdasarkan rentang tanggal
       - Analisis statistik dan visualisasi
       - Download data dalam format CSV
    
    ### 🔬 Kriteria Visibilitas Yallop
    
    Nilai q Yallop menentukan kemungkinan visibilitas hilal:
    - **q ≥ +0.216**: Mudah dilihat dengan mata telanjang
    - **-0.014 ≤ q < +0.216**: Dapat dilihat dalam kondisi ideal
    - **-0.160 ≤ q < -0.014**: Memerlukan alat optik
    - **-0.232 ≤ q < -0.160**: Hanya dengan teleskop
    - **q < -0.232**: Tidak dapat dilihat
    
    ### 📁 Struktur Folder
    
    ```
    hilal_detection_app/
    ├── app.py              # Aplikasi utama
    ├── models/             # Model YOLOv5
    │   └── best.pt
    ├── data/               # Data input
    │   └── sample.jpg
    ├── results/            # Hasil deteksi
    └── hilalpy/            # Modul perhitungan
    ```
    
    ### 🚀 Menjalankan Aplikasi
    
    ```bash
    streamlit run app.py
    ```
    
    ### 📞 Kontak & Dukungan
    
    Untuk pertanyaan atau dukungan, silakan hubungi 📨kholidnacunk@gmail.com.
    """)
    
    st.info("💡 **Tips**: Pastikan model YOLOv5 (best.pt) sudah dilatih dengan dataset hilal yang memadai untuk hasil optimal!")

# Footer
st.markdown("---")
st.caption("© 2024 Sistem Deteksi Hilal Otomatis | Powered by YOLOv5 & HilalPy")
