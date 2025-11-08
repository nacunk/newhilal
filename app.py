import streamlit as st
import torch
try:
    import cv2
    CV2_IMPORT_ERROR = None
except Exception as e:
    cv2 = None
    CV2_IMPORT_ERROR = str(e)
import numpy as np
from PIL import Image
import os
import tempfile
from datetime import datetime, timedelta
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys
from skyfield.api import load, Topos
from skyfield import almanac
import pytz

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
        model = torch.hub.load('ultralytics/yolov5', 'custom', 
                               path=str(model_path), force_reload=False)
        model.conf = 0.25
        model.iou = 0.45
        return model
    except Exception as e:
        st.error(f"Error memuat model: {str(e)}")
        return None

# Fungsi deteksi hilal
def detect_hilal(image, model):
    """Mendeteksi hilal pada gambar menggunakan YOLOv5"""
    if model is None:
        return None, None, []
    
    img_array = np.array(image)
    results = model(img_array)
    detections = results.pandas().xyxy[0]
    img_result = np.squeeze(results.render())
    
    return img_result, detections, results

# Fungsi perhitungan visibilitas hilal
def calculate_hilal_visibility(alt, elongation, width):
    """
    Menghitung visibilitas hilal menggunakan kriteria Yallop
    """
    try:
        w = float(width)
        w2 = w ** 2
        w3 = w ** 3
        threshold = 11.8371 - 6.3226 * w + 0.7319 * w2 - 0.1018 * w3
        q_value = (float(alt) - threshold) / 10.0

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

# Fungsi untuk mendapatkan status MABIMS
def get_mabims_status(altitude, elongation):
    """Mendapatkan status visibilitas berdasarkan kriteria MABIMS"""
    try:
        status = mabims(altitude, elongation)
        return status
    except Exception as e:
        # Implementasi manual jika fungsi hilalpy gagal
        if altitude >= 2 and elongation >= 3:
            return "Kriteria MABIMS terpenuhi"
        else:
            return "Kriteria MABIMS tidak terpenuhi"

# Fungsi untuk mendapatkan status Yallop
def get_yallop_status(q_value):
    """Mendapatkan status visibilitas berdasarkan kriteria Yallop"""
    try:
        status = yallop(q_value)
        return status
    except Exception as e:
        # Implementasi manual jika fungsi hilalpy gagal
        if q_value >= 0.216:
            return "Mudah terlihat"
        elif q_value >= -0.014:
            return "Terlihat dalam kondisi ideal"
        elif q_value >= -0.160:
            return "Memerlukan alat optik"
        elif q_value >= -0.232:
            return "Hanya dengan teleskop"
        else:
            return "Tidak dapat dilihat"

# Fungsi untuk simulasi data historis (REVISI UTAMA)
def get_historical_data(start='2024-01-01', end='2024-12-31',
                        location_lat=-6.2, location_lon=106.8, offset_minutes=10):
    """
    Generate data historis hilal menggunakan metode Skyfield sesuai buku
    'Python untuk Astronomi Islam (Kasmui, 2025)'.
    Parameter dihitung pada waktu sunset + offset (default 10 menit).
    Filter hanya tanggal ijtimak ± 1 hari.
    """
    try:
        ts = load.timescale()
        eph = load('de421.bsp')
        earth, moon, sun = eph['earth'], eph['moon'], eph['sun']
        observer = earth + Topos(latitude_degrees=location_lat, longitude_degrees=location_lon)
        
        # Zona waktu lokal
        tz = pytz.timezone('Asia/Jakarta')
        
        # Dapatkan tanggal-tanggal ijtimak (new moon)
        start_date = pd.to_datetime(start)
        end_date = pd.to_datetime(end)
        # Pastikan urutan waktu benar
        if start_date > end_date:
            start_date, end_date = end_date, start_date
            
        t0 = ts.utc(start_date.year, start_date.month, start_date.day)
        t1 = ts.utc(end_date.year, end_date.month, end_date.day, 23, 59, 59)

        
        # Cari fase bulan baru (ijtimak)
        t, y = almanac.find_discrete(t0, t1, almanac.moon_phases(eph))
        
        # Filter hanya new moon (y == 0)
        new_moon_times = [ti for ti, phase in zip(t, y) if phase == 0]
        
        rows = []
        
        # Untuk setiap ijtimak, ambil data ±1 hari
        for ijtimak_time in new_moon_times:
            ijtimak_date = ijtimak_time.utc_datetime().date()
            
            # Range ±1 hari dari ijtimak
            for day_offset in range(-1, 2):  # -1, 0, 1
                check_date = ijtimak_date + timedelta(days=day_offset)
                
                try:
                    # Cari waktu sunset untuk tanggal ini
                    t0_day = ts.utc(check_date.year, check_date.month, check_date.day, 0, 0, 0)
                    t1_day = ts.utc(check_date.year, check_date.month, check_date.day, 23, 59, 59)
                    
                    f = almanac.sunrise_sunset(eph, observer)
                    times, events = almanac.find_discrete(t0, t1, f)
                    
                    sunset_t = None
                    for ti, ev in zip(times, events):
                        #Antisipasi perbedaan label event
                        # Cek altitude mataharri untuk konfirmasi
                        alt_sun, _, _ = observe.at(ti).observe(sun).apparent().altaz()
                        if alt_sun.degrees < 0:
                            sunset_t = ti
                            break
                    if sunset_t is None:
                        # fallback: ambil jam 17:45 lokal sebagai perkiraan sunset
                        dt_guess = datetime.datetime(d.year, d.month, d.day, 17, 45)
                        sunset_t = ts.utc(dt_guess - datetime.datetimedelta(hours=7)) # Jakarta UTC +7
                    
                    # Waktu pengamatan = sunset + offset
                    target_time = ts.utc(sunset_t.utc_datetime() + timedelta(minutes=offset_minutes))
                    
                    # Posisi Bulan dan Matahari
                    astrometric_moon = observer.at(target_time).observe(moon).apparent()
                    astrometric_sun = observer.at(target_time).observe(sun).apparent()
                    
                    alt_moon, az_moon, dist_moon = astrometric_moon.altaz()
                    elong = astrometric_moon.separation_from(astrometric_sun).degrees
                    
                    # Fase bulan (sudut fase)
                    moon_phase_angle = almanac.moon_phase(eph, target_time).degrees
                    
                    # Illumination (%)
                    illumination = (1 - np.cos(np.radians(moon_phase_angle))) / 2 * 100
                    
                    # Lebar hilal (arcmin) - aproksimasi dari illumination
                    # Diameter sudut rata-rata bulan ~30 arcmin
                    width_arcmin = (illumination / 100.0) * 30.0
                    
                    # Hitung kriteria Yallop
                    q_value, vis_status = calculate_hilal_visibility(
                        alt_moon.degrees, elong, width_arcmin
                    )
                    
                    # Status Yallop
                    yallop_status = get_yallop_status(q_value) if q_value is not None else "N/A"
                    
                    # Status MABIMS
                    mabims_status = get_mabims_status(alt_moon.degrees, elong)
                    
                    # Deteksi (berdasarkan q_value)
                    detected = "Ya" if q_value and q_value > -0.232 else "Tidak"
                    
                    rows.append({
                        'Tanggal': pd.Timestamp(check_date),
                        'Ijtimak': pd.Timestamp(ijtimak_date),
                        'Hari ke-': day_offset,
                        'Altitude (°)': round(alt_moon.degrees, 2),
                        'Elongasi (°)': round(elong, 2),
                        'Lebar (arcmin)': round(width_arcmin, 2),
                        'Illumination (%)': round(illumination, 2),
                        'Q-Value': round(q_value, 3) if q_value else None,
                        'Status Yallop': yallop_status,
                        'Status MABIMS': mabims_status,
                        'Terdeteksi': detected
                    })
                    
                except Exception as e:
                    continue
        
        return pd.DataFrame(rows)
    
    except Exception as e:
        st.error(f"Error menghasilkan data historis: {str(e)}")
        # Fallback ke data dummy
        dates = pd.date_range(start=start, end=end, freq='29D')
        data = {
            'Tanggal': dates,
            'Ijtimak': dates,
            'Hari ke-': [0] * len(dates),
            'Altitude (°)': np.random.uniform(3, 15, len(dates)),
            'Elongasi (°)': np.random.uniform(8, 20, len(dates)),
            'Lebar (arcmin)': np.random.uniform(0.5, 2.5, len(dates)),
            'Illumination (%)': np.random.uniform(1, 10, len(dates)),
            'Q-Value': np.random.uniform(-0.5, 0.5, len(dates)),
            'Status Yallop': ['N/A'] * len(dates),
            'Status MABIMS': ['N/A'] * len(dates),
            'Terdeteksi': np.random.choice(['Ya', 'Tidak'], len(dates), p=[0.7, 0.3])
        }
        st.warning(f"Menggunakan data dummy: {e}")
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

    if cv2 is None:
        st.error("OpenCV (cv2) gagal diimpor: libGL atau dependensi sistem mungkin hilang.")
        if CV2_IMPORT_ERROR:
            st.caption(f"Error import: {CV2_IMPORT_ERROR}")
        st.info("Solusi: tambahkan file 'Aptfile' di root repo dengan paket: libgl1-mesa-glx, libglib2.0-0, libsm6, libxrender1, libxext6 lalu redeploy.")
        st.stop()

    with st.spinner("Memuat model YOLOv5..."):
        model = load_model()

    if model is not None:
        st.success("✅ Model berhasil dimuat!")

    mode = st.radio("Pilih mode deteksi:", ["Deteksi Gambar", "Deteksi Video"], horizontal=True)

    if mode == "Deteksi Gambar":
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

            criteria = st.selectbox(
                "Pilih Kriteria Visibilitas Hilal:",
                ("Yallop", "MABIMS")
            )

            if st.button("🔬 Analisis Visibilitas", type="primary"):
                q_value, visibility = calculate_hilal_visibility(altitude, elongation, width)

                if criteria == "MABIMS":
                    status = get_mabims_status(altitude, elongation)
                    st.info(f"Kriteria: MABIMS – Altitude={altitude}°, Elongasi={elongation}°")
                    if "tidak" in status.lower():
                        st.error(f"❌ {status}")
                    else:
                        st.success(f"✅ {status}")
                else:
                    if q_value is None:
                        st.error(visibility)
                    else:
                        st.metric("Nilai q Yallop", f"{q_value:.3f}")
                        status = get_yallop_status(q_value)
                        if "Mudah" in status or "Terlihat" in status:
                            st.success(f"✅ {status}")
                        elif "optik" in status or "teleskop" in status:
                            st.warning(f"⚠️ {status}")
                        else:
                            st.error(f"❌ {status}")

        if image is not None and model is not None:
            if st.button("🚀 Deteksi Hilal (Gambar)", type="primary"):
                with st.spinner("Mendeteksi hilal pada gambar..."):
                    result_img, detections, results = detect_hilal(image, model)

                    if result_img is not None:
                        st.success(f"✅ Deteksi selesai! Ditemukan {len(detections)} objek")

                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        result_path = RESULTS_DIR / f"output_{timestamp}.jpg"
                        cv2.imwrite(str(result_path), cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR))

                        col1, col2 = st.columns(2)

                        with col1:
                            st.subheader("Gambar Asli")
                            st.image(image, use_column_width=True)

                        with col2:
                            st.subheader("Hasil Deteksi")
                            st.image(result_img, use_column_width=True)

                        if len(detections) > 0:
                            st.subheader("📋 Detail Deteksi")
                            st.dataframe(detections[['name', 'confidence', 'xmin', 'ymin', 'xmax', 'ymax']], use_container_width=True)

                            with open(result_path, 'rb') as f:
                                st.download_button(
                                    "💾 Download Hasil Deteksi",
                                    f,
                                    file_name=f"hilal_detection_{timestamp}.jpg",
                                    mime="image/jpeg"
                                )

    else:
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

                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        results = model(frame_rgb)
                        annotated = np.squeeze(results.render())
                        annotated_bgr = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
                        out.write(annotated_bgr)

                        try:
                            has_detection = len(results.xyxy[0]) > 0
                        except Exception:
                            has_detection = len(results.pandas().xyxy[0]) > 0

                        if has_detection:
                            detected_frames += 1

                        stframe.image(annotated, caption=f'Frame {frame_count}', use_column_width=True)

                vid_cap.release()
                out.release()

                st.success("✅ Proses deteksi video selesai!")
                st.video(str(output_path))

                detection_rate = (detected_frames / frame_count) * 100 if frame_count > 0 else 0.0
                st.info(f"Hilal terdeteksi di {detected_frames} dari {frame_count} frame ({detection_rate:.2f}%).")

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
    
    with st.expander("⚙️ Pengaturan Rentang & Lokasi", expanded=True):
        col_a, col_b = st.columns(2)
        with col_a:
            start_date = st.date_input("Dari Tanggal", value=datetime(2024, 1, 1).date())
            end_date = st.date_input("Sampai Tanggal", value=datetime(2024, 12, 31).date())
        with col_b:
            latitude = st.number_input("Latitude (°)", value=-6.2, format="%.6f")
            longitude = st.number_input("Longitude (°)", value=106.8, format="%.6f")
            offset_minutes = st.number_input("Offset dari Sunset (menit)", min_value=0, max_value=60, value=10)
        
        generate = st.button("🔄 Generate Data Historis")
    
    if generate or 'df_historical' not in st.session_state:
        with st.spinner("Menghasilkan data historis..."):
            df_historical = get_historical_data(
                start=start_date.strftime("%Y-%m-%d"),
                end=end_date.strftime("%Y-%m-%d"),
                location_lat=latitude,
                location_lon=longitude,
                offset_minutes=int(offset_minutes)
            )
            st.session_state['df_historical'] = df_historical
    else:
        df_historical = st.session_state['df_historical']
    
    if not df_historical.empty:
        df_historical['Tanggal'] = pd.to_datetime(df_historical['Tanggal'])
    
    col1, col2 = st.columns(2)
    with col1:
        ui_start = st.date_input("Filter Dari", value=df_historical['Tanggal'].min().date() if not df_historical.empty else start_date)
    with col2:
        ui_end = st.date_input("Filter Sampai", value=df_historical['Tanggal'].max().date() if not df_historical.empty else end_date)
    
    mask = True
    if not df_historical.empty:
        mask = (df_historical['Tanggal'].dt.date >= ui_start) & (df_historical['Tanggal'].dt.date <= ui_end)
    df_filtered = df_historical[mask] if not df_historical.empty else df_historical.copy()
    
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
        avg_illum = df_filtered['Illumination (%)'].mean() if not df_filtered.empty and 'Illumination (%)' in df_filtered.columns else 0.0
        st.metric("Rata-rata Illumination", f"{avg_illum:.2f}%")
    
    st.subheader("📋 Tabel Data")
    st.dataframe(df_filtered, use_container_width=True)
    
    st.subheader("📈 Visualisasi Data")
    
    tab1, tab2, tab3 = st.tabs(["Grafik Altitude", "Distribusi Deteksi", "Illumination vs Altitude"])
    
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
    
    with tab3:
        if df_filtered.empty or 'Illumination (%)' not in df_filtered.columns:
            st.info("Tidak ada data untuk diplot.")
        else:
            fig, ax = plt.subplots(figsize=(8, 5))
            scatter = ax.scatter(df_filtered['Illumination (%)'], df_filtered['Altitude (°)'], 
                               c=df_filtered['Terdeteksi'].map({'Ya': 'green', 'Tidak': 'red'}),
                               alpha=0.6, s=100)
            ax.set_xlabel('Illumination (%)')
            ax.set_ylabel('Altitude (°)')
            ax.set_title('Hubungan Illumination vs Altitude')
            ax.grid(True, alpha=0.3)
            ax.legend(['Terdeteksi: Ya', 'Terdeteksi: Tidak'])
            plt.tight_layout()
            st.pyplot(fig)
    
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
    - **Skyfield**: Library astronomi untuk perhitungan akurat posisi bulan
    - **Streamlit**: Framework untuk antarmuka web interaktif
    
    ### 📖 Cara Penggunaan
    
    1. **Deteksi Hilal**:
       - Upload gambar hilal atau video
       - Masukkan parameter visibilitas (altitude, elongasi, lebar)
       - Klik tombol "Deteksi Hilal" untuk memulai analisis
       - Lihat hasil deteksi dan analisis visibilitas
    
    2. **Data Historis**:
       - Generate data observasi hilal berdasarkan ijtimak
       - Data hanya untuk tanggal ±1 hari dari ijtimak
       - Filter berdasarkan rentang tanggal
       - Analisis statistik dan visualisasi
       - Download data dalam format CSV
    
    ### 🔬 Kriteria Visibilitas
    
    **Yallop (q-value)**:
    - q ≥ +0.216: Mudah dilihat dengan mata telanjang
    - -0.014 ≤ q < +0.216: Dapat dilihat dalam kondisi ideal
    - -0.160 ≤ q < -0.014: Memerlukan alat optik
    - -0.232 ≤ q < -0.160: Hanya dengan teleskop
    - q < -0.232: Tidak dapat dilihat
    
    **MABIMS**:
    - Altitude ≥ 2° DAN Elongasi ≥ 3°: Kriteria terpenuhi
    - Selain itu: Kriteria tidak terpenuhi
    
    ### 📊 Fitur Data Historis (Baru)
    
    - **Filter Ijtimak**: Data hanya untuk ±1 hari dari tanggal ijtimak
    - **Illumination (%)**: Persentase pencahayaan bulan
    - **Status Yallop**: Klasifikasi visibilitas berdasarkan q-value
    - **Status MABIMS**: Klasifikasi berdasarkan kriteria MABIMS
    - **Perhitungan Akurat**: Menggunakan Skyfield dengan ephemeris DE421
    - **Waktu Pengamatan**: Sunset + offset (default 10 menit)
    
    ### 🔍 Metodologi
    
    Data historis dihitung menggunakan metode dari buku:
    **"Python untuk Astronomi Islam"** (Kasmui, 2025)
    
    - Posisi bulan dan matahari: Skyfield dengan DE421 ephemeris
    - Waktu pengamatan: Sunset + offset yang dapat disesuaikan
    - Fase bulan: `almanac.moon_phase()`
    - Filter ijtimak: `almanac.moon_phases()` untuk new moon
    - Illumination: Dihitung dari sudut fase bulan
    
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
    
    ### 📦 Dependencies
    
    ```
    streamlit
    torch
    opencv-python-headless
    numpy
    pandas
    matplotlib
    pillow
    skyfield
    pytz
    ```
    
    ### 🆕 Revisi Terbaru
    
    **Versi 2.0** (November 2025):
    - ✅ Filter data hanya untuk tanggal ijtimak ±1 hari
    - ✅ Tambahan kolom Illumination (%)
    - ✅ Tambahan kolom Status Yallop dan MABIMS
    - ✅ Perhitungan waktu sunset yang akurat
    - ✅ Offset waktu pengamatan yang dapat disesuaikan
    - ✅ Visualisasi hubungan Illumination vs Altitude
    - ✅ Metadata tanggal ijtimak dan hari ke-
    
    ### 📞 Kontak & Dukungan
    
    Untuk pertanyaan atau dukungan, silakan hubungi 📧kholidnacunk@gmail.com.
    """)
    
    st.info("💡 **Tips**: Pastikan model YOLOv5 (best.pt) sudah dilatih dengan dataset hilal yang memadai untuk hasil optimal!")
    
    st.success("✨ **Fitur Baru**: Data historis sekarang menggunakan perhitungan astronomi yang lebih akurat dengan filter ijtimak!")

# Footer
st.markdown("---")
st.caption("© 2024 Sistem Deteksi Hilal Otomatis | Powered by YOLOv5, HilalPy & Skyfield")
st.caption("Metodologi: Python untuk Astronomi Islam (Kasmui, 2025)")