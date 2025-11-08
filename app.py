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
from pathlib import Path
import sys

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



# Header aplikasi
st.markdown('<div class="main-header">🌙 Sistem Deteksi Hilal Otomatis</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Menggunakan YOLOv5 dan Analisis Visibilitas HilalPy</div>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image(ICON_URL, width=100)
    st.header("⚙️ Pengaturan")
    
    menu = st.radio(
        "Pilih Menu:",
        ["🔍 Deteksi Hilal", "ℹ️ Informasi"]  # HAPUS "📊 Data Historis"
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

# Menu 2: Informasi
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
       - Upload gambar hilal atau video
       - Masukkan parameter visibilitas (altitude, elongasi, lebar)
       - Klik tombol "Deteksi Hilal" untuk memulai analisis
       - Lihat hasil deteksi dan analisis visibilitas
    
    ### 🔬 Kriteria Visibilitas
    
    **Yallop (q-value)**:
    - q ≥ +0.216: Mudah dilihat dengan mata telanjang
    - -0.014 ≤ q < +0.216: Dapat dilihat dalam kondisi ideal
    - -0.160 ≤ q < -0.014: Memerlukan alat optik
    - -0.232 ≤ q < -0.160: Hanya dengan teleskop
    - q < -0.232: Tidak dapat dilihat
    
    **MABIMS**:
    - Altitude ≥ 2° dan Elongasi ≥ 3°
    
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
    
    Untuk pertanyaan atau dukungan, silakan hubungi 📨 kholidnacunk@gmail.com.
    """)