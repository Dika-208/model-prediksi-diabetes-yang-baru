import os
from collections import Counter
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE
from sklearn.pipeline import Pipeline

# KONFIGURASI HALAMAN
st.set_page_config(
    page_title="Sistem Deteksi Diabetes - ML",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Injeksi Google Material Symbols dan Custom Styling yang Modern & Bersih
st.markdown("""
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:opsz,wght,FILL,GRAD@20..48,100..700,0..1,-50..200" />
<style>
    /* Styling Google Icons */
    .g-icon {
        font-family: 'Material Symbols Outlined';
        font-weight: normal;
        font-style: normal;
        font-size: 20px;
        line-height: 1;
        letter-spacing: normal;
        text-transform: none;
        display: inline-block;
        white-space: nowrap;
        word-wrap: normal;
        direction: ltr;
        vertical-align: middle;
        margin-right: 6px;
    }
    
    /* Header Utama */
    .app-header {
        background: linear-gradient(135deg, #1E3A8A 0%, #2563EB 50%, #3B82F6 100%);
        padding: 22px 28px;
        border-radius: 12px;
        color: white;
        margin-bottom: 25px;
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.2);
    }
    .app-header h1 {
        color: #FFFFFF !important;
        font-size: 26px;
        font-weight: 700;
        margin: 0;
        display: flex;
        align-items: center;
    }
    .app-header p {
        color: #E0E7FF !important;
        font-size: 14px;
        margin: 6px 0 0 0;
    }

    /* Kartu Informasi & Notifikasi */
    .custom-card {
        padding: 14px 18px;
        border-radius: 10px;
        margin-bottom: 15px;
        font-size: 14px;
        display: flex;
        align-items: center;
        border-left: 5px solid;
    }
    .card-success {
        background-color: #ECFDF5;
        border-color: #10B981;
        color: #065F46;
    }
    .card-warning {
        background-color: #FFFBEB;
        border-color: #F59E0B;
        color: #92400E;
    }
    .card-danger {
        background-color: #FEF2F2;
        border-color: #EF4444;
        color: #991B1B;
    }
    .card-info {
        background-color: #EFF6FF;
        border-color: #3B82F6;
        color: #1E3A8A;
    }
    
    /* Kartu Hasil Diagnosis */
    .result-box {
        padding: 20px;
        border-radius: 12px;
        margin-top: 15px;
        text-align: center;
        border: 2px solid;
    }
    .result-positive {
        background-color: #FEF2F2;
        border-color: #F87171;
        color: #991B1B;
    }
    .result-negative {
        background-color: #ECFDF5;
        border-color: #34D399;
        color: #065F46;
    }
    
    /* Rekomendasi Box */
    .advice-box {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 16px;
        margin-top: 15px;
    }
</style>
""", unsafe_allow_html=True)

def g_icon(name, color="#2563EB", size="20px"):
    """Helper untuk memanggil Google Material Symbol secara dinamis"""
    return f'<span class="g-icon" style="color: {color}; font-size: {size};">{name}</span>'

def show_card(card_type, icon_name, icon_color, message):
    """Menampilkan notifikasi dengan Google Icon"""
    st.markdown(f"""
    <div class="custom-card card-{card_type}">
        {g_icon(icon_name, icon_color, "24px")}
        <span>{message}</span>
    </div>
    """, unsafe_allow_html=True)

# DEFINISI FITUR & PEMUATAN DATA
features = [
    'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
    'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
]

DB_FILE = 'diabetes.csv'
MODEL_PATH = 'models/diabetes_model.pkl'
os.makedirs('models', exist_ok=True)

@st.cache_data
def load_data():
    """Membaca data primer dari dataset teks/csv yang disediakan"""
    if os.path.exists('Dataset_Repeel.txt'):
        df = pd.read_csv('Dataset_Repeel.txt', delimiter=';')
    elif os.path.exists('Dataset_Repeel.csv'):
        df = pd.read_csv('Dataset_Repeel.csv', delimiter=';')
    else:
        raise FileNotFoundError("File Dataset_Repeel tidak ditemukan.")
    
    df.columns = df.columns.str.strip()
    df = df.dropna(subset=['Outcome'])
    df['Outcome'] = df['Outcome'].astype(int)
    return df

try:
    df_raw = load_data()
    # Inisialisasi database lokal diabetes.csv jika belum pernah dibuat
    if not os.path.exists(DB_FILE):
        df_raw.to_csv(DB_FILE, index=False, sep=';')
except Exception as e:
    show_card("danger", "error", "#EF4444", f"Error saat memuat dataset: {e}")
    st.stop()

# ==========================================
# 3. SIDEBAR: STATUS SISTEM & INFORMASI
# ==========================================
with st.sidebar:
    st.markdown(f"### {g_icon('health_and_safety', '#2563EB', '24px')} Status Sistem", unsafe_allow_html=True)
    
    # Cek status model pkl
    model_tersedia = os.path.exists(MODEL_PATH)
    if model_tersedia:
        st.markdown(f"{g_icon('check_circle', '#10B981')} **Model ML:** Siap Digunakan", unsafe_allow_html=True)
    else:
        st.markdown(f"{g_icon('warning', '#F59E0B')} **Model ML:** Belum Dilatih", unsafe_allow_html=True)

    # Cek status dataset
    st.markdown(f"{g_icon('database', '#3B82F6')} **Dataset:** {len(df_raw)} Baris Data", unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown(f"### {g_icon('info', '#64748B', '22px')} Informasi Parameter Medis", unsafe_allow_html=True)
    st.caption("Penjelasan fitur untuk standar diagnosis:")
    st.markdown("""
    - **Pregnancies**: Frekuensi kehamilan
    - **Glucose**: Glukosa plasma 2 jam (Normal: < 140 mg/dL)
    - **BloodPressure**: Tekanan darah diastolik (Normal: 60 - 80 mm Hg)
    - **SkinThickness**: Ketebalan lipatan kulit trisep (mm)
    - **Insulin**: Kadar serum insulin 2 jam (mu U/ml)
    - **BMI**: Indeks Massa Tubuh (kg/m²)
    - **Pedigree**: Riwayat faktor genetik diabetes
    - **Age**: Usia pasien (Tahun)
    """)
    st.markdown("---")
    st.caption("Sistem Berstandar Lomba Kejuruan (SMK) • Python 3 & Scikit-Learn")

# Header Utama Aplikasi
st.markdown(f"""
<div class="app-header">
    <h1>{g_icon('medical_services', '#FFFFFF', '32px')} Sistem Deteksi Dini & Analisis Prediksi Diabetes Pasien</h1>
    <p>Implementasi Machine Learning Random Forest & SMOTE untuk Klasifikasi Risiko Kesehatan Pasien</p>
</div>
""", unsafe_allow_html=True)

# Muat Model
model = None
if model_tersedia:
    try:
        model = joblib.load(MODEL_PATH)
    except Exception as e:
        show_card("warning", "warning", "#F59E0B", f"Gagal membaca file model: {e}")

#NAVIGASI UTAMA
tab_dashboard, tab_prediksi, tab_cari, tab_training, tab_unduh = st.tabs([
    "Dashboard & Analisis",
    "Prediksi Pasien Baru",
    "Pencarian & Riwayat",
    "Pelatihan Model & MLflow",
    "Unduh Dataset"
])

#DASHBOARD & ANALISIS DATASET LENGKAP
with tab_dashboard:
    st.markdown(f"### {g_icon('analytics', '#2563EB', '26px')} Ringkasan Dataset & Analisis", unsafe_allow_html=True)
    
    df = df_raw.copy()
    X = df[features].fillna(df[features].mean())
    y = df['Outcome']
    
    # Ringkasan KPI Cards
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    total_pasien = len(df)
    total_positif = int((y == 1).sum())
    total_negatif = int((y == 0).sum())
    avg_usia = df['Age'].mean()
    
    with kpi1:
        st.metric("Total Sampel Pasien", f"{total_pasien} Data")
    with kpi2:
        st.metric("Pasien Diabetes (Aktual)", f"{total_positif} Orang", f"{(total_positif/total_pasien)*100:.1f}%")
    with kpi3:
        st.metric("Bebas Diabetes (Aktual)", f"{total_negatif} Orang", f"{(total_negatif/total_pasien)*100:.1f}%")
    with kpi4:
        st.metric("Rata-Rata Usia Pasien", f"{avg_usia:.1f} Tahun")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    subtab_data, subtab_chart = st.tabs(["Tabel Dataset Lengkap", "Statistik & Visualisasi Grafik"])
    
    with subtab_data:
        st.dataframe(df, use_container_width=True)
        
    with subtab_chart:
        if model is not None:
            # Prediksi untuk seluruh dataset
            y_pred_full = model.predict(X)
            y_prob_full = model.predict_proba(X)
            
            df['Status_Sekarang'] = df['Outcome'].map({1: 'Memiliki Diabetes', 0: 'Tidak Memiliki Diabetes'})
            df['Prediksi_Model'] = np.where(y_pred_full == 1, 'Akan Memiliki Diabetes', 'Tidak Akan Memiliki Diabetes')
            df['Probabilitas_Diabetes'] = y_prob_full[:, 1]
            df['Probabilitas_Tidak_Diabetes'] = y_prob_full[:, 0]
            
            col_chart1, col_chart2 = st.columns(2)
            
            with col_chart1:
                st.markdown(f"#### {g_icon('bar_chart', '#2563EB')} Distribusi Status", unsafe_allow_html=True)
                
                st.write("**Distribusi Status Aktual:**")
                status_counts = df['Status_Sekarang'].value_counts()
                st.dataframe(status_counts, use_container_width=True)
                
                st.write("**Distribusi Hasil Prediksi Model:**")
                pred_counts = df['Prediksi_Model'].value_counts()
                st.dataframe(pred_counts, use_container_width=True)
                
                df['Prediksi_Benar'] = (df['Status_Sekarang'] == 'Memiliki Diabetes') == (
                        df['Prediksi_Model'] == 'Akan Memiliki Diabetes')
                accuracy = df['Prediksi_Benar'].mean()
                st.metric("Akurasi Prediksi vs Data Aktual", f"{accuracy * 100:.2f}%")
            
            with col_chart2:
                st.markdown(f"#### {g_icon('pie_chart', '#2563EB')} Perbandingan Diagram Lingkaran", unsafe_allow_html=True)
                fig, axes = plt.subplots(1, 2, figsize=(8, 4))
                
                status_counts.plot.pie(
                    ax=axes[0],
                    labels=status_counts.index,
                    autopct='%1.1f%%',
                    colors=['#93C5FD', '#FCA5A5'],
                    textprops={'fontsize': 8},
                    startangle=90
                )
                axes[0].set_title('Status Diabetes Aktual', fontsize=10, fontweight='bold')
                axes[0].set_ylabel("")
                
                pred_counts.plot.pie(
                    ax=axes[1],
                    labels=pred_counts.index,
                    autopct='%1.1f%%',
                    colors=['#93C5FD', '#FCA5A5'],
                    textprops={'fontsize': 8},
                    startangle=90
                )
                axes[1].set_title('Prediksi Model', fontsize=10, fontweight='bold')
                axes[1].set_ylabel("")
                
                plt.tight_layout()
                st.pyplot(fig)
        else:
            show_card("warning", "warning", "#F59E0B", "Visualisasi evaluasi prediksi memerlukan model. Silakan latih model terlebih dahulu di tab Pelatihan Model.")

# PREDIKSI PASIEN BARU
with tab_prediksi:
    st.markdown(f"### {g_icon('person_add', '#2563EB', '26px')} Form Skrining Pasien Baru", unsafe_allow_html=True)
    
    if model is None:
        show_card("warning", "warning", "#F59E0B", "Model Machine Learning belum dilatih. Silakan buka tab 'Pelatihan Model & MLflow' dan klik Train Model terlebih dahulu.")
    else:
        with st.form("input_pasien_form"):
            st.markdown(f"##### {g_icon('badge', '#64748B')} Identitas Pasien", unsafe_allow_html=True)
            col_id, col_info = st.columns([1, 2])
            with col_id:
                pasien_id = st.text_input("Nomor / ID Pasien", placeholder="Contoh: PASIEN-001")
            with col_info:
                st.caption("Masukkan ID unik untuk rekam medis pasien agar dapat disimpan dan dicari kembali di kemudian hari.")
                
            st.markdown(f"##### {g_icon('vital_signs', '#2563EB')} Parameter Medis & Laboratorium", unsafe_allow_html=True)
            cols = st.columns(3)
            
            # Input angka yang terstandar dengan batasan wajar medis & petunjuk yang jelas
            with cols[0]:
                val_preg = st.number_input("Pregnancies (Jumlah Kehamilan)", min_value=0, max_value=20, value=1, step=1, help="Frekuensi kehamilan yang pernah dialami pasien wanita.")
                val_skin = st.number_input("SkinThickness (Tebal Kulit Trisep - mm)", min_value=0.0, max_value=99.0, value=20.0, step=1.0, help="Ketebalan lipatan kulit trisep lengan (mm).")
                val_pedigree = st.number_input("DiabetesPedigreeFunction (Faktor Genetik)", min_value=0.05, max_value=2.5, value=0.35, step=0.01, help="Skor riwayat silsilah diabetes pada keluarga.")
                
            with cols[1]:
                val_glucose = st.number_input("Glucose (Kadar Glukosa - mg/dL)", min_value=0.0, max_value=300.0, value=120.0, step=1.0, help="Glukosa plasma 2 jam tes toleransi glukosa oral (Normal: < 140 mg/dL).")
                val_insulin = st.number_input("Insulin (Serum Insulin - mu U/ml)", min_value=0.0, max_value=900.0, value=80.0, step=1.0, help="Kadar insulin serum 2 jam.")
                val_age = st.number_input("Age (Usia Pasien - Tahun)", min_value=1, max_value=120, value=30, step=1, help="Usia lengkap pasien.")
                
            with cols[2]:
                val_bp = st.number_input("BloodPressure (Tekanan Diastolik - mm Hg)", min_value=0.0, max_value=180.0, value=70.0, step=1.0, help="Tekanan darah diastolik (Normal: 60 - 80 mm Hg).")
                val_bmi = st.number_input("BMI (Indeks Massa Tubuh - kg/m²)", min_value=10.0, max_value=70.0, value=25.0, step=0.1, help="Berat badan (kg) dibagi tinggi kuadrat (m²). Kategori normal: 18.5 - 24.9.")
            
            st.markdown("<br>", unsafe_allow_html=True)
            submitted = st.form_submit_button("Lakukan Prediksi & Simpan Hasil")
            
        if submitted:
            if pasien_id.strip() == "":
                show_card("warning", "warning", "#F59E0B", "Harap masukkan Nomor/ID Pasien terlebih dahulu.")
            else:
                input_data = {
                    'Pregnancies': float(val_preg),
                    'Glucose': float(val_glucose),
                    'BloodPressure': float(val_bp),
                    'SkinThickness': float(val_skin),
                    'Insulin': float(val_insulin),
                    'BMI': float(val_bmi),
                    'DiabetesPedigreeFunction': float(val_pedigree),
                    'Age': float(val_age)
                }
                
                try:
                    df_new = pd.DataFrame([input_data])
                    
                    # Prediksi model (Pipeline scikit-learn secara otomatis menstandarkan data)
                    hasil = model.predict(df_new)
                    prob = model.predict_proba(df_new)
                    
                    prob_negatif = prob[0][0] * 100
                    prob_positif = prob[0][1] * 100
                    is_diabetes = (hasil[0] == 1)
                    
                    st.markdown("---")
                    st.markdown(f"#### {g_icon('clinical_notes', '#2563EB', '26px')} Hasil Diagnosis Sistem", unsafe_allow_html=True)
                    
                    # Tampilan Hasil Visual
                    if is_diabetes:
                        st.markdown(f"""
                        <div class="result-box result-positive">
                            <h2 style="margin: 0; color: #991B1B; display: flex; justify-content: center; align-items: center;">
                                {g_icon('warning', '#DC2626', '30px')} TERINDIKASI RISIKO DIABETES
                            </h2>
                            <p style="margin: 8px 0 0 0; font-size: 15px;">
                                Pasien dengan ID <b>{pasien_id}</b> memiliki probabilitas tinggi mengalami diabetes.
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="result-box result-negative">
                            <h2 style="margin: 0; color: #065F46; display: flex; justify-content: center; align-items: center;">
                                {g_icon('check_circle', '#059669', '30px')} TIDAK TERINDIKASI DIABETES
                            </h2>
                            <p style="margin: 8px 0 0 0; font-size: 15px;">
                                Pasien dengan ID <b>{pasien_id}</b> diprediksi dalam kategori sehat / risiko rendah.
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                    st.markdown("<br>", unsafe_allow_html=True)
                    c_res1, c_res2, c_res3 = st.columns(3)
                    with c_res1:
                        st.metric("ID Pasien Terdaftar", str(pasien_id))
                    with c_res2:
                        st.metric("Probabilitas Bebas Diabetes", f"{prob_negatif:.1f}%")
                    with c_res3:
                        st.metric("Probabilitas Mengidap Diabetes", f"{prob_positif:.1f}%")
                        
                    # Fitur Tambahan Edukatif: Rekomendasi Klinis / Gaya Hidup
                    st.markdown(f"""
                    <div class="advice-box">
                        <h4 style="margin: 0 0 10px 0; color: #1E293B; display: flex; align-items: center;">
                            {g_icon('tips_and_updates', '#D97706', '22px')} Rekomendasi & Catatan Medis Berdasarkan Parameter
                        </h4>
                        <ul style="margin: 0; padding-left: 20px; font-size: 14px; color: #334155; line-height: 1.6;">
                            <li><b>Kadar Glukosa ({val_glucose} mg/dL):</b> {'⚠️ Glukosa berada di atas ambang batas normal (>= 140 mg/dL). Sangat disarankan tes lanjutan HbA1c dan konsultasi ke dokter.' if val_glucose >= 140 else '✅ Glukosa dalam batas normal.'}</li>
                            <li><b>Indeks Massa Tubuh / BMI ({val_bmi}):</b> {'⚠️ Masuk kategori obesitas (BMI >= 30). Disarankan mengatur asupan kalori dan meningkatkan aktivitas fisik harian.' if val_bmi >= 30 else ('⚠️ Masuk kategori kelebihan berat badan (overweight).' if val_bmi >= 25 else '✅ BMI dalam batas proporsional.')}</li>
                            <li><b>Tekanan Darah ({val_bp} mm Hg):</b> {'⚠️ Tekanan darah diastolik cenderung tinggi (>= 90 mm Hg). Batasi asupan garam dan pantau tensi secara berkala.' if val_bp >= 90 else '✅ Tekanan darah dalam rentang normal.'}</li>
                            <li><b>Rekomendasi Gaya Hidup:</b> Lakukan olahraga kardio ringan minimal 30 menit per hari, cukupi hidrasi air putih, dan kurangi makanan berpemanis buatan.</li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Simpan data ke diabetes.csv
                    input_data['id'] = str(pasien_id).strip()
                    input_data['Outcome'] = int(hasil[0])
                    input_data['Prediksi_Model'] = "Akan Memiliki Diabetes" if is_diabetes else "Tidak Akan Memiliki Diabetes"
                    input_data['Probabilitas_Tidak_Diabetes'] = round(float(prob[0][0]), 4)
                    input_data['Probabilitas_Diabetes'] = round(float(prob[0][1]), 4)
                    
                    df_append = pd.DataFrame([input_data])
                    
                    try:
                        if os.path.exists(DB_FILE):
                            df_existing = pd.read_csv(DB_FILE, delimiter=';')
                            df_combined = pd.concat([df_existing, df_append], ignore_index=True)
                        else:
                            df_combined = df_append
                            
                        df_combined.to_csv(DB_FILE, index=False, sep=';')
                        show_card("success", "save", "#10B981", f"Data pasien dengan ID <b>{pasien_id}</b> berhasil disimpan ke database <code>{DB_FILE}</code>.")
                    except Exception as err:
                        show_card("danger", "error", "#EF4444", f"Gagal menyimpan data ke file: {err}")
                        
                except Exception as ex:
                    show_card("danger", "error", "#EF4444", f"Terjadi kesalahan saat memproses prediksi: {ex}")

#PENCARIAN & RIWAYAT PASIEN
with tab_cari:
    st.markdown(f"### {g_icon('search', '#2563EB', '26px')} Cari Rekam Medis Pasien", unsafe_allow_html=True)
    
    col_s1, col_s2 = st.columns([3, 1])
    with col_s1:
        search_id = st.text_input("Ketikkan ID Pasien yang Ingin Dicari", placeholder="Masukkan ID pasien...")
    with col_s2:
        st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
        btn_cari = st.button("Cari Data Pasien")
        
    if btn_cari:
        if search_id.strip() == "":
            show_card("warning", "warning", "#F59E0B", "Harap masukkan ID pasien terlebih dahulu.")
        else:
            try:
                if not os.path.exists(DB_FILE):
                    show_card("warning", "warning", "#F59E0B", "Database pasien belum dibuat. Silakan lakukan skrining pasien baru terlebih dahulu.")
                else:
                    df_check = pd.read_csv(DB_FILE, delimiter=';')
                    result = df_check[df_check['id'].astype(str) == search_id.strip()]
                    
                    if not result.empty:
                        result_display = result.copy()
                        for col in result_display.columns:
                            if result_display[col].dtype == float:
                                result_display[col] = result_display[col].round(2)
                                
                        st.markdown(f"##### {g_icon('folder_shared', '#10B981')} Data Ditemukan untuk ID: <b>{search_id}</b>", unsafe_allow_html=True)
                        st.dataframe(result_display, use_container_width=True)
                        
                        prediksi = result_display.iloc[0].get('Prediksi_Model', 'Tidak Diketahui')
                        prob_diab = float(result_display.iloc[0].get('Probabilitas_Diabetes', 0.0)) * 100
                        
                        if prediksi == "Akan Memiliki Diabetes":
                            show_card("danger", "warning", "#EF4444", f"Pasien ini berstatus: <b>{prediksi}</b> (Tingkat Risiko: {prob_diab:.1f}%)")
                        else:
                            show_card("success", "check_circle", "#10B981", f"Pasien ini berstatus: <b>{prediksi}</b> (Tingkat Risiko: {prob_diab:.1f}%)")
                    else:
                        show_card("danger", "person_search", "#EF4444", f"Data pasien dengan ID <code>{search_id}</code> tidak ditemukan dalam database.")
            except Exception as e:
                show_card("danger", "error", "#EF4444", f"Error saat membaca database: {e}")
                
    st.markdown("---")
    st.markdown(f"#### {g_icon('list_alt', '#64748B', '22px')} Riwayat Seluruh Pasien Terdaftar", unsafe_allow_html=True)
    if os.path.exists(DB_FILE):
        df_all = pd.read_csv(DB_FILE, delimiter=';')
        st.dataframe(df_all.tail(20), use_container_width=True)
        st.caption(f"Menampilkan data terakhir dari total {len(df_all)} data tersimpan di {DB_FILE}")
    else:
        st.info("Belum ada data tersimpan di file database.")

#PELATIHAN MODEL MACHINE LEARNING & MLFLOW
with tab_training:
    st.markdown(f"### {g_icon('model_training', '#2563EB', '26px')} Pelatihan Model Machine Learning", unsafe_allow_html=True)
    st.write("Modul ini melatih model Random Forest Classifier dengan penyeimbangan data menggunakan teknik **SMOTE** dan validasi silang (Cross Validation) via **GridSearchCV**.")
    
    col_t1, col_t2 = st.columns([1, 2])
    with col_t1:
        st.markdown(f"""
        <div style="background:#F1F5F9; padding: 14px; border-radius: 8px;">
            <b>Alur Pelatihan Model:</b>
            <ol style="margin: 6px 0 0 0; padding-left: 20px; font-size: 13px;">
                <li>Pembersihan & Imputasi Nilai Hilang</li>
                <li>SMOTE (Oversampling Kelas Minoritas)</li>
                <li>Standarisasi Fitur (StandardScaler)</li>
                <li>Hyperparameter Tuning (Random Forest)</li>
                <li>Logging Eksperimen ke MLflow</li>
                <li>Penyimpanan Model ke <code>models/</code></li>
            </ol>
        </div>
        """, unsafe_allow_html=True)
    
    with col_t2:
        btn_train = st.button("Mulai Training Model & Log ke MLflow")
        
    if btn_train:
        with st.spinner("Sedang memproses pelatihan model..."):
            count = Counter(y)
            min_class_count = min(count.values())
            
            # Penerapan SMOTE
            try:
                k_neighbors = min(5, min_class_count - 1)
                smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
                X_resampled, y_resampled = smote.fit_resample(X, y)
                show_card("info", "balance", "#3B82F6", "SMOTE berhasil diterapkan untuk menyeimbangkan distribusi kelas data.")
                use_class_weight = False
            except Exception as e:
                show_card("warning", "warning", "#F59E0B", f"Gagal menjalankan SMOTE ({e}). Menggunakan opsi class_weight='balanced'.")
                X_resampled, y_resampled = X, y
                use_class_weight = True

            # Validasi Target
            if pd.isnull(y_resampled).any():
                show_card("danger", "error", "#EF4444", "Target y mengandung nilai kosong (NaN). Pelatihan dihentikan.")
                st.stop()

            # Split Data Train 80% dan Test 20%
            X_train, X_test, y_train, y_test = train_test_split(
                X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled
            )

            # Konfigurasi Classifier
            if use_class_weight:
                classifier = RandomForestClassifier(random_state=42, class_weight="balanced")
            else:
                classifier = RandomForestClassifier(random_state=42)

            # Pipeline Pemodelan
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('classifier', classifier)
            ])

            param_grid = {
                'classifier__n_estimators': [100],
                'classifier__max_depth': [None],
                'classifier__min_samples_split': [2]
            }

            grid_search = GridSearchCV(
                pipeline, param_grid, cv=3, scoring='accuracy', n_jobs=-1
            )

            mlflow.set_experiment("Diabetes_Prediction")

            with mlflow.start_run():
                grid_search.fit(X_train, y_train)
                model_terbaik = grid_search.best_estimator_

                y_pred = model_terbaik.predict(X_test)
                acc = accuracy_score(y_test, y_pred)

                mlflow.log_params(grid_search.best_params_)
                mlflow.log_metric("accuracy", acc)
                mlflow.sklearn.log_model(model_terbaik, "model")

                # Simpan model ke file pkl
                joblib.dump(model_terbaik, MODEL_PATH)

            show_card("success", "verified", "#10B981", f"Model berhasil dilatih dengan Akurasi Testing: <b>{acc * 100:.2f}%</b>")
            show_card("info", "cloud_done", "#3B82F6", "Metrik, parameter, dan artefak model berhasil dicatat ke MLflow.")
            show_card("success", "save", "#10B981", f"Model tersimpan di <code>{MODEL_PATH}</code>. Anda sekarang dapat menggunakan form skrining pasien.")

#UNDUH DATASET & LAPORAN
with tab_unduh:
    st.markdown(f"### {g_icon('download', '#2563EB', '26px')} Unduh Data Rekam Medis", unsafe_allow_html=True)
    st.write("Unduh dataset pasien lengkap dalam format CSV untuk keperluan pelaporan atau analisis lanjutan.")
    
    # Siapkan data export dengan hasil prediksi jika model sudah ada
    df_export = df_raw.copy()
    if model is not None:
        X_exp = df_export[features].fillna(df_export[features].mean())
        y_pred_exp = model.predict(X_exp)
        y_prob_exp = model.predict_proba(X_exp)
        df_export['Prediksi_Model'] = np.where(y_pred_exp == 1, 'Akan Memiliki Diabetes', 'Tidak Akan Memiliki Diabetes')
        df_export['Probabilitas_Tidak_Diabetes'] = np.round(y_prob_exp[:, 0], 4)
        df_export['Probabilitas_Diabetes'] = np.round(y_prob_exp[:, 1], 4)
    
    csv_bytes = df_export.to_csv(index=False, sep=';').encode('utf-8')
    
    st.download_button(
        label="Unduh CSV Dataset Lengkap",
        data=csv_bytes,
        file_name='data_diabetes_lengkap.csv',
        mime='text/csv'
    )
    
    st.caption("File diekspor menggunakan pemisah titik koma (;) sesuai standar dataset awal.")