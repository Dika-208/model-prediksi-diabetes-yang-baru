# 🩺 Sistem Deteksi Dini & Analisis Prediksi Diabetes

Aplikasi Machine Learning berbasis web (**Streamlit**) untuk membantu klasifikasi dan skrining risiko diabetes pada pasien menggunakan algoritma **Random Forest Classifier**, teknik penyeimbangan data **SMOTE**, dan pencatatan eksperimen **MLflow**. Proyek ini dikembangkan untuk kebutuhan kompetisi dan analisis data kesehatan.

---

## 🚀 Fitur Utama

1. **Dashboard & Analisis Dataset:**
   - Ringkasan metrik pasien (KPI Cards: Total Sampel, Positif/Negatif Diabetes, Rata-rata Usia).
   - Visualisasi distribusi status aktual vs hasil prediksi model (*Pie Chart* & perbandingan akurasi).
2. **Skrining & Prediksi Pasien Baru:**
   - Form input rekam medis dengan parameter terstandar medis (Glukosa, BMI, Tekanan Darah, dll.).
   - Hasil diagnosis visual yang jelas (Risiko Rendah / Terindikasi Diabetes) beserta persentase probabilitas.
   - **Rekomendasi Medis Cerdas:** Analisis otomatis gaya hidup dan saran tindakan medis berdasarkan nilai parameter pasien.
   - Otomatis tersimpan ke database lokal (`diabetes.csv`).
3. **Pencarian & Riwayat Pasien:**
   - Pencarian rekam medis pasien berdasarkan ID.
   - Tabel riwayat pasien yang tersimpan secara lokal.
4. **Pelatihan Model & Eksperimen MLflow:**
   - Penyeimbangan kelas data minoritas dengan **SMOTE** (*Synthetic Minority Over-sampling Technique*).
   - Optimasi hyperparameter menggunakan **GridSearchCV** (Cross-Validation).
   - Pelacakan parameter, metrik akurasi, dan artefak model ke **MLflow**.
5. **Ekspor Data:**
   - Unduh dataset lengkap hasil prediksi dalam format CSV.

---

## 🛠️ Teknologi yang Digunakan

* **Bahasa:** Python 3.13
* **Framework Web:** Streamlit
* **Machine Learning:** Scikit-Learn, Imbalanced-Learn (SMOTE)
* **Visualisasi Data:** Matplotlib, Seaborn
* **Model Tracking & MLOps:** MLflow
* **UI Icons:** Google Material Symbols

---

## 📁 Struktur Direktori

```text
├── models/                 # Model machine learning tersimpan (.pkl)
├── plots/                  # Visualisasi grafik evaluasi model
├── reports/                # Laporan teks evaluasi model
├── Dataset_Repeel.txt      # Dataset primer
├── diabetes.csv            # Database lokal rekam medis pasien
├── diagram.py              # Skrip evaluasi model & generate visualisasi grafik
├── prediksi.py             # Aplikasi web utama (Streamlit)
├── requirements.txt        # Daftar dependensi library
└── README.md               # Dokumentasi proyek
```

---

## 💻 Cara Menjalankan Proyek

### 1. Persiapan Environment
Pastikan Python 3 sudah terpasang. Install seluruh library yang diperlukan:
```bash
pip install -r requirements.txt
```

### 2. Menjalankan Aplikasi Web
```bash
streamlit run prediksi.py
```
> Aplikasi akan otomatis terbuka di browser pada alamat `http://localhost:8501`.

### 3. Menjalankan Evaluasi Grafik Model
```bash
python diagram.py
```
> Hasil grafik evaluasi akan otomatis tersimpan di folder `plots/` dan laporan teks di folder `reports/`.
