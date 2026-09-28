# 🏦 Prediksi Risiko Kredit Nasabah
### Studi Kasus: South German Credit Dataset (UCI Machine Learning Repository)

---

## 📋 Informasi Proyek

| Atribut | Detail |
|---|---|
| **Mata Kuliah** | Predictive Analytics |
| **Jenis Tugas** | Midterm Project (UTS) |
| **Nama** | Suci Fransisca Sisilia Rahmat (Sisil) |
| **NIM** | 24120500008 |
| **Program Studi** | Data Science, Universitas Cakrawala |
| **Jenis Task** | Classification (Klasifikasi Biner) |
| **Tanggal Pengerjaan** | 27 September 2026 |

---

## 🎯 Project Overview

### Konteks Masalah
Bank dan lembaga pembiayaan menghadapi *trade-off* klasik setiap kali menyetujui aplikasi kredit:
- **Opportunity Cost**: Menolak nasabah yang sebenarnya layak → kehilangan pendapatan bunga
- **Credit Loss**: Menyetujui nasabah yang berisiko gagal bayar → kerugian langsung

Model prediktif yang terlatih dari data historis dapat mempercepat proses *screening* awal, membuat kriteria keputusan lebih konsisten, dan dapat diaudit.

### Tujuan Prediksi
Membangun model klasifikasi biner untuk memprediksi `credit_risk` seorang pemohon kredit:
- **1 (Good Risk)** → Pemohon layak kredit
- **0 (Bad Risk)** → Pemohon berisiko gagal bayar

Berdasarkan **20 atribut** yang direkam saat pengajuan: riwayat rekening, riwayat kredit, tujuan pinjaman, jumlah pinjaman, tabungan, lama bekerja, status personal, properti, usia, pekerjaan, dll.

### Stakeholder
- **Primary**: *Credit risk analyst* & *loan officer* di lembaga pembiayaan konsumen
- **Peran Model**: *Decision-support tool* pada tahap *pre-screening* (bukan pengganti keputusan akhir analis)

---

## 📊 Dataset

| Atribut | Detail |
|---|---|
| **Nama Dataset** | South German Credit (UPDATE) |
| **Publisher** | UCI Machine Learning Repository |
| **Kontributor** | Dr. Hans Hofmann; update oleh Dr. Ulrike Grömping (Beuth University of Applied Sciences Berlin) |
| **Link** | [https://archive.ics.uci.edu/dataset/573/south+german+credit+update](https://archive.ics.uci.edu/dataset/573/south+german+credit+update) |
| **Tanggal Akses** | 27 September 2026 |
| **Dimensi** | 1.000 observasi × 21 kolom (20 prediktor + 1 target) |
| **Unit Observasi** | Satu aplikasi kredit individu |

> **Alasan pemilihan dataset ini**: Versi *update* (2019) telah memperbaiki kesalahan pengkodean atribut pada versi German Credit klasik yang lebih populer, sehingga lebih valid untuk pengambilan keputusan.

### Data Dictionary

| Variabel | Tipe | Keterangan |
|---|---|---|
| `status` | ordinal (1–4) | Status rekening giro saat ini |
| `duration` | numerik | Jangka waktu kredit (bulan) |
| `credit_history` | ordinal (0–4) | Riwayat pembayaran kredit sebelumnya |
| `purpose` | nominal (0–10) | Tujuan pinjaman (mobil, elektronik, pendidikan, dll.) |
| `amount` | numerik | Jumlah pinjaman (DM) |
| `savings` | ordinal (1–5) | Saldo tabungan |
| `employment_duration` | ordinal (1–5) | Lama bekerja di pekerjaan saat ini |
| `installment_rate` | ordinal (1–4) | Cicilan sebagai % dari pendapatan |
| `personal_status_sex` | nominal (1–4) | Status personal & jenis kelamin |
| `other_debtors` | nominal (1–3) | Ada tidaknya penjamin/co-applicant |
| `present_residence` | ordinal (1–4) | Lama tinggal di alamat saat ini |
| `property` | ordinal (1–4) | Jenis aset yang dimiliki |
| `age` | numerik | Usia pemohon (tahun) |
| `other_installment_plans` | nominal (1–3) | Cicilan lain di bank/toko lain |
| `housing` | nominal (1–3) | Status tempat tinggal (sewa/milik sendiri/gratis) |
| `number_credits` | ordinal (1–4) | Jumlah kredit yang sudah dimiliki di bank ini |
| `job` | ordinal (1–4) | Tingkat kualifikasi pekerjaan |
| `people_liable` | ordinal (1–2) | Jumlah tanggungan |
| `telephone` | nominal (1–2) | Punya telepon atas nama sendiri |
| `foreign_worker` | nominal (1–2) | Status pekerja asing |
| **`credit_risk`** | **target biner** | **0 = bad (gagal bayar), 1 = good (layak)** |

---

## 🛠️ Metodologi

### Alur Kerja

```
Data Raw (.asc)
     ↓
Data Loading & Konversi CSV
     ↓
Data Quality Assessment
     ↓
Data Preparation & Feature Engineering
     ↓
Exploratory Data Analysis (EDA)
     ↓
Modeling (Baseline → Logistic Regression → Random Forest → XGBoost)
     ↓
Evaluasi & Pemilihan Model Terbaik
     ↓
Analisis Feature Importance & Interpretasi
     ↓
Kesimpulan & Rekomendasi
```

### Temuan Kualitas Data
1. ✅ Tidak ada *missing value* eksplisit dan tidak ada baris duplikat
2. ⚠️ Semua kolom terkodekan sebagai integer — perlu encoding yang tepat (campuran numerik, ordinal, dan nominal)
3. ⚠️ **Distribusi target tidak seimbang**: 70% "good" vs 30% "bad" — ditangani lewat pemilihan metrik dan class weighting
4. ✅ Tidak ditemukan potensi *data leakage* — semua prediktor tersedia pada saat pengajuan

### Keputusan Data Preparation
| Keputusan | Alasan |
|---|---|
| Tidak imputasi missing value | Kode "unknown" mengandung sinyal bisnis yang valid |
| Tidak menghapus baris/kolom | Semua 20 variabel punya justifikasi bisnis |
| Variabel **numerik kontinu** (`duration`, `amount`, `age`) → `StandardScaler` | Mencegah bias skala pada Logistic Regression |
| Variabel **ordinal** → dipertahankan sebagai numerik | Urutan level bermakna |
| Variabel **nominal** → `OneHotEncoder` | Tidak ada urutan alami antar kategori |
| Preprocessing hanya di-*fit* pada data latih | Mencegah data leakage |
| Split data **80:20, stratified** | Menjaga proporsi kelas di train & test set |
| Outlier tidak dihapus | Nilai ekstrem adalah pola bisnis yang valid |

---

## 🤖 Model yang Digunakan

| Model | Keterangan |
|---|---|
| **Dummy Classifier** | Baseline sederhana (prediksi mayoritas kelas) |
| **Logistic Regression** | Model linear, interpretable |
| **Random Forest** | Ensemble berbasis decision tree, robust terhadap outlier |
| **XGBoost** | Gradient boosting, performa tinggi |

### Library Utama
```python
scikit-learn     # Preprocessing, model, evaluasi
xgboost          # XGBoost classifier
pandas           # Manipulasi data
numpy            # Komputasi numerik
matplotlib       # Visualisasi
seaborn          # Visualisasi statistik
```

---

## 📁 Struktur File

```
uts-predictive-analysis/
│
├── README.md                                           # Dokumentasi proyek (file ini)
├── Predictive Analysis - German Credit Risk.ipynb     # Notebook utama
├── south_german_credit.csv                            # Dataset (CSV hasil konversi)
├── credit+approval (1).zip                            # File zip original dari UCI
└── Executive_Summary_German_Credit.pdf               # Ringkasan eksekutif hasil analisis
```

---

## ▶️ Cara Menjalankan

### Prerequisites
Pastikan Python dan library berikut sudah terinstal:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost
```

### Langkah
1. Clone repository ini:
   ```bash
   git clone https://github.com/Sisilfr/uts-predictive-analysis.git
   cd uts-predictive-analysis
   ```

2. Buka notebook di Jupyter atau Google Colab:
   ```bash
   jupyter notebook "Predictive Analysis - German Credit Risk.ipynb"
   ```
   Atau upload ke [Google Colab](https://colab.research.google.com/) dan jalankan seluruh cell secara berurutan.

3. Dataset (`south_german_credit.csv`) sudah tersedia di repository. Jika ingin mengunduh ulang dari sumber aslinya, jalankan cell pertama pada notebook yang akan otomatis mengunduh dan mengkonversi file `.asc` dari UCI.

---

## 📈 Metrik Evaluasi

Karena distribusi kelas tidak seimbang dan *cost* kesalahan tidak simetris (meloloskan nasabah "bad" lebih merugikan daripada menolak nasabah "good"), evaluasi model menggunakan:

- **ROC-AUC** — kemampuan model membedakan dua kelas secara keseluruhan
- **Recall (Sensitivity)** untuk kelas "bad" — seberapa banyak nasabah berisiko yang berhasil dideteksi
- **Precision** — seberapa akurat prediksi "bad" dari model
- **F1-Score** — keseimbangan antara precision dan recall
- **Confusion Matrix** — gambaran detail distribusi prediksi benar/salah

---

## 📝 Catatan Penting / Limitations

- Dataset ini berasal dari satu bank di Jerman Selatan pada era tertentu — generalisasi ke konteks lain perlu validasi ulang
- Beberapa variabel ordinal memiliki arah kode yang tidak selalu intuitif (sesuai `codetable.txt` resmi UCI)
- Model ini dimaksudkan sebagai **decision-support tool**, bukan pengganti keputusan akhir credit analyst

---

*Proyek ini dibuat sebagai bagian dari Ujian Tengah Semester (UTS) mata kuliah Predictive Analytics, Program Studi Data Science, Universitas Cakrawala.*
