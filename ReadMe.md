# ⚡ Analisis Prediksi Permintaan Listrik Global (Global Electricity Demand)

Proyek ini mengimplementasikan model _Machine Learning_ untuk memprediksi tingkat permintaan listrik (_electricity demand_) global berdasarkan indikator makroekonomi dan bauran energi. Proyek ini bertujuan untuk memberikan analisis prediktif yang dapat digunakan dalam menyeimbangkan pertumbuhan ekonomi dengan keberlanjutan lingkungan.

## Anggota Kelompok 9

- **Noviana Nur Aisyah** (2310817120005)
- **Naufal Elyzar** (2310817210019)
- **Ghani Mudzakir** (2310817110011)

📊 Dataset
Dataset yang digunakan berasal dari **Our World in Data (OWID) - Energy & Emissions**.

- **Jumlah Data:** 6.149 baris observasi (setelah pembersihan).
- **Fitur Utama:** 23 fitur prediktor (termasuk GDP, populasi, bauran energi fosil/terbarukan, dan emisi karbon).
- **Target Variabel:** `electricity_demand` (dalam satuan Terawatt-jam / TWh).

⚙️ Pipeline Pre-processing
Sebelum dilatih, data mentah diproses melalui tahapan sistematis untuk memastikan kualitas model dan mencegah _overfitting_:

1. **Pembersihan Target:** Menghapus data yang tidak memiliki nilai pada target prediksi.
2. **Pencegahan Data Leakage:** Menghapus fitur-fitur yang mengandung kata kunci _'electricity'_ atau _'generation'_ yang dapat membocorkan target ke dalam model.
3. **Seleksi Fitur & Missing Values:** Membuang fitur kategorikal identitas (`country`, `iso_code`), fitur turunan (`_per_capita`), serta kolom dengan _missing value_ ekstrem (>70%).
4. **Standardisasi:** Menggunakan `Pipeline` dari Scikit-Learn untuk mengintegrasikan `SimpleImputer` (median) dan `StandardScaler`.

🤖 Pemodelan & Keterbaruan (Hyperparameter Tuning)
Sebagai keterbaruan dari model _baseline_ (Regresi Linear), proyek ini menggunakan algoritma _Ensemble Tree_ lanjutan untuk menangani sifat non-linear dan _outlier_ pada data energi global.

- **Random Forest Regressor:** Dilakukan _tuning_ manual pada `max_depth` (hasil optimal pada kedalaman $\ge$ 10) dan menggunakan `oob_score=True` untuk validasi internal.
- **XGBoost Regressor:** Dimodifikasi untuk mencegah _overfitting_ dengan `n_estimators=500`, `learning_rate=0.05`, `max_depth=6`, serta _subsampling_ fitur dan data sebesar 80%.

📈 Hasil Evaluasi & Feature Importance
Model dievaluasi menggunakan metrik **MAE, RMSE, dan R² Score**. Performa terbaik diraih oleh algoritma **Random Forest (`max_depth=None`)**:

| Model                          | MAE   | RMSE   | R² Score | OOB Score |
| :----------------------------- | :---- | :----- | :------- | :-------- |
| Linear Regression _(Baseline)_ | 99.44 | 251.52 | 0.9897   | -         |
| Random Forest _(Tuned)_        | 26.16 | 128.49 | 0.9973   | 0.9972    |
| XGBoost _(Tuned)_              | 33.33 | 156.93 | 0.9959   | -         |

Berdasarkan ekstraksi **Feature Importance**, fitur `greenhouse_gas_emissions` (Emisi Gas Rumah Kaca) menyumbang **97.55%** kontribusi prediktif, menjadikannya prediktor terkuat terhadap lonjakan konsumsi listrik suatu negara.

🛡️ Uji Ketahanan Model (Stress Test)
Untuk membuktikan bahwa model tidak sekadar menghafal data, kami merancang modul **Stress Test** pasca-pelatihan secara manual. Modul ini mendeteksi anomali dengan menghitung _residual_ (selisih aktual vs prediksi) dan menerapkan batas statistik **IQR (Interquartile Range)**.

**Hasil Stress Test (Penghapusan Outlier Ekstrem):**
| Kondisi Data | RMSE | R² Score |
| :--- | :--- | :--- |
| Sebelum Outlier Removal | 128.49 | 0.9973 |
| Setelah Outlier Removal | 0.69 | 0.9999 |

**Kesimpulan:** Penurunan _error_ (RMSE) secara drastis dari 128.49 menjadi 0.69 membuktikan bahwa sebagian besar tingkat kesalahan model murni disebabkan oleh beberapa negara dengan anomali konsumsi listrik yang sangat ekstrem. Secara umum, model terbukti sangat tangguh (_robust_) dalam memprediksi mayoritas data distribusi normal di tingkat global.
