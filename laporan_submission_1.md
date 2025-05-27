# Laporan Proyek Machine Learning Terapan 1 - Wisnu Al Hussaeni

## Domain Proyek
Harga saham merupakan indikator penting dalam pasar keuangan yang mencerminkan performa perusahaan dan dinamika pasar. PT GoTo Gojek Tokopedia Tbk (GOTO), sebuah perusahaan teknologi terkemuka di Indonesia, memiliki saham yang diperdagangkan di Bursa Efek Indonesia (BEI) dengan kode **GOTO.JK**. Data historis saham GOTO dapat diakses melalui [Yahoo Finance](https://finance.yahoo.com/quote/GOTO.JK/history/). Prediksi harga saham yang akurat sangat penting bagi investor, trader, dan institusi keuangan untuk mendukung pengambilan keputusan investasi, manajemen risiko, dan strategi perdagangan. Pendekatan tradisional seperti analisis teknikal dan fundamental sering kali tidak mampu menangkap pola kompleks dan non-linear dalam data harga saham, yang dipengaruhi oleh faktor seperti sentimen pasar, kebijakan ekonomi, dan peristiwa global. Oleh karena itu, pendekatan berbasis *machine learning*, khususnya **Long Short-Term Memory (LSTM)**, menjadi relevan karena kemampuannya dalam menangani data deret waktu dengan ketergantungan jangka panjang. Proyek ini bertujuan untuk membangun model prediktif menggunakan LSTM untuk memprediksi harga penutupan saham **GOTO.JK** berdasarkan data historis dari Yahoo Finance. Dengan memanfaatkan kemampuan LSTM untuk menangkap pola temporal, proyek ini diharapkan dapat memberikan wawasan prediktif yang mendukung investor dalam membuat keputusan yang lebih tepat waktu dan *informed*, baik untuk strategi jangka pendek maupun menengah.

## Business Understanding
### Problem Statements
Berdasarkan latar belakang di atas, permasalahan yang akan dibahas dalam proyek ini adalah:
1. Seberapa akurat model LSTM dalam memprediksi harga penutupan saham **GOTO.JK** untuk 1 hari dan 5 hari ke depan berdasarkan data historis?
2. Bagaimana performa model LSTM dibandingkan dengan metrik evaluasi seperti **Root Mean Squared Error (RMSE)** dan **Mean Absolute Error (MAE)**?
3. Apakah model LSTM dapat digunakan untuk mendukung keputusan investasi jangka pendek atau menengah?

### Goals
Berdasarkan *problem statements*, tujuan proyek ini adalah:
1. Membangun model LSTM yang akurat untuk memprediksi harga penutupan saham **GOTO.JK**.
2. Mengevaluasi performa model menggunakan metrik **RMSE** dan **MAE**.
3. Menyediakan wawasan prediktif yang dapat mendukung keputusan investasi.

### Solution Statement
1. Melakukan **Exploratory Data Analysis (EDA)** untuk mengidentifikasi pola, tren, dan korelasi dalam data harga saham **GOTO.JK**.
2. Menggunakan model **LSTM** untuk memprediksi harga penutupan saham berdasarkan data historis.
3. Menggunakan metrik evaluasi seperti **RMSE** dan **MAE** untuk menilai performa model.
4. Melakukan normalisasi data menggunakan **StandardScaler** untuk memastikan data sesuai dengan kebutuhan model LSTM.
5. Mengoptimalkan model dengan **EarlyStopping** dan **ReduceLROnPlateau** untuk meningkatkan akurasi prediksi.


## Data Understanding
### Deskripsi Dataset
Dataset yang digunakan diambil dari [Yahoo Finance](https://finance.yahoo.com/quote/GOTO.JK/history/) menggunakan library `yfinance` dengan kode saham **GOTO.JK**. Dataset ini mencakup periode perdagangan harian dari **11 April 2022 hingga 25 Mei 2025**, terdiri dari **1124 baris** dan **7 kolom**: **Tanggal**, **Terakhir**, **Pembukaan**, **Tertinggi**, **Terendah**, **Volume**, dan **Perubahan%**. Dataset ini bersifat deret waktu dan berisi data numerik tanpa nilai kategorikal.

### Tipe Data
![image](https://github.com/user-attachments/assets/f863ea1b-11ca-4c36-b100-304ea941de88)
| Kolom         | Tipe Data Awal | Tipe Data Setelah Pemrosesan |
|---------------|----------------|------------------------------|
| Tanggal       | String         | Datetime64                   |
| Terakhir      | String         | Int64                        |
| Pembukaan     | String         | Int64                        |
| Tertinggi     | String         | Int64                        |
| Terendah      | String         | Int64                        |
| Vol.          | String         | Float64                      |
| Perubahan%    | String         | Float64                      |

### Bentuk Data
![image](https://github.com/user-attachments/assets/da72f837-2a95-498d-a5e9-8cb5ee899d04)
- **Jumlah Baris**: 1124
- **Jumlah Kolom**: 7
- **Periode**: 11 April 2022 – 25 Mei 2025

### Deskripsi Variabel
![image](https://github.com/user-attachments/assets/4a133e11-bc9b-4c89-ad32-16dd3e04a812)

| Variabel      | Keterangan                                                                 |
|---------------|----------------------------------------------------------------------------|
| Tanggal       | Tanggal perdagangan (format: DD/MM/YYYY).                                  |
| Terakhir      | Harga penutupan saham pada hari tersebut (dalam IDR, variabel target).     |
| Pembukaan     | Harga pembukaan saham pada hari tersebut (dalam IDR).                      |
| Tertinggi     | Harga tertinggi saham pada hari tersebut (dalam IDR).                      |
| Terendah      | Harga terendah saham pada hari tersebut (dalam IDR).                       |
| Vol.          | Jumlah saham yang diperdagangkan pada hari tersebut (dalam ribuan).        |
| Perubahan%    | Persentase perubahan harga penutupan dibandingkan hari sebelumnya.         |


### Menangani Missing Value dan Duplicate Data
![image](https://github.com/user-attachments/assets/fc134934-86db-4a7d-9a38-ad9b897c41c4)

Berdasarkan analisis awal:
- **Missing Values**: Tidak ada nilai yang hilang pada dataset (dikonfirmasi dengan `data.isnull().sum()`).
- **Duplicate Data**: Tidak ada data duplikat yang ditemukan (dikonfirmasi dengan `data.duplicated().sum()`).
- Kolom seperti **Terakhir**, **Pembukaan**, **Tertinggi**, **Terendah**, **Vol.**, dan **Perubahan%** awalnya bertipe string dengan karakter khusus (misalnya, koma untuk desimal atau simbol persen), sehingga memerlukan pembersihan dan konversi ke tipe numerik.

### Visualisasi Data (EDA)
![image](https://github.com/user-attachments/assets/694548a3-c94d-415d-b566-2a22f432f9f4)
# Insight Distribusi Data
## 1. Harga (Terakhir, Pembukaan, Tertinggi, Terendah)
- **Modus**: Sekitar `2500–3000`
- **Bentuk distribusi**: *Right-skewed*
- **Insight**:
  - Mayoritas harga berada di kisaran rendah.
  - Terdapat beberapa outlier dengan nilai sangat tinggi.
  - Menunjukkan stabilitas dengan sesekali fluktuasi ekstrem.

## 2. Volume Transaksi ("Vol.")
- **Modus**: Sekitar `0.0–0.1`
- **Bentuk distribusi**: *Left-skewed*
- **Insight**:
  - Sebagian besar volume transaksi kecil.
  - Ada beberapa kasus dengan volume sangat besar (outlier).

## 3. Perubahan Persentase ("Perubahan%")
- **Bentuk distribusi**: Simetris / mendekati normal
- **Pusat distribusi**: Di sekitar `0%`
- **Insight**:
  - Pergerakan harga cenderung stabil.
  - Variasi naik dan turun seimbang.

![image](https://github.com/user-attachments/assets/6642117d-641f-472b-9a9b-703193032c04)

### Pembukaan dan Penutupan
- **Stabilitas Awal (2021)**: Harga stabil di kisaran tinggi.
- **Krisis 2022**: Penurunan drastis, menandakan dampak besar dari faktor eksternal.
- **Pemulihan Lambat (2023–2025)**: Harga pulih secara bertahap, tetapi belum mencapai level sebelum krisis.
- **Insight**: Aset ini memiliki volatilitas tinggi, memerlukan strategi investasi yang hati-hati.

![image](https://github.com/user-attachments/assets/879ad4b3-224b-44cb-8e5f-a61a41f161f1)

### Vol
- **Aktivitas Tinggi (Awal 2022)**: Volume mencapai puncak tertinggi.
- **Penurunan Aktivitas (2022–2023)**: Volume turun drastis dan stabil di kisaran rendah.
- **Pemulihan Aktivitas (2024–2025)**: Volume meningkat kembali, meskipun belum mencapai level tertinggi.
- **Insight**: Volume transaksi berkorelasi dengan aktivitas pasar dan volatilitas harga.

![image](https://github.com/user-attachments/assets/4fbe1670-b122-46ef-a7be-4fae09bcea3f)

### Perubahan
- **Stabilitas Awal (2021)**: Persentase perubahan stabil di kisaran positif.
- **Krisis 2022**: Penurunan drastis hingga **-40%**, menandakan dampak besar dari faktor eksternal.
- **Pemulihan Lambat (2022–2025)**: Persentase perubahan pulih secara bertahap, meskipun belum mencapai level tertinggi.
- **Insight**: Aset ini memiliki volatilitas tinggi, memerlukan strategi investasi yang hati-hati.

![image](https://github.com/user-attachments/assets/c9ff8609-d6b9-45ef-b95f-8dd014e18265)

- **Hubungan Waktu dan Harga**: Ada hubungan negatif kuat antara waktu dan harga.
- **Volume Transaksi**: Memiliki hubungan negatif menengah dengan harga.
- **Perubahan Persentase**: Hubungan rendah dengan variabel lainnya.
- **Model Prediksi**: Fokus pada variabel yang memiliki korelasi tinggi dengan target prediksi.

## Data Preparation
Beberapa tahapan persiapan data dilakukan agar data dapat digunakan dalam model LSTM:

1. **Konversi Tanggal**: Kolom `'Tanggal'` diubah ke format `datetime` agar dapat diurutkan secara kronologis.
   
   ![image](https://github.com/user-attachments/assets/b5c9fee9-251b-4015-9b5b-0d0b9971d4ac)
   
2. **Pembersihan Data**:
   
   ![image](https://github.com/user-attachments/assets/55f16599-9f25-4a4d-981b-1634986205d5)
   
   - Kolom numerik seperti `'Terakhir'`, `'Pembukaan'`, `'Tertinggi'`, `'Terendah'`, dan `'Vol.'` dibersihkan dari karakter khusus (misalnya tanda titik atau koma) dan dikonversi ke tipe numerik (`float`).
3. **Penghapusan Kolom Tidak Relevan**:
   
   ![image](https://github.com/user-attachments/assets/a2bb0b4b-c8a6-425a-bcee-1ece7567785c)
   
   - Kolom `'Perubahan%'` dihapus karena redundan dan tidak digunakan dalam prediksi langsung.
4. **Normalisasi Data**:
   
   ![image](https://github.com/user-attachments/assets/03eeda7d-0198-458a-90a9-be0ef042d314)
   
   - Kolom `'Terakhir'` dinormalisasi menggunakan **StandardScaler** agar data memiliki rata-rata 0 dan standar deviasi 1.
5. **Pembentukan Dataset Time Series**:
    
   ![image](https://github.com/user-attachments/assets/a88af98e-12af-4184-89d2-293ae7b13dd9)
   
   - Fungsi `split_target()` digunakan untuk membentuk urutan input dan target.
   - Dengan `look_back = 1`, model belajar menggunakan harga 1 hari sebelumnya untuk memprediksi harga hari berikutnya.
6. **Format Input LSTM**:
   - Data diubah ke format 3D `[samples, time steps, features]` sesuai kebutuhan input model LSTM.
7. **Split Data**:

   ![image](https://github.com/user-attachments/assets/ee3d9bdd-5a3b-41cf-bea2-384eea290b0a)

   - **80% data latih**, **20% data uji**.
---

## 4. Modeling
### Cara Kerja LSTM
LSTM (Long Short-Term Memory) adalah varian dari Recurrent Neural Network (RNN) yang dirancang untuk mengatasi masalah *long-term dependency*. LSTM memiliki tiga jenis gerbang utama:
- **Forget Gate**: Menentukan informasi mana yang perlu dilupakan.
- **Input Gate**: Menentukan informasi mana yang akan disimpan dalam memori.
- **Output Gate**: Menentukan informasi apa yang akan dikeluarkan.
Dengan memanfaatkan cell state dan gating mechanism ini, LSTM dapat mempelajari pola jangka panjang dalam data sekuensial seperti harga saham.
---

### Model LSTM 
![image](https://github.com/user-attachments/assets/6200def0-bc7a-4774-8712-f9b006a4a98e)
![image](https://github.com/user-attachments/assets/bbf1ace0-f1ec-476d-b53f-975a66309b99)

- **Arsitektur**:
  - LSTM (50 units)
  - Dropout (0.2)
  - LSTM (50 units)
  - Dropout (0.2)
  - Dense (25 units)
  - Dense (1 unit) → Output

- **Parameter**:
  - Optimizer: `Adam`
  - Loss Function: `Mean Squared Error (MSE)`
  - Epoch: Maksimal 50
  - Batch Size: 32

- **Callback**:
  - `EarlyStopping(patience=10, restore_best_weights=True)`
  - `ReduceLROnPlateau(factor=0.2, patience=5, min_lr=1e-5)`
---
![image](https://github.com/user-attachments/assets/6bbf2d1f-df0d-4f5f-b496-b397efa84f71)
Setelah dilatih, model membuat prediksi pada data uji, lalu hasilnya dikembalikan ke skala asli menggunakan scaler.inverse_transform.
Hasil evaluasi:
- RMSE (Root Mean Squared Error): 151.20 → rata-rata error prediksi sekitar 151 unit dalam skala harga asli.
- MAE (Mean Absolute Error): 98.02 → rata-rata selisih absolut antara harga asli dan prediksi adalah 98 unit.

Grafik menunjukkan perbandingan harga saham asli (biru) dan prediksi (jingga) dari Juli 2024 hingga Juni 2025:
- Prediksi mengikuti tren harga asli dengan cukup baik.
- Secara keseluruhan, prediksi cukup dekat dengan harga asli, sesuai dengan nilai RMSE dan MAE.
---

### Membuat urutan input-output (30 hari untuk memprediksi 5 hari berikutnya)
- create_sequences: Fungsi untuk membuat urutan data input (X) dan output (y)
- seq_length: panjang urutan input (berapa hari data sebelumnya yang digunakan)
- pred_length: panjang urutan prediksi (berapa hari ke depan yang akan diprediksi)

### 4.2 Model LSTM (Prediksi 5 Hari)
![image](https://github.com/user-attachments/assets/cdacfbe9-3bab-4572-bc8e-393c33cac0ea)
![image](https://github.com/user-attachments/assets/aceef4ef-37b9-4399-b1b5-2f6653ef3a46)
![image](https://github.com/user-attachments/assets/e03b43c0-49db-47a1-9514-1ce6f06055cc)

- **Strategi**:
  - Input: 30 hari terakhir (`seq_length = 30`)
  - Output: 5 hari ke depan (`pred_length = 5`)

- **Arsitektur**:
  - LSTM (100 units)
  - Dropout (0.2)
  - LSTM (100 units)
  - Dropout (0.2)
  - Dense (50 units)
  - Dense (5 units) → Output

- **Parameter**:
  - Optimizer: `Adam`
  - Loss Function: `MSE`
  - Epoch: 50
  - Batch Size: 32

- **Callback**:
  - `EarlyStopping(patience=10, restore_best_weights=True)`
  - `ReduceLROnPlateau(factor=0.2, patience=5, min_lr=1e-5)`
---

model cukup baik mengikuti tren tetapi terlambat merespons lonjakan tajam, terutama saat harga naik tajam (des 2024) dan drop mendadak (apr 2025).
Kemungkinan penyebab eror
- Kurang banyak fitur: Jika hanya menggunakan harga sebelumnya (lag features), maka model cenderung underfitting terhadap dinamika pasar.
- Model tidak autoregressive: Jika prediksi dilakukan secara langsung untuk 5 hari ke depan, tanpa memperhatikan ketergantungan antar-hari, maka kesalahan akumulatif bisa besar.
Saran Perbaikan
- Tambahkan fitur jangan menggunakan satu fitur saja
- Kombinasikan prediksi dari beberapa model: misalnya kombinasi ARIMA + LSTM atau XGBoost + LSTM.

## Kesimpulan Proyek Prediksi Harga Saham GOTO

1. Menjawab Problem Statement
- Prediksi harga saham GOTO untuk beberapa hari ke depan berhasil dilakukan menggunakan data historis harga penutupan.
- Model LSTM terbukti mampu menangkap pola temporal dari data historis, seperti tren naik dan turun harga dalam jangka pendek.
- Namun, performa model masih dapat ditingkatkan, terutama dalam menangkap perubahan harga yang tajam (lonjakan dan penurunan ekstrem), yang terlihat dari peningkatan error saat horizon prediksi lebih jauh.

2. Hasil evaluasi menggunakan metrik:
- RMSE: 300.17
- MAE: 219.97
- Ini menunjukkan bahwa model memiliki kemampuan dasar dalam memprediksi arah harga, namun masih terdapat deviasi yang cukup besar dari nilai aktual, terutama untuk horizon lebih panjang.
- Jika dibandingkan dengan baseline seperti naive forecast (misalnya: harga hari ini = harga kemarin), model LSTM menunjukkan peningkatan akurasi, khususnya dalam mengenali pola jangka pendek.

3. Goals Tercapai
- Model LSTM berhasil dibangun dan diimplementasikan untuk memprediksi harga saham.
- Pola pergerakan harga saham seperti tren dan siklus jangka pendek berhasil ditangkap oleh model.
- Evaluasi dengan metrik RMSE dan MAE memberikan gambaran seberapa baik model bekerja dan menjadi dasar untuk perbaikan selanjutnya (misalnya penambahan fitur, tuning parameter, atau pemilihan arsitektur lanjutan).

## Referensi
- Brownlee, J. (2020). *Deep Learning for Time Series Forecasting*. Machine Learning Mastery. Diakses pada 25 Mei 2025 dari [https://machinelearningmastery.com/deep-learning-for-time-series-forecasting/](https://machinelearningmastery.com/deep-learning-for-time-series-forecasting/)
- Dicoding. (2024). *Machine Learning Terapan*. Diakses pada 25 Mei 2025 dari [https://www.dicoding.com/academies/319-machine-learning-terapan](https://www.dicoding.com/academies/319-machine-learning-terapan)
- Yahoo Finance. (2025). *GOTO.JK Historical Data*. Diakses pada 25 Mei 2025 dari [https://finance.yahoo.com/quote/GOTO.JK/history/](https://finance.yahoo.com/quote/GOTO.JK/history/)
