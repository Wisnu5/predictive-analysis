# Laporan Proyek Machine Learning Terapan 1 - Prediksi Harga Saham GOTO Menggunakan LSTM

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
| Kolom         | Tipe Data Awal | Tipe Data Setelah Pemrosesan |
|---------------|----------------|------------------------------|
| Tanggal       | String         | Datetime                     |
| Terakhir      | String         | Float                        |
| Pembukaan     | String         | Float                        |
| Tertinggi     | String         | Float                        |
| Terendah      | String         | Float                        |
| Vol.          | String         | Float                        |
| Perubahan%    | String         | Float (dihapus setelahnya)   |

### Bentuk Data
![image](https://github.com/user-attachments/assets/da72f837-2a95-498d-a5e9-8cb5ee899d04)
- **Jumlah Baris**: 1124
- **Jumlah Kolom**: 7
- **Periode**: 11 April 2022 – 25 Mei 2025

## 3. Data Preparation

- Kolom **Terakhir** (harga penutupan) dinormalisasi menggunakan `StandardScaler` agar data memiliki rata-rata 0 dan standar deviasi 1.
- Data dibagi menjadi:
  - **80% data latih (training)**: untuk melatih model.
  - **20% data uji (testing)**: untuk menguji performa model.
- Fungsi `split_target` digunakan untuk membuat urutan data. Dengan `look_back = 1`, model belajar menggunakan harga 1 hari sebelumnya untuk memprediksi harga hari berikutnya.
- Data diubah menjadi format 3D: `[jumlah sampel, langkah waktu, jumlah fitur]` sesuai kebutuhan arsitektur LSTM.

