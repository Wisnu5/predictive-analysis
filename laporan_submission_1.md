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
Visualisasi data dilakukan menggunakan library `matplotlib` untuk melihat tren harga penutupan:
- **Interpretasi**:
  - Harga penutupan saham **GOTO** menunjukkan volatilitas tinggi, terutama pada periode awal *listing* (2022) dan beberapa periode fluktuasi signifikan, kemungkinan akibat sentimen pasar teknologi.
  - Tidak ada *outlier* ekstrem yang terdeteksi dalam data harga.
- **Multivariate Analysis**:
  - Korelasi antar variabel numerik (**Open**, **High**, **Low**, **Close**, **Volume**) dianalisis untuk memahami hubungan antar fitur. Harga penutupan (**Terakhir**) memiliki korelasi tinggi dengan **Pembukaan**, **Tertinggi**, dan **Terendah**, tetapi korelasi rendah dengan **Volume**.


