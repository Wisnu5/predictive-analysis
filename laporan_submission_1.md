# Prediksi Harga Saham GOTO Menggunakan LSTM

## 1. Business Understanding

### Problem Statement
- Bagaimana memprediksi harga penutupan saham GOTO untuk beberapa hari ke depan menggunakan data historis?
- Apakah model LSTM mampu menangkap pola temporal dari data harga dan volume saham?
- Bagaimana performa model prediksi dibandingkan dengan baseline?

### Goals
- Membangun model prediksi harga saham menggunakan algoritma LSTM.
- Mengidentifikasi pola pergerakan harga saham berdasarkan data historis.
- Mengevaluasi performa model menggunakan metrik error seperti RMSE dan MAE.

---

## 2. Data Understanding

### Variabel pada Dataset:
- **Tanggal**: Tanggal perdagangan (format: DD/MM/YYYY).
- **Terakhir**: Harga penutupan saham pada hari tersebut.
- **Pembukaan**: Harga pembukaan saham pada hari tersebut.
- **Tertinggi**: Harga tertinggi saham pada hari tersebut.
- **Terendah**: Harga terendah saham pada hari tersebut.
- **Vol.**: Volume perdagangan saham (dalam ribuan).
- **Perubahan%**: Persentase perubahan harga penutupan dibandingkan hari sebelumnya.

---

## 3. Data Preparation

- Kolom **Terakhir** (harga penutupan) dinormalisasi menggunakan `StandardScaler` agar data memiliki rata-rata 0 dan standar deviasi 1.
- Data dibagi menjadi:
  - **80% data latih (training)**: untuk melatih model.
  - **20% data uji (testing)**: untuk menguji performa model.
- Fungsi `split_target` digunakan untuk membuat urutan data. Dengan `look_back = 1`, model belajar menggunakan harga 1 hari sebelumnya untuk memprediksi harga hari berikutnya.
- Data diubah menjadi format 3D: `[jumlah sampel, langkah waktu, jumlah fitur]` sesuai kebutuhan arsitektur LSTM.

---

## 4. Modeling

### 4.1 Model LSTM (Prediksi 1 Hari)

- Arsitektur:
  - 2 lapisan **LSTM** (masing-masing 50 unit).
  - Lapisan **Dropout (rate = 0.2)** untuk mencegah overfitting.
  - 2 lapisan **Dense**:
    - Dense(25)
    - Dense(1) sebagai output harga penutupan.
- Optimizer: **Adam**
- Loss function: **Mean Squared Error (MSE)**
- Epoch: Maksimal 50
- Batch size: 32
- Callback:
  - **EarlyStopping**: berhenti jika validasi error stagnan selama 10 epoch.
  - **ReduceLROnPlateau**: menurunkan learning rate jika error tidak membaik.

---

## 5. Evaluation (Prediksi 1 Hari)

- Hasil prediksi dikembalikan ke skala asli menggunakan `scaler.inverse_transform`.
- Metrik evaluasi:
  - **RMSE**: 151.20 → rata-rata error sekitar 151 poin harga.
  - **MAE**: 98.02 → rata-rata selisih absolut sekitar 98 poin.
- Visualisasi menunjukkan bahwa model mengikuti tren harga asli dengan cukup baik.

---

## 6. Multi-step Forecasting (Prediksi 5 Hari)

### 6.1 Strategi

- Menggunakan fungsi `create_sequences` untuk membentuk:
  - Input (`X`): 30 hari terakhir (`seq_length = 30`)
  - Output (`y`): 5 hari ke depan (`pred_length = 5`)
- Data dinormalisasi dan diformat dalam bentuk 3D.

### 6.2 Hasil Evaluasi

- Metrik untuk masing-masing hari dari hari ke-1 hingga ke-5:
  - **Overall RMSE**: 300.17
  - **Overall MAE**: 219.98
- Hasil ini menunjukkan bahwa:
  - Model memiliki performa yang menurun dalam prediksi 5 hari ke depan.
  - Prediksi jangka pendek (1 hari) lebih akurat dibandingkan prediksi jangka menengah (5 hari).

---

## 7. Kesimpulan

- Model LSTM mampu mempelajari pola temporal dari data harga saham GOTO dan menunjukkan performa cukup baik untuk prediksi 1 hari ke depan.
- Dengan RMSE sebesar 151.20 dan MAE sebesar 98.02, model menunjukkan hasil yang akurat pada jangka pendek.
- Namun, akurasi menurun ketika memprediksi 5 hari ke depan (RMSE: 300.17, MAE: 219.98), menunjukkan bahwa kompleksitas prediksi meningkat seiring bertambahnya horizon waktu.
- LSTM cocok untuk prediksi harga saham berbasis data historis, terutama untuk jangka pendek. Untuk prediksi jangka panjang, diperlukan pendekatan lanjutan seperti stacked LSTM, attention mechanism, atau model hybrid.
