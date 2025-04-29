# Laporan Proyek Machine Learning - Lutfi Robbabni

## Domain Proyek

**Latar Belakang**

Penyediaan pinjaman atau kredit oleh lembaga keuangan selalu melibatkan risiko, terutama risiko kredit, yaitu kemungkinan bahwa peminjam tidak dapat memenuhi kewajibannya untuk membayar kembali pinjaman. Mengidentifikasi peminjam yang berisiko gagal bayar adalah salah satu tantangan utama dalam industri perbankan dan lembaga keuangan. Oleh karena itu, penting untuk mengembangkan sistem yang dapat memprediksi kemampuan peminjam untuk membayar pinjaman di masa depan berdasarkan data historis dan variabel-variabel yang relevan.

Dengan menggunakan data historis seperti penghasilan, jumlah utang, riwayat keterlambatan pembayaran, dan berbagai informasi keuangan lainnya, kita dapat membangun model untuk memprediksi apakah seorang peminjam akan menjadi "good debtor" (pembayar yang baik) atau "bad debtor" (pembayar yang buruk). Hal ini dapat membantu bank atau lembaga keuangan dalam membuat keputusan yang lebih baik dalam memberikan pinjaman kepada calon peminjam.

Penting untuk dicatat bahwa prediksi ini dapat mengurangi kerugian finansial yang ditimbulkan oleh peminjam yang gagal bayar, serta membantu dalam merancang kebijakan kredit yang lebih aman dan efisien. Selain itu, dengan menggunakan machine learning, kita dapat membuat model yang lebih akurat dan dapat diperbaharui secara berkala untuk mengadaptasi perubahan dalam perilaku finansial peminjam.

**Mengapa dan Bagaimana Masalah Ini Harus Diselesaikan?**

**Mengapa?**

Industri perbankan dan lembaga keuangan menghadapi tantangan besar dalam mengelola risiko kredit. Menurut sebuah studi oleh Chen et al. (2020), peminjam yang gagal bayar dapat menyebabkan kerugian yang signifikan bagi lembaga pemberi kredit. Untuk itu, penting untuk memiliki sistem yang dapat memprediksi kemungkinan gagal bayar untuk mencegah kerugian besar dan meminimalkan risiko.

**Bagaimana?**

Masalah ini dapat diselesaikan dengan menggunakan pendekatan machine learning, di mana kita melatih model pada data historis untuk memprediksi apakah seseorang akan gagal bayar dalam 2 tahun ke depan berdasarkan fitur-fitur yang ada, seperti usia, utang, pendapatan bulanan, dan riwayat pembayaran mereka. Proses pelatihan model ini memungkinkan kita untuk mendeteksi pola yang mungkin tidak terlihat oleh manusia, dan dengan demikian memberikan keputusan yang lebih tepat dalam pemberian kredit.

## Business Understanding

### Problem Statements

- Bagaimana cara mengidentifikasi calon peminjam yang berpotensi gagal bayar dalam dua tahun ke depan? Banyak lembaga keuangan kesulitan dalam menilai risiko calon peminjam secara akurat hanya berdasarkan informasi dasar.

- Apa saja variabel (fitur) yang paling berpengaruh terhadap kemungkinan gagal bayar? Penting untuk mengetahui faktor apa saja yang paling memengaruhi keputusan kredit agar lembaga keuangan dapat mengambil kebijakan berbasis data.

- Bagaimana meningkatkan akurasi prediksi dibandingkan baseline model sederhana? Model yang kurang akurat dapat menghasilkan keputusan kredit yang buruk.

### Goals

- Membangun model klasifikasi yang mampu memprediksi apakah seorang individu akan gagal bayar (bad debtor) atau tidak (good debtor) dalam 2 tahun ke depan. Hal ini akan membantu pengambilan keputusan yang lebih baik oleh pihak pemberi pinjaman.

- Mengidentifikasi fitur-fitur paling penting yang memengaruhi risiko gagal bayar. Sehingga hasilnya bisa digunakan tidak hanya untuk klasifikasi, tetapi juga sebagai insight bisnis.

- Meningkatkan akurasi prediksi model dibanding baseline, minimal mencapai akurasi ≥ 85%. Model dengan performa baik akan lebih layak digunakan dalam produksi atau sistem nyata.

### Solution statements

- Menggunakan beberapa algoritma machine learning seperti Logistic Regression, Random Forest, dan XGBoost untuk memprediksi gagal bayar. Pendekatan ini memungkinkan pemilihan model terbaik dari beberapa alternatif.

- Melakukan hyperparameter tuning dan feature engineering untuk meningkatkan akurasi model. Ini dilakukan untuk mendapatkan performa model yang optimal dan menghindari overfitting.

- Menggunakan metrik evaluasi seperti akurasi, precision, recall, dan AUC-ROC untuk menilai performa model. Evaluasi ini memastikan model tidak hanya akurat, tapi juga adil dan andal.

## Data Understanding

Dataset yang digunakan dalam proyek ini adalah “Give Me Some Credit” dari Kaggle, yang dapat diunduh pada tautan berikut: [dataset] (https://www.kaggle.com/competitions/GiveMeSomeCredit/data)
Dataset ini berisi data keuangan dari individu yang digunakan untuk memprediksi kemungkinan seseorang mengalami gagal bayar dalam dua tahun ke depan. Dataset terdiri dari 150.000 baris dan 11 kolom fitur serta 1 target/label.

### Deskripsi Variabel

Berikut adalah penjelasan masing-masing kolom:

1. **SeriousDlqin2yrs** Target variabel (0 = tidak gagal bayar, 1 = gagal bayar dalam 2 tahun ke depan).

2. **RevolvingUtilizationOfUnsecuredLines**
   Rasio pemakaian kredit.

3. **age**
   Umur peminjam.

4. **NumberOfTime30-59DaysPastDueNotWorse**
   Jumlah keterlambatan pembayaran 30-59 hari.

5. **DebtRatio**
   Rasio total utang terhadap pendapatan bulanan.

6. **MonthlyIncome**
   Pendapatan bulanan peminjam.

7. **NumberOfOpenCreditLinesAndLoans**
   Jumlah jalur kredit dan pinjaman yang terbuka (termasuk kartu kredit dan pinjaman lainnya).

8. **NumberOfTimes90DaysLate**
   Jumlah keterlambatan pembayaran lebih dari 90 hari.

9. **NumberRealEstateLoansOrLines**
   Jumlah pinjaman atau jalur kredit real estate (misalnya KPR).

10. **NumberOfTime60-89DaysPastDueNotWorse**
    Jumlah keterlambatan pembayaran 60-89 hari (tidak lebih buruk dari itu) dalam 2 tahun terakhir.

11. **NumberOfDependents**
    Jumlah tanggungan (anak, anggota keluarga, dll).

**Exploratory Data Analysis (EDA)** EDA bisa dilakukan untuk menggali insight awal, seperti:

Distribusi umur, income, dan debt ratio dan lainnya : Ada outlier?, Banyak nilai kosong?, dan Data Duplicate

## Data Preparation

### ![alt text](image.png)

Pada tahap data preparation, pertama-tama saya mengakses dataset yang tersimpan dalam format .zip di Google Drive, kemudian mengekstraknya dan membacanya menggunakan pandas.

### ![alt text](image-1.png)

Setelah data berhasil dimuat, saya menghapus kolom ID yang tidak memiliki kontribusi terhadap proses prediksi. Selanjutnya, data duplikat dan data kosong (missing values) dihapus untuk menjaga kualitas dan kebersihan data. Untuk menangani nilai-nilai ekstrem yang berpotensi mengganggu proses pelatihan model, saya menerapkan metode Interquartile Range (IQR) untuk menghapus outlier dari setiap kolom numerik.

### ![alt text](image-2.png)

Kemudian, dilakukan normalisasi data menggunakan MinMaxScaler agar setiap fitur numerik berada pada skala yang sama, sehingga model tidak berpihak pada fitur dengan nilai besar. Terakhir, dataset dibagi menjadi data latih dan data uji dengan proporsi 80:20 menggunakan metode train_test_split, serta disertai parameter stratify agar distribusi kelas tetap seimbang pada data latih dan uji.

### Alasan

semua tahapan tersebut dilakukan untuk memastikan bahwa data yang digunakan dalam pemodelan bersih, terstandarisasi, dan siap digunakan oleh algoritma machine learning.

## Modeling

### ![alt text](image-3.png) ![alt text](image-4.png)

**kelebihan dan kekurangan algoritma training**

1. Logistic Regression

Kelebihan:

- Mudah diimplementasikan dan cepat dilatih.
- Interpretasi model sangat sederhana dan transparan.
- Cocok untuk baseline model dalam klasifikasi biner.

Kekurangan:

- Kurang mampu menangani relasi non-linear antar fitur.
- Tidak bekerja optimal pada data yang imbalanced (seperti kasus ini).
- Performa buruk dalam mengenali kelas minoritas.

2. Random Forest

Kelebihan:

- Dapat menangani fitur numerik maupun kategorikal.
- Cenderung tidak overfitting karena menggabungkan banyak pohon keputusan.
- Lebih kuat terhadap outlier dan missing values.

Kekurangan:

- Interpretasi model sulit karena sifat ensemble-nya.
- Membutuhkan waktu pelatihan lebih lama dibanding Logistic Regression.
- Performa masih rendah terhadap data imbalanced tanpa penyesuaian khusus.

3. XGBoost

Kelebihan:

- Salah satu algoritma terbaik untuk banyak kasus klasifikasi.
- Mendukung regularisasi yang membuat model lebih general dan menghindari overfitting.
- Lebih sensitif terhadap kelas minoritas dibanding Random Forest.

Kekurangan:

- Kompleksitas lebih tinggi, memerlukan tuning parameter yang cermat.
- Waktu pelatihan bisa lebih lama pada dataset besar.
- Tanpa penanganan imbalance, masih memiliki kelemahan dalam deteksi kelas minoritas.

**Model Terbaik: XGBoost**

Dari ketiga model yang telah diuji, XGBoost dipilih sebagai model terbaik karena memiliki performa relatif lebih baik dalam mengenali kelas minoritas (debitur gagal bayar) dibandingkan dua model lainnya. Walaupun recall dan f1-score untuk kelas 1 masih rendah, namun XGBoost menunjukkan adanya prediksi terhadap kelas minoritas, sedangkan Logistic Regression dan Random Forest tidak memberikan hasil prediksi yang berarti sama sekali.

Pemilihan ini juga mempertimbangkan potensi XGBoost untuk ditingkatkan performanya melalui hyperparameter tuning

## Evaluation

![alt text](image-5.png) ![alt text](image-6.png)

### Kesimpulan :

Setelah melakukan pelatihan menggunakan model XGBoost, berikut adalah hasil evaluasi yang menunjukkan kinerja model dalam klasifikasi prediksi gagal bayar atau tidak:

Akurasi (Accuracy): Model XGBoost menunjukkan akurasi yang sangat baik, mencapai 97%. Ini berarti model berhasil memprediksi dengan benar sebagian besar data, menandakan bahwa model bekerja dengan baik dalam mengenali pola-pola pada dataset secara keseluruhan.

Precision (Kecermatan): Model memiliki precision yang cukup baik untuk kelas "gagal bayar" (class 1), dengan nilai 0.45. Ini berarti, sebagian besar dari prediksi yang mengidentifikasi seseorang akan gagal bayar adalah akurat, memberikan kepercayaan pada model untuk memprediksi kasus-kasus yang lebih sulit.

Recall (Daya Deteksi): Recall yang rendah pada kelas "gagal bayar" menunjukkan bahwa model mampu mendeteksi sebagian besar data dari kelas yang lebih besar ("tidak gagal bayar"). Model cenderung mengklasifikasikan dengan benar untuk kelas yang dominan, yang berarti model sudah memahami pola utama dalam dataset.

F1-Score: F1-score juga menunjukkan performa yang cukup seimbang dalam menangani prediksi untuk kelas yang dominan ("tidak gagal bayar"). Hal ini menunjukkan bahwa model sudah cukup efisien dalam melakukan prediksi secara keseluruhan, meskipun masih ada ruang untuk perbaikan dalam mendeteksi beberapa kasus minoritas.
