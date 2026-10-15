# CNN-LSTM v9 Webapp Final Bundle

Paket ini adalah versi final website skripsi untuk prediksi **MET**, estimasi **kalori**, dan **BMI profile**.

## Yang sudah termasuk di dalam bundle
- `models/best_model.keras`
- `artifacts/minmax_scaler.pkl`
- source website lengkap
- worker subprocess agar server utama tidak ikut mati saat inferensi
- bridge loader yang **memprioritaskan SavedModel** jika nanti Anda menaruh hasil konversi di `models/best_model_savedmodel/`

## Fungsi website
- unggah CSV sensor
- prediksi **MET** per window
- hitung **kalori** dengan **berat badan**
- tampilkan **BMI profile** dari berat, tinggi, dan umur
- unduh hasil prediksi CSV

## Input profil pengguna
- **Berat badan**: wajib, dipakai untuk perhitungan kalori
- **Tinggi badan**: untuk BMI/profile
- **Umur**: untuk BMI/profile dan metadata pengguna

## Kolom CSV wajib
- `heart_rate`
- `hand_acc_16_x`, `hand_acc_16_y`, `hand_acc_16_z`
- `chest_acc_16_x`, `chest_acc_16_y`, `chest_acc_16_z`
- `ankle_acc_16_x`, `ankle_acc_16_y`, `ankle_acc_16_z`

Kolom opsional:
- `timestamp`
- `activity_id`

## Environment yang disarankan
Karena artifact model Anda berasal dari archive Keras 3, sedangkan Windows Anda sebelumnya crash saat import stack `keras/optree`, jalur yang paling aman tetap:
- coba bundle ini dulu apa adanya;
- kalau environment Windows masih crash saat membuka `.keras`, konversi dulu model ke **SavedModel** memakai `convert_keras3_to_savedmodel.py` di Linux/WSL/Colab, lalu taruh hasilnya di `models/best_model_savedmodel/`.

Website ini otomatis akan **memprioritaskan SavedModel** bila folder itu tersedia.

## Instalasi
Disarankan memakai virtual environment baru.

### Windows
```bash
py -3.11 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip uninstall -y keras optree tensorflow tensorflow-intel tf-keras tf_keras
pip install -r requirements.txt
```

## Diagnosa awal
```bash
python env_check.py
python diagnose_model.py
```

## Menjalankan website
```bash
python app.py
```
Atau:
```bash
run_app.bat
```

Lalu buka:
- `http://127.0.0.1:5000`

## Jika prediksi gagal
Periksa file berikut:
- `server.log`
- `last_worker_output.json`
- `last_worker_stderr.txt`
- `worker_stage.txt`
- `worker_faulthandler.log`

## Catatan metodologi
- pipeline inference memakai **window 5 detik** dan **overlap 50%**
- **berat badan** dipakai untuk hitung kalori
- **tinggi badan** dan **umur** hanya untuk **BMI profile**
- input model tetap **10 fitur sensor** seperti training CNN-LSTM v9

## Konversi ke SavedModel (opsional tapi disarankan jika Windows masih bermasalah)
Lihat file:
- `README_CONVERSION.md`
- `convert_keras3_to_savedmodel.py`
