# Bridge Keras 3 -> TensorFlow 2.15 SavedModel

Masalah Anda sekarang ada pada format artifact:
- `best_model.keras` dibuat oleh Keras 3.13.2
- website Windows Anda lebih stabil di TensorFlow 2.15 (`tf.keras`)
- loader fallback `.weights.h5` tidak kompatibel langsung antar runtime tersebut

## Solusi paling aman
Konversi `best_model.keras` menjadi `SavedModel` di environment Keras 3 yang sehat (disarankan Linux / WSL / Google Colab), lalu copy folder hasil export ke website Windows.

## Langkah cepat di Colab / Linux / WSL
```bash
pip install keras==3.13.2 tensorflow==2.16.1
python convert_keras3_to_savedmodel.py   --input /path/to/best_model.keras   --output_dir /path/to/best_model_savedmodel
```

Setelah selesai, copy folder `best_model_savedmodel/` ke:
```text
models/best_model_savedmodel/
```

Website patch ini otomatis akan memprioritaskan folder `models/best_model_savedmodel/`.
Kalau folder itu ada, website tidak lagi mencoba membuka `best_model.keras` secara langsung.

## Diagnosa setelah copy
Di Windows website env Anda:
```bash
python diagnose_model.py
python app.py
```

## Catatan scaler
Log Anda menunjukkan `minmax_scaler.pkl` dibuat dari scikit-learn 1.8.0 tetapi website env memakai 1.4.2.
Itu sering masih bisa load, tetapi secara resmi tidak dijamin kompatibel. Bila perlu, samakan versi scikit-learn website dengan versi saat scaler dibuat.
