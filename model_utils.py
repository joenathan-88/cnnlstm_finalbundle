from __future__ import annotations

import io
import inspect
import json
import os
import pickle
import tempfile
import zipfile
import warnings
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")

import numpy as np
import pandas as pd
import tensorflow as tf

FS = 100
WINDOW_SIZE = FS * 5
HOP_SIZE = int(WINDOW_SIZE * 0.5)

REQUIRED_FEATURES = [
    "heart_rate",
    "hand_acc_16_x", "hand_acc_16_y", "hand_acc_16_z",
    "chest_acc_16_x", "chest_acc_16_y", "chest_acc_16_z",
    "ankle_acc_16_x", "ankle_acc_16_y", "ankle_acc_16_z",
]
INTENSITY_LABELS = ("Low (<3)", "Medium (3–6)", "High (>6)")


@dataclass
class WindowMeta:
    index: int
    start_idx: int
    end_idx: int
    start_time_sec: float
    end_time_sec: float
    effective_duration_sec: float


class UserInputError(ValueError):
    pass


class ArtifactError(RuntimeError):
    pass


class AttentionPooling(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.score_dense = tf.keras.layers.Dense(1)

    def call(self, x):
        scores = self.score_dense(x)
        weights = tf.nn.softmax(scores, axis=1)
        return tf.reduce_sum(weights * x, axis=1)

    def get_config(self):
        return super().get_config()


CUSTOM_OBJECTS = {"AttentionPooling": AttentionPooling}


class SavedModelPredictor:
    def __init__(self, export_dir: str | Path):
        self.export_dir = str(export_dir)
        self._loaded = tf.saved_model.load(self.export_dir)
        sigs = getattr(self._loaded, 'signatures', {})
        self._fn = sigs.get('serving_default') or sigs.get('serve')
        if self._fn is None:
            raise ArtifactError(
                "SavedModel berhasil ditemukan tetapi endpoint inferensi tidak tersedia. "
                "Expected signature 'serving_default' atau 'serve'."
            )
        _, kw = self._fn.structured_input_signature
        self._input_key = next(iter(kw.keys()), None)

    def predict(self, x: np.ndarray, verbose: int = 0):
        tensor = tf.convert_to_tensor(x, dtype=tf.float32)
        try:
            if self._input_key is not None:
                out = self._fn(**{self._input_key: tensor})
            else:
                out = self._fn(tensor)
        except TypeError:
            out = self._fn(tensor)
        if isinstance(out, dict):
            if len(out) == 1:
                out = next(iter(out.values()))
            else:
                out = out[next(iter(out.keys()))]
        return out.numpy()


@lru_cache(maxsize=1)
def get_scaler(scaler_path: str | Path):
    path = Path(scaler_path)
    if not path.exists():
        raise ArtifactError(
            f"Scaler belum ditemukan di: {path}. "
            "Salin file minmax_scaler.pkl ke folder artifacts/."
        )
    with path.open('rb') as f:
        scaler = pickle.load(f)
    return scaler


def _maybe_get_keras_archive_info(path: Path) -> dict[str, Any]:
    info: dict[str, Any] = {
        'is_zip': False,
        'keras_version': None,
        'backend': None,
        'weight_file': None,
        'custom_classes': [],
    }
    try:
        with zipfile.ZipFile(path, 'r') as zf:
            info['is_zip'] = True
            names = zf.namelist()
            if 'metadata.json' in names:
                meta = json.loads(zf.read('metadata.json').decode('utf-8', errors='ignore'))
                info['keras_version'] = meta.get('keras_version')
                info['backend'] = meta.get('backend')
            if 'config.json' in names:
                cfg = json.loads(zf.read('config.json').decode('utf-8', errors='ignore'))
                found_classes: list[str] = []

                def walk(obj: Any) -> None:
                    if isinstance(obj, dict):
                        if 'class_name' in obj:
                            found_classes.append(str(obj['class_name']))
                        for value in obj.values():
                            walk(value)
                    elif isinstance(obj, list):
                        for item in obj:
                            walk(item)
                walk(cfg)
                info['custom_classes'] = sorted(set(found_classes))
            weight_files = [n for n in names if n.endswith('.weights.h5') or n.endswith('weights.h5')]
            if weight_files:
                info['weight_file'] = weight_files[0]
    except Exception:
        return info
    return info


def _build_cnnlstm_v9_architecture() -> tf.keras.Model:
    inp = tf.keras.Input(shape=(WINDOW_SIZE, len(REQUIRED_FEATURES)))
    x = tf.keras.layers.Conv1D(64, 5, padding='same', activation='relu')(inp)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling1D(2)(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Conv1D(128, 5, padding='same', activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling1D(2)(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True))(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True))(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = AttentionPooling()(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.25)(x)
    out = tf.keras.layers.Dense(1, activation='linear')(x)
    return tf.keras.Model(inp, out)


def _try_load_savedmodel(path: Path):
    if not path.is_dir():
        raise RuntimeError('Model path bukan directory SavedModel.')
    return SavedModelPredictor(path)


def _try_load_full_model(path: Path) -> tf.keras.Model:
    load_model_fn = tf.keras.models.load_model
    kwargs: dict[str, Any] = {
        'filepath': str(path),
        'custom_objects': CUSTOM_OBJECTS,
        'compile': False,
    }
    try:
        sig = inspect.signature(load_model_fn)
        if 'safe_mode' in sig.parameters:
            kwargs['safe_mode'] = False
    except Exception:
        pass
    return load_model_fn(**kwargs)


def _try_rebuild_and_load_weights(path: Path) -> tf.keras.Model:
    archive_info = _maybe_get_keras_archive_info(path)
    if not archive_info.get('is_zip'):
        raise RuntimeError('File model bukan archive .keras yang valid.')
    weight_member = archive_info.get('weight_file')
    if not weight_member:
        raise RuntimeError('Archive .keras tidak berisi file weights yang dapat dibaca.')
    model = _build_cnnlstm_v9_architecture()
    _ = model(np.zeros((1, WINDOW_SIZE, len(REQUIRED_FEATURES)), dtype=np.float32), training=False)
    tmpdir = tempfile.mkdtemp(prefix='weights_extract_')
    tmpdir_path = Path(tmpdir)
    with zipfile.ZipFile(path, 'r') as zf:
        zf.extract(weight_member, path=tmpdir_path)
    weight_path = tmpdir_path / weight_member
    model.load_weights(str(weight_path))
    return model


def _format_model_error(path: Path, exc: Exception) -> str:
    info = _maybe_get_keras_archive_info(path) if path.is_file() else {}
    meta_parts: list[str] = [f"detail={exc.__class__.__name__}: {exc}"]
    if info.get('keras_version'):
        meta_parts.append(f"archive_keras={info['keras_version']}")
    meta_parts.append(f"runtime_tf={tf.__version__}")
    if info.get('custom_classes'):
        meta_parts.append('custom=' + ', '.join(info['custom_classes'][:8]))
    return 'Model gagal dimuat. ' + ' | '.join(meta_parts)


@lru_cache(maxsize=1)
def get_model(model_path: str | Path):
    path = Path(model_path)
    if not path.exists():
        raise ArtifactError(
            f"Model belum ditemukan di: {path}. "
            "Taruh model di models/best_model_savedmodel atau models/best_model.keras."
        )
    errors: list[Exception] = []
    strategies = (_try_load_savedmodel,) if path.is_dir() else (_try_load_full_model, _try_rebuild_and_load_weights)
    for strategy in strategies:
        try:
            return strategy(path)
        except Exception as exc:
            errors.append(exc)
    raise ArtifactError(_format_model_error(path, errors[-1])) from errors[-1]


def artifact_status(model_path: str | Path, scaler_path: str | Path) -> dict[str, bool]:
    return {'model_exists': Path(model_path).exists(), 'scaler_exists': Path(scaler_path).exists()}


def parse_uploaded_csv(file_bytes: bytes) -> pd.DataFrame:
    if not file_bytes:
        raise UserInputError('File CSV kosong.')
    last_error: Exception | None = None
    for sep in (None, ',', ';', '\t'):
        try:
            if sep is None:
                df = pd.read_csv(io.BytesIO(file_bytes), sep=None, engine='python')
            else:
                df = pd.read_csv(io.BytesIO(file_bytes), sep=sep)
            if df.shape[1] > 1:
                df.columns = [str(c).strip() for c in df.columns]
                return df
        except Exception as exc:
            last_error = exc
    raise UserInputError('CSV tidak bisa dibaca. Pastikan file berupa CSV dengan header kolom yang benar.') from last_error


def _coerce_numeric(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    df = df.copy()
    for col in columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df


def clean_input_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    df = df.copy()
    warnings_list: list[str] = []
    missing = [col for col in REQUIRED_FEATURES if col not in df.columns]
    if missing:
        raise UserInputError('Kolom wajib belum lengkap. Minimal kolom yang harus ada: ' + ', '.join(REQUIRED_FEATURES))
    has_timestamp = 'timestamp' in df.columns
    has_activity = 'activity_id' in df.columns
    if has_timestamp:
        df = _coerce_numeric(df, ['timestamp'])
        ts_missing = int(df['timestamp'].isna().sum())
        if ts_missing:
            warnings_list.append(f'{ts_missing} baris dibuang karena timestamp tidak valid.')
            df = df.dropna(subset=['timestamp'])
        df = df.sort_values('timestamp').reset_index(drop=True)
        if len(df) >= 3:
            diffs = np.diff(df['timestamp'].to_numpy(dtype=float))
            diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
            if len(diffs):
                median_dt = float(np.median(diffs))
                if abs(median_dt - (1 / FS)) > 0.003:
                    warnings_list.append(
                        f'Median interval timestamp terdeteksi sekitar {median_dt:.4f} detik. Model dilatih pada data 100 Hz, jadi hasil bisa bergeser.'
                    )
    else:
        df['timestamp'] = np.arange(len(df), dtype=np.float64) / FS
        warnings_list.append('Kolom timestamp tidak ditemukan; waktu diasumsikan berdasarkan sampling 100 Hz.')
    if has_activity:
        df = _coerce_numeric(df, ['activity_id'])
        activity_missing = int(df['activity_id'].isna().sum())
        if activity_missing:
            warnings_list.append(f'{activity_missing} nilai activity_id tidak valid dan diperlakukan sebagai 0/transisi.')
        df['activity_id'] = df['activity_id'].fillna(0).astype(int)
        transition_rows = int((df['activity_id'] == 0).sum())
        if transition_rows:
            warnings_list.append(f'{transition_rows} baris dengan activity_id = 0 dibuang karena dianggap transisi.')
            df = df[df['activity_id'] != 0]
    else:
        df['activity_id'] = np.nan
        warnings_list.append('Kolom activity_id tidak ditemukan; window dibuat berurutan tanpa pembatas aktivitas.')
    df = _coerce_numeric(df, REQUIRED_FEATURES)
    imu_cols = [c for c in REQUIRED_FEATURES if c != 'heart_rate']
    imu_invalid = int(df[imu_cols].isna().any(axis=1).sum())
    if imu_invalid:
        warnings_list.append(f'{imu_invalid} baris dibuang karena ada nilai IMU yang kosong/tidak valid.')
        df = df.dropna(subset=imu_cols)
    hr_missing = int(df['heart_rate'].isna().sum())
    if hr_missing:
        df['heart_rate'] = df['heart_rate'].interpolate(method='linear', limit_direction='both')
        warnings_list.append(f'{hr_missing} nilai heart_rate diinterpolasi secara linear.')
    if df['heart_rate'].isna().any():
        raise UserInputError('Kolom heart_rate masih memiliki nilai kosong setelah interpolasi. Pastikan ada data heart rate yang valid.')
    df = df.reset_index(drop=True)
    if len(df) < WINDOW_SIZE:
        raise UserInputError(f'Data setelah preprocessing hanya {len(df)} baris. Minimal {WINDOW_SIZE} baris dibutuhkan untuk 1 window 5 detik pada 100 Hz.')
    return df, warnings_list


def make_windows_for_inference(df: pd.DataFrame) -> tuple[np.ndarray, list[WindowMeta]]:
    feat = df[REQUIRED_FEATURES].to_numpy(dtype=np.float32)
    activity = df['activity_id'].to_numpy()
    time_values = df['timestamp'].to_numpy(dtype=np.float64)
    X_list: list[np.ndarray] = []
    windows: list[WindowMeta] = []
    for start in range(0, len(df) - WINDOW_SIZE + 1, HOP_SIZE):
        end = start + WINDOW_SIZE
        if not np.isnan(activity[start]):
            if np.any(activity[start:end] != activity[start]):
                continue
        start_time_sec = float(time_values[start])
        end_time_sec = float(time_values[end - 1] + (1 / FS))
        X_list.append(feat[start:end])
        windows.append(WindowMeta(index=len(windows) + 1, start_idx=start, end_idx=end, start_time_sec=start_time_sec, end_time_sec=end_time_sec, effective_duration_sec=0.0))
    if not X_list:
        raise UserInputError('Tidak ada window valid yang terbentuk. Periksa panjang data, konsistensi aktivitas, dan format CSV.')
    X = np.stack(X_list).astype(np.float32)
    covered_until = -1
    adjusted_windows: list[WindowMeta] = []
    for meta in windows:
        new_start = max(meta.start_idx, covered_until + 1)
        effective_samples = max(0, meta.end_idx - new_start)
        effective_duration_sec = effective_samples / FS
        covered_until = max(covered_until, meta.end_idx - 1)
        adjusted_windows.append(WindowMeta(index=meta.index, start_idx=meta.start_idx, end_idx=meta.end_idx, start_time_sec=meta.start_time_sec, end_time_sec=meta.end_time_sec, effective_duration_sec=effective_duration_sec))
    return X, adjusted_windows


def transform_minmax(X: np.ndarray, scaler: Any) -> np.ndarray:
    n_samples, seq_len, n_features = X.shape
    flat = X.reshape(n_samples * seq_len, n_features)
    scaled = scaler.transform(flat)
    return scaled.reshape(n_samples, seq_len, n_features).astype(np.float32)


def predict_met(model: Any, X_scaled: np.ndarray) -> np.ndarray:
    pred_out = model.predict(X_scaled, verbose=0).reshape(-1)
    pred_met = np.exp(pred_out).astype(np.float32)
    pred_met = np.clip(pred_met, 0.5, 20.0)
    return pred_met


def met_to_intensity_label(met_value: float) -> str:
    if met_value < 3:
        return INTENSITY_LABELS[0]
    if met_value <= 6:
        return INTENSITY_LABELS[1]
    return INTENSITY_LABELS[2]


def format_seconds(total_seconds: float) -> str:
    total_seconds = float(max(0.0, total_seconds))
    seconds_rounded = int(round(total_seconds))
    minutes, seconds = divmod(seconds_rounded, 60)
    hours, minutes = divmod(minutes, 60)
    if hours > 0:
        return f"{hours}j {minutes}m {seconds}d"
    if minutes > 0:
        return f"{minutes}m {seconds}d"
    return f"{seconds_rounded}d"


def compute_bmi_profile(weight_kg: float, height_cm: float, age_years: int) -> dict[str, Any]:
    height_m = height_cm / 100.0
    bmi = weight_kg / (height_m ** 2)
    if bmi < 18.5:
        category = 'Berat badan kurang'
    elif bmi < 25:
        category = 'Berat badan normal'
    elif bmi < 30:
        category = 'Berat badan berlebih'
    else:
        category = 'Obesitas'
    return {
        'weight_kg': round(weight_kg, 2),
        'height_cm': round(height_cm, 2),
        'age_years': int(age_years),
        'bmi': round(float(bmi), 2),
        'category': category,
        'bmi_category': category,
        'age_note': 'BMI hanya ditampilkan sebagai informasi profil. Umur dan tinggi badan tidak dipakai sebagai input model.',
    }


def build_activity_conclusion(avg_met: float, total_duration_sec: float, intensity_rows: list[dict[str, Any]], total_kcal: float) -> dict[str, Any]:
    high_pct = next((float(r['percent']) for r in intensity_rows if r['label'] == 'High (>6)'), 0.0)
    medium_pct = next((float(r['percent']) for r in intensity_rows if r['label'] == 'Medium (3–6)'), 0.0)
    low_pct = next((float(r['percent']) for r in intensity_rows if r['label'] == 'Low (<3)'), 0.0)

    duration_text = format_seconds(total_duration_sec)

    if avg_met < 3.0:
        title = 'Aktivitas masih tergolong ringan'
        status = 'Kurang efektif'
        tone = 'warn'
        summary = (
            f'Berdasarkan prediksi MET, aktivitas Anda cenderung ringan selama {duration_text}. '
            'Aktivitas seperti ini baik untuk pemanasan atau pemulihan, tetapi biasanya belum cukup kuat jika tujuan Anda adalah sesi olahraga utama.'
        )
        recommendation = 'Coba tingkatkan tempo, durasi, atau kontinuitas latihan secara bertahap agar lebih efektif.'
    elif avg_met < 6.0:
        if total_duration_sec >= 1800:
            title = 'Aktivitas sudah baik untuk olahraga harian'
            status = 'Sudah baik'
            tone = 'ok'
            summary = (
                f'Prediksi menunjukkan olahraga Anda berada pada intensitas sedang dengan durasi {duration_text}. '
                'Ini umumnya sudah baik untuk menjaga kebugaran harian.'
            )
            recommendation = 'Pertahankan konsistensi latihan. Anda bisa menambah variasi atau sedikit menaikkan intensitas bila tubuh sudah terbiasa.'
        else:
            title = 'Aktivitas sudah cukup, tetapi durasinya masih bisa ditambah'
            status = 'Sudah cukup'
            tone = 'ok'
            summary = (
                f'Intensitas olahraga Anda sudah masuk kategori sedang, tetapi durasi efektif yang teranalisis baru {duration_text}. '
                'Sesi ini sudah cukup, namun masih bisa dibuat lebih optimal dengan durasi yang sedikit lebih panjang.'
            )
            recommendation = 'Bila kondisi memungkinkan, tambahkan durasi latihan secara bertahap agar manfaat kebugarannya lebih terasa.'
    elif avg_met < 8.5:
        title = 'Aktivitas tergolong berat dan efektif'
        status = 'Sudah baik'
        tone = 'ok'
        summary = (
            f'Olahraga Anda terdeteksi pada intensitas berat selama {duration_text}. '
            'Ini menunjukkan sesi yang efektif untuk meningkatkan kebugaran, terutama bila Anda memang terbiasa berolahraga.'
        )
        recommendation = 'Pastikan tetap memberi waktu pemanasan, pendinginan, dan istirahat yang cukup agar tubuh tidak terlalu terbebani.'
    else:
        title = 'Aktivitas sangat berat'
        status = 'Terlalu berat'
        tone = 'warn'
        summary = (
            f'Prediksi MET menunjukkan aktivitas yang sangat berat selama {duration_text}. '
            'Sesi seperti ini bisa efektif untuk latihan intens, tetapi bagi sebagian orang bisa terasa terlalu berat bila dilakukan terlalu sering atau tanpa pemulihan yang cukup.'
        )
        recommendation = 'Perhatikan respons tubuh. Bila perlu, kurangi intensitas atau selingi dengan hari latihan yang lebih ringan.'

    reasons = [
        f'Rata-rata intensitas aktivitas Anda adalah {avg_met:.2f} MET.',
        f'Sebagian besar waktu berada pada intensitas sedang/tinggi sebesar {medium_pct + high_pct:.1f}%.',
        f'Estimasi energi yang dikeluarkan selama sesi ini sekitar {total_kcal:.2f} kcal.',
    ]

    if high_pct >= 60:
        reasons.append('Porsi intensitas tinggi cukup dominan, sehingga sesi terasa lebih berat bagi pengguna awam.')
    elif low_pct >= 60:
        reasons.append('Porsi intensitas ringan masih dominan, sehingga sesi ini lebih mirip aktivitas ringan daripada latihan utama.')

    return {
        'title': title,
        'status': status,
        'tone': tone,
        'summary': summary,
        'recommendation': recommendation,
        'reasons': reasons,
        'disclaimer': 'Kesimpulan ini adalah interpretasi otomatis berbasis prediksi MET dari data sensor, bukan diagnosis medis atau saran profesional.',
    }


def compute_energy_kcal(predictions: np.ndarray, windows: list[WindowMeta], weight_kg: float) -> tuple[list[dict[str, Any]], float]:
    rows: list[dict[str, Any]] = []
    total_kcal = 0.0
    for meta, met in zip(windows, predictions):
        duration_h = meta.effective_duration_sec / 3600.0
        kcal = float(met) * float(weight_kg) * duration_h
        total_kcal += kcal
        intensity = met_to_intensity_label(float(met))
        row = {
            'window': meta.index,
            'window_index': meta.index,
            'start_time_sec': round(meta.start_time_sec, 3),
            'start_sec': round(meta.start_time_sec, 3),
            'end_time_sec': round(meta.end_time_sec, 3),
            'end_sec': round(meta.end_time_sec, 3),
            'effective_duration_sec': round(meta.effective_duration_sec, 3),
            'predicted_met': round(float(met), 4),
            'intensity': intensity,
            'intensity_label': intensity,
            'kcal_contribution': round(kcal, 6),
            'estimated_kcal': round(kcal, 6),
        }
        rows.append(row)
    return rows, float(total_kcal)


def summarize_predictions(predictions: np.ndarray) -> dict[str, Any]:
    labels = [met_to_intensity_label(float(v)) for v in predictions]
    counts = {label: labels.count(label) for label in INTENSITY_LABELS}
    return {
        'mean_met': round(float(np.mean(predictions)), 4),
        'median_met': round(float(np.median(predictions)), 4),
        'min_met': round(float(np.min(predictions)), 4),
        'max_met': round(float(np.max(predictions)), 4),
        'window_count': int(len(predictions)),
        'intensity_counts': counts,
    }


def build_results(predictions: np.ndarray, windows: list[WindowMeta], weight_kg: float, height_cm: float, age_years: int) -> dict[str, Any]:
    if len(predictions) != len(windows):
        raise ValueError("Jumlah prediksi dan metadata window tidak sama.")

    records, total_kcal = compute_energy_kcal(predictions, windows, weight_kg)

    duration_by_intensity = {label: 0.0 for label in INTENSITY_LABELS}
    total_duration_sec = 0.0
    weighted_met_sum = 0.0

    for pred_met, meta in zip(predictions, windows, strict=True):
        pred_met_f = float(pred_met)
        effective_sec = float(meta.effective_duration_sec)
        intensity = met_to_intensity_label(pred_met_f)
        total_duration_sec += effective_sec
        weighted_met_sum += pred_met_f * effective_sec
        duration_by_intensity[intensity] += effective_sec

    avg_met = weighted_met_sum / total_duration_sec if total_duration_sec > 0 else 0.0
    preview = records[:30]

    intensity_rows = []
    for label in INTENSITY_LABELS:
        dur = duration_by_intensity[label]
        pct = (dur / total_duration_sec * 100.0) if total_duration_sec > 0 else 0.0
        intensity_rows.append(
            {
                "label": label,
                "duration_sec": round(dur, 2),
                "duration_text": format_seconds(dur),
                "percent": round(pct, 2),
            }
        )

    explanation_cards = [
        {
            'title': 'Apa itu MET?',
            'text': 'MET adalah ukuran sederhana untuk menunjukkan seberapa berat aktivitas Anda. Semakin tinggi angkanya, semakin berat aktivitasnya.'
        },
        {
            'title': 'Apa arti kalori di halaman ini?',
            'text': 'Kalori adalah perkiraan energi yang dikeluarkan selama aktivitas, dihitung dari prediksi MET dan berat badan Anda.'
        },
        {
            'title': 'Apa arti kategori intensitas?',
            'text': 'Ringan berarti aktivitas santai, sedang berarti cukup aktif, dan tinggi berarti aktivitas berat.'
        },
    ]

    conclusion = build_activity_conclusion(avg_met, total_duration_sec, intensity_rows, total_kcal)

    return {
        "summary": {
            "total_windows": len(records),
            "covered_duration_sec": round(total_duration_sec, 2),
            "covered_duration_text": format_seconds(total_duration_sec),
            "average_met": round(avg_met, 4),
            "max_met": round(float(np.max(predictions)), 4),
            "min_met": round(float(np.min(predictions)), 4),
            "total_kcal": round(total_kcal, 4),
        },
        "bmi_profile": compute_bmi_profile(weight_kg, height_cm, age_years),
        "intensity_rows": intensity_rows,
        "prediction_rows": records,
        "preview_rows": preview,
        "conclusion": conclusion,
        "explanation_cards": explanation_cards,
    }
