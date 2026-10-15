from __future__ import annotations

import csv
import json
import logging
import os
import secrets
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")

from flask import Flask, flash, redirect, render_template, request, send_file, url_for

from model_utils import REQUIRED_FEATURES, artifact_status

BASE_DIR = Path(__file__).resolve().parent
def pick_model_path() -> Path:
    savedmodel = BASE_DIR / "models" / "best_model_savedmodel"
    legacy_keras = BASE_DIR / "models" / "best_model.keras"
    if savedmodel.exists() and savedmodel.is_dir():
        return savedmodel
    return legacy_keras

MODEL_PATH = pick_model_path()
SCALER_PATH = BASE_DIR / "artifacts" / "minmax_scaler.pkl"
EXPORT_DIR = BASE_DIR / "exports"
WORKER_PATH = BASE_DIR / "predict_worker.py"
LOG_PATH = BASE_DIR / "server.log"
LAST_STDERR_PATH = BASE_DIR / "last_worker_stderr.txt"
LAST_STDOUT_PATH = BASE_DIR / "last_worker_stdout.txt"
LAST_PAYLOAD_PATH = BASE_DIR / "last_worker_output.json"
EXPORT_DIR.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)
app.config["SECRET_KEY"] = secrets.token_hex(16)
app.config["MAX_CONTENT_LENGTH"] = 256 * 1024 * 1024


def configure_logging() -> None:
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers.clear()

    fh = logging.FileHandler(LOG_PATH, encoding="utf-8")
    fh.setFormatter(formatter)
    root.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    root.addHandler(sh)


configure_logging()
logger = logging.getLogger("cnnlstm_v9_webapp")
logger.info("Starting app from %s", BASE_DIR)


@app.get("/")
def index():
    status = artifact_status(MODEL_PATH, SCALER_PATH)
    return render_template(
        "index.html",
        status=status,
        required_features=REQUIRED_FEATURES,
        default_weight=70,
        default_height=170,
        default_age=22,
    )


@app.post("/predict")
def predict():
    try:
        weight_kg = float(request.form.get("weight_kg", "").strip())
        height_cm = float(request.form.get("height_cm", "").strip())
        age_years = int(float(request.form.get("age_years", "").strip()))
    except Exception:
        flash("Berat badan, tinggi badan, dan umur harus diisi dengan angka yang valid.", "error")
        return redirect(url_for("index"))

    if weight_kg <= 0 or height_cm <= 0 or age_years <= 0:
        flash("Berat badan, tinggi badan, dan umur harus bernilai positif.", "error")
        return redirect(url_for("index"))

    upload = request.files.get("sensor_csv")
    if upload is None or not upload.filename:
        flash("Silakan unggah file CSV sensor terlebih dahulu.", "error")
        return redirect(url_for("index"))

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        input_path = tmpdir_path / "input.csv"
        output_path = tmpdir_path / "output.json"
        input_path.write_bytes(upload.read())

        cmd = [
            sys.executable,
            str(WORKER_PATH),
            "--input", str(input_path),
            "--output", str(output_path),
            "--model", str(MODEL_PATH),
            "--scaler", str(SCALER_PATH),
            "--weight", str(weight_kg),
            "--height", str(height_cm),
            "--age", str(age_years),
            "--filename", upload.filename,
        ]
        logger.info("Launching worker subprocess for %s", upload.filename)
        try:
            proc = subprocess.run(
                cmd,
                cwd=str(BASE_DIR),
                capture_output=True,
                text=True,
                timeout=300,
                env=os.environ.copy(),
            )
        except subprocess.TimeoutExpired:
            flash("Prediksi melebihi batas waktu 300 detik. Coba file yang lebih kecil atau restart aplikasi.", "error")
            logger.exception("Worker timeout")
            return redirect(url_for("index"))

        LAST_STDERR_PATH.write_text(proc.stderr or "", encoding="utf-8", errors="ignore")
        LAST_STDOUT_PATH.write_text(proc.stdout or "", encoding="utf-8", errors="ignore")
        logger.info("Worker finished. returncode=%s", proc.returncode)

        payload = None
        if output_path.exists():
            text = output_path.read_text(encoding="utf-8", errors="ignore")
            LAST_PAYLOAD_PATH.write_text(text, encoding="utf-8", errors="ignore")
            try:
                payload = json.loads(text)
            except Exception:
                payload = None

        if proc.returncode != 0:
            detail = None
            if payload and isinstance(payload, dict):
                detail = payload.get("error")
            if not detail and proc.stderr:
                lines = [line.strip() for line in proc.stderr.splitlines() if line.strip()]
                detail = lines[-1] if lines else None
            if not detail:
                detail = (
                    f"Prediksi gagal di worker (return code {proc.returncode}). "
                    "Cek last_worker_output.json, last_worker_stderr.txt, worker_stage.txt, dan worker_faulthandler.log."
                )
            flash(detail, "error")
            return redirect(url_for("index"))

        if not payload:
            flash("Worker selesai tetapi output tidak valid. Cek last_worker_output.json.", "error")
            return redirect(url_for("index"))

        if not payload.get("ok"):
            flash(payload.get("error", "Prediksi gagal."), "error")
            return redirect(url_for("index"))

        results = payload["results"]
        rows = results.get("prediction_rows") or results.get("window_results") or []
        export_name = save_predictions_csv(rows)
        results["prediction_rows"] = rows
        results.setdefault("preview_rows", rows[:30])
        results["download_name"] = export_name
        return render_template("result.html", **results)


@app.get("/download/<path:filename>")
def download(filename: str):
    safe_path = (EXPORT_DIR / filename).resolve()
    if EXPORT_DIR.resolve() not in safe_path.parents:
        flash("File tidak ditemukan.", "error")
        return redirect(url_for("index"))
    if not safe_path.exists():
        flash("File hasil prediksi tidak ditemukan.", "error")
        return redirect(url_for("index"))
    return send_file(safe_path, as_attachment=True, download_name=safe_path.name)


@app.get("/health")
def health():
    status = artifact_status(MODEL_PATH, SCALER_PATH)
    return {
        "status": "ok",
        "model_exists": status["model_exists"],
        "scaler_exists": status["scaler_exists"],
    }


def _normalize_export_row(row: dict) -> dict:
    return {
        "window": row.get("window", row.get("window_index", "")),
        "start_time_sec": row.get("start_time_sec", row.get("start_sec", "")),
        "end_time_sec": row.get("end_time_sec", row.get("end_sec", "")),
        "effective_duration_sec": row.get("effective_duration_sec", ""),
        "predicted_met": row.get("predicted_met", ""),
        "intensity": row.get("intensity", row.get("intensity_label", "")),
        "kcal_contribution": row.get("kcal_contribution", row.get("estimated_kcal", "")),
    }



def save_predictions_csv(rows: list[dict]) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"predictions_{timestamp}.csv"
    path = EXPORT_DIR / filename

    fieldnames = [
        "window",
        "start_time_sec",
        "end_time_sec",
        "effective_duration_sec",
        "predicted_met",
        "intensity",
        "kcal_contribution",
    ]

    normalized_rows = [_normalize_export_row(row) for row in rows]

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(normalized_rows)

    return filename


if __name__ == "__main__":
    from waitress import serve

    logger.info("Serving on http://127.0.0.1:5000")
    serve(app, host="127.0.0.1", port=5000, threads=1)
