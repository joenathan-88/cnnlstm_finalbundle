from __future__ import annotations

import argparse
import faulthandler
import json
import os
import traceback
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")

BASE_DIR = Path(__file__).resolve().parent
STAGE_PATH = BASE_DIR / "worker_stage.txt"
FAULT_PATH = BASE_DIR / "worker_faulthandler.log"


def mark(stage: str) -> None:
    STAGE_PATH.write_text(stage, encoding="utf-8")


def write_payload(out_path: Path, payload: dict) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    with FAULT_PATH.open("w", encoding="utf-8") as fh:
        faulthandler.enable(file=fh, all_threads=True)

        ap = argparse.ArgumentParser()
        ap.add_argument("--input", required=True)
        ap.add_argument("--output", required=True)
        ap.add_argument("--model", required=True)
        ap.add_argument("--scaler", required=True)
        ap.add_argument("--weight", required=True, type=float)
        ap.add_argument("--height", required=True, type=float)
        ap.add_argument("--age", required=True, type=int)
        ap.add_argument("--filename", required=True)
        args = ap.parse_args()

        out_path = Path(args.output)

        try:
            mark("import model_utils begin")
            from model_utils import (
                ArtifactError,
                UserInputError,
                build_results,
                clean_input_dataframe,
                get_model,
                get_scaler,
                make_windows_for_inference,
                parse_uploaded_csv,
                predict_met,
                transform_minmax,
            )
            mark("import model_utils ok")
        except Exception as exc:
            write_payload(
                out_path,
                {
                    "ok": False,
                    "stage": "import_model_utils",
                    "error": f"Gagal import dependency worker: {exc.__class__.__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                },
            )
            return 10

        try:
            mark("read input begin")
            file_bytes = Path(args.input).read_bytes()
            raw_df = parse_uploaded_csv(file_bytes)
            mark(f"raw_df_rows={len(raw_df)}")

            clean_df, warnings = clean_input_dataframe(raw_df)
            mark(f"clean_df_rows={len(clean_df)}")

            X, windows = make_windows_for_inference(clean_df)
            mark(f"X_shape={tuple(X.shape)} | windows={len(windows)}")

            scaler = get_scaler(args.scaler)
            mark("get_scaler ok")

            X_scaled = transform_minmax(X, scaler)
            mark(f"X_scaled_shape={tuple(X_scaled.shape)}")

            model = get_model(args.model)
            mark("get_model ok")

            predictions = predict_met(model, X_scaled)
            mark(f"predict ok | n={len(predictions)}")

            results = build_results(
                predictions=predictions,
                windows=windows,
                weight_kg=args.weight,
                height_cm=args.height,
                age_years=args.age,
            )
            results["input_info"] = {
                "filename": args.filename,
                "raw_rows": len(raw_df),
                "clean_rows": len(clean_df),
                "weight_kg": round(args.weight, 2),
                "height_cm": round(args.height, 2),
                "age_years": args.age,
            }
            results["warnings"] = warnings
            mark("write payload ok")
            write_payload(out_path, {"ok": True, "results": results})
            return 0
        except (UserInputError, ArtifactError) as exc:
            write_payload(out_path, {"ok": False, "stage": "user_or_artifact", "error": str(exc)})
            return 2
        except Exception as exc:
            write_payload(
                out_path,
                {
                    "ok": False,
                    "stage": "runtime",
                    "error": f"{exc.__class__.__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                },
            )
            return 3


if __name__ == "__main__":
    raise SystemExit(main())
