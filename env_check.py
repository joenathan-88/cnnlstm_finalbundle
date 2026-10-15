from __future__ import annotations

import os
import sys
from pathlib import Path

print(f"[ENV] Python      : {sys.version.split()[0]}")
print(f"[ENV] Executable  : {sys.executable}")
print(f"[ENV] CWD         : {Path.cwd()}")
print(f"[ENV] TF_ENABLE_ONEDNN_OPTS: {os.environ.get('TF_ENABLE_ONEDNN_OPTS', '(not set)')}")

try:
    import tensorflow as tf
    print(f"[IMPORT] tensorflow: OK | version={tf.__version__} | path={tf.__file__}")
    print(f"[IMPORT] tf.keras  : {'OK' if getattr(tf, 'keras', None) is not None else 'MISSING'}")
except Exception as exc:
    print(f"[IMPORT] tensorflow: FAIL | {exc.__class__.__name__}: {exc}")
