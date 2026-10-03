"""Pipeline logging helpers: mirror terminal logs to a per-run file."""

import logging
import os
import re
import time
from pathlib import Path
from typing import Optional, Tuple


class _FlushFileHandler(logging.FileHandler):
    def emit(self, record):
        super().emit(record)
        self.flush()


def create_pipeline_log(run_label: Optional[str] = None) -> Tuple[str, logging.Handler]:
    log_dir = Path(os.environ.get("PIPELINE_LOG_DIR", "logs"))
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    safe_label = re.sub(r"[^A-Za-z0-9_.-]+", "_", run_label or "pipeline").strip("_") or "pipeline"
    log_path = log_dir / f"{timestamp}_{safe_label}.log"

    handler = _FlushFileHandler(log_path, encoding="utf-8")
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    logging.getLogger().addHandler(handler)
    return str(log_path), handler


def close_pipeline_log(handler: Optional[logging.Handler]) -> None:
    if handler is None:
        return
    root = logging.getLogger()
    root.removeHandler(handler)
    handler.close()
