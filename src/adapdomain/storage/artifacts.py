"""Bundle a run directory into a portable archive."""
from __future__ import annotations

import shutil
from pathlib import Path


def bundle_run(run_dir: str | Path, *, target_root: str | Path) -> Path:
    src = Path(run_dir)
    tgt_root = Path(target_root)
    tgt_root.mkdir(parents=True, exist_ok=True)
    archive = shutil.make_archive(str(tgt_root / src.name), "zip", root_dir=src)
    return Path(archive)
