from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal


Track = Literal["t1", "t2", "t3"]


@dataclass(frozen=True)
class RunSpec:
    run_id: str
    track: Track
    preprocessing: str
    checkpoint_path: str
    has_projection: bool | None = None
    data_dir: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)


def sanitize_run_id(run_id: str) -> str:
    # Conservative filename-safe sanitizer.
    out = []
    for ch in str(run_id):
        if ch.isalnum() or ch in ("-", "_", ".", "+"):
            out.append(ch)
        else:
            out.append("_")
    cleaned = "".join(out).strip("._")
    return cleaned or "run"


def load_run_specs(path: Path) -> tuple[dict[str, Any], list[RunSpec]]:
    """
    Load run specs from a TOML file.

    The file format is:

        processed_root = "/path/to/processedMaster"
        meta_path = "/path/to/processedMaster/metadata.csv"

        [[runs]]
        run_id = "exp01_fold0"
        track = "t1"
        preprocessing = "bp"
        checkpoint_path = "/path/to/model.ckpt"
        has_projection = false

        [[runs]]
        ...
    """
    import tomllib

    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    runs_raw = raw.get("runs", [])
    if not isinstance(runs_raw, list):
        raise TypeError(f"'runs' must be a list, got {type(runs_raw)}")

    global_cfg = {k: v for k, v in raw.items() if k != "runs"}

    runs: list[RunSpec] = []
    known = {"run_id", "track", "preprocessing", "checkpoint_path", "has_projection", "data_dir"}
    for i, entry in enumerate(runs_raw):
        if not isinstance(entry, dict):
            raise TypeError(f"runs[{i}] must be a table/dict, got {type(entry)}")
        run_id = str(entry.get("run_id", "")).strip()
        if not run_id:
            raise ValueError(f"runs[{i}] missing non-empty 'run_id'")
        track = str(entry.get("track", "")).strip()
        if track not in ("t1", "t2", "t3"):
            raise ValueError(f"runs[{i}] invalid track={track!r}; expected 't1'|'t2'|'t3'")
        preprocessing = str(entry.get("preprocessing", "")).strip()
        if not preprocessing:
            raise ValueError(f"runs[{i}] missing non-empty 'preprocessing'")
        checkpoint_path = str(entry.get("checkpoint_path", "")).strip()
        has_projection = entry.get("has_projection", None)
        if has_projection is not None:
            has_projection = bool(has_projection)
        data_dir = entry.get("data_dir", None)
        data_dir = str(data_dir).strip() if data_dir else None
        meta = {k: v for k, v in entry.items() if k not in known}
        runs.append(
            RunSpec(
                run_id=run_id,
                track=track,  # type: ignore[arg-type]
                preprocessing=preprocessing,
                checkpoint_path=checkpoint_path,
                has_projection=has_projection,
                data_dir=data_dir,
                meta=meta,
            )
        )

    return global_cfg, runs


def resolve_data_dir(run: RunSpec, *, processed_root: str | Path) -> Path:
    if run.data_dir:
        return Path(run.data_dir)
    return Path(processed_root) / run.preprocessing

