"""Shared run-parameter logging.

Every entry point calls open_run_log() once, which creates a timestamped
file under logs/ recording the script, the resolved argparse namespace,
the environment, and (via close) the outcome.
"""

import datetime as _dt
import json
import platform
import subprocess
import sys
from pathlib import Path

LOG_DIR = Path(__file__).resolve().parent / "logs"


def timestamp():
    """UTC timestamp safe for filenames: 20260906T144233Z."""
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _package_versions():
    versions = {}
    for name in ("numpy", "scipy", "matplotlib", "ripser", "persim"):
        try:
            mod = __import__(name)
            versions[name] = getattr(mod, "__version__", "unknown")
        except Exception as exc:  # pragma: no cover - diagnostic path only
            versions[name] = f"IMPORT FAILED: {exc}"
    return versions


def _git_rev():
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            capture_output=True, text=True, timeout=5,
        )
        return out.stdout.strip() if out.returncode == 0 else None
    except Exception:
        return None


class RunLog:
    def __init__(self, path, record):
        self.path = path
        self.record = record

    def note(self, key, value):
        """Attach a result to the log record (written on close)."""
        self.record.setdefault("results", {})[key] = value

    def close(self, status="ok", error=None):
        self.record["status"] = status
        if error is not None:
            self.record["error"] = str(error)
        self.record["finished_utc"] = _dt.datetime.now(_dt.timezone.utc).isoformat()
        self.path.write_text(json.dumps(self.record, indent=2, default=str) + "\n")
        return self.path


def open_run_log(script_name, args, stamp=None):
    """Create logs/<script>_<timestamp>.json capturing all run parameters."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    stamp = stamp or timestamp()
    path = LOG_DIR / f"{script_name}_{stamp}.json"
    record = {
        "script": script_name,
        "timestamp_utc": stamp,
        "started_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "argv": sys.argv,
        "parameters": vars(args) if hasattr(args, "__dict__") else dict(args),
        "python": sys.version,
        "platform": platform.platform(),
        "packages": _package_versions(),
        "git_rev": _git_rev(),
    }
    # Write immediately so the parameters survive a crash mid-run.
    path.write_text(json.dumps(record, indent=2, default=str) + "\n")
    return RunLog(path, record)
