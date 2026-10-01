"""Shared run logging: every script logs its parameters to a timestamped file."""

import datetime
import logging
import pathlib
import sys

LOG_DIR = pathlib.Path(__file__).resolve().parent / "logs"


def start_run_log(script_name, params):
    """Create logs/<script>_<YYYYmmdd-HHMMSS>.log, record all params, return logger.

    params is a dict (e.g. vars(args) from argparse). The log also mirrors to
    stderr so runs are visible on the command line.
    """
    LOG_DIR.mkdir(exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = LOG_DIR / f"{script_name}_{stamp}.log"

    logger = logging.getLogger(script_name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    for handler in (logging.FileHandler(log_path), logging.StreamHandler(sys.stderr)):
        handler.setFormatter(fmt)
        logger.addHandler(handler)

    logger.info("run start: %s", script_name)
    logger.info("command line: %s", " ".join(sys.argv))
    for key in sorted(params):
        logger.info("param %s = %r", key, params[key])
    logger.info("log file: %s", log_path)
    return logger
