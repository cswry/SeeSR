"""Shared helpers for the SeeSR experiment framework.

Nothing here imports torch at module level, so configuration handling and
metadata utilities can be used (and tested) on machines without a GPU.
"""
import ast
import copy
import csv
import datetime as _dt
import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path, PurePath

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
UPSTREAM_TEST_SCRIPT = REPO_ROOT / "test_seesr.py"

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")


# --------------------------------------------------------------------------- #
# Upstream defaults
# --------------------------------------------------------------------------- #
def upstream_defaults(script_path=UPSTREAM_TEST_SCRIPT):
    """Return the argparse defaults declared in upstream ``test_seesr.py``.

    The defaults are read statically with ``ast`` so they always reflect the
    checked-out upstream file, and no heavy dependency has to be imported.
    """
    tree = ast.parse(Path(script_path).read_text(encoding="utf-8"))
    defaults = {}
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and getattr(node.func, "attr", None) == "add_argument"):
            continue
        flag = node.args[0].value
        name = flag.lstrip("-")
        kw = {k.arg: k.value for k in node.keywords}
        if "action" in kw and ast.literal_eval(kw["action"]) == "store_true":
            defaults[name] = False
        elif "default" in kw:
            defaults[name] = ast.literal_eval(kw["default"])
        else:
            defaults[name] = None
    return defaults


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
def deep_update(base, overrides):
    out = copy.deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_update(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def set_dotted(cfg, dotted_key, value):
    """Set ``cfg['a']['b'] = value`` for ``dotted_key='a.b'``; the path must exist."""
    keys = dotted_key.split(".")
    node = cfg
    for key in keys[:-1]:
        if key not in node or not isinstance(node[key], dict):
            raise KeyError(f"Unknown config section '{key}' in override '{dotted_key}'")
        node = node[key]
    if keys[-1] not in node:
        raise KeyError(f"Unknown config key '{dotted_key}'")
    node[keys[-1]] = value


def parse_override(text):
    """Parse ``key=value`` where value is YAML (so 30 -> int, null -> None)."""
    if "=" not in text:
        raise ValueError(f"Override must look like key=value, got '{text}'")
    key, raw = text.split("=", 1)
    return key.strip(), yaml.safe_load(raw)


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path} does not contain a YAML mapping")
    return data


def load_sweep(path):
    """Expand a sweep file into ``[(name, resolved_config), ...]``.

    Either an explicit ``experiments`` list, or a ``grid``: every combination of ``variants`` (named
    groups of overrides; optional) and ``axes`` (dotted key -> list of values), named with
    ``name_template``. Template fields are ``variant`` and the last component of each axis key.
    The variant name is stored as ``experiment.variant``.
    """
    sweep = load_yaml(path)
    base_path = (Path(path).parent / sweep["base_config"]).resolve()
    base = load_yaml(base_path)
    items = list(sweep.get("experiments") or [])
    grid = sweep.get("grid")
    if grid:
        import itertools
        axes = grid.get("axes") or {}
        variants = grid.get("variants") or {"default": {}}
        keys = list(axes)
        for vname, combo in itertools.product(variants, itertools.product(*(axes[k] for k in keys))):
            values = dict(zip(keys, combo))
            short = {"variant": vname, **{k.split(".")[-1]: v for k, v in values.items()}}
            items.append({"name": grid["name_template"].format(**short),
                          "description": grid.get("description", "").format(**short),
                          "variant": vname,
                          "overrides": {**(grid.get("fixed") or {}), **(variants[vname] or {}), **values}})
    runs, seen = [], set()
    for item in items:
        cfg = copy.deepcopy(base)
        for key, value in (item.get("overrides") or {}).items():
            set_dotted(cfg, key, value)
        cfg["experiment"]["name"] = item["name"]
        cfg["experiment"]["variant"] = item.get("variant", "default")
        if item.get("description"):
            cfg["experiment"]["description"] = item["description"]
        if item["name"] in seen:
            raise ValueError(f"Duplicate experiment name in sweep: {item['name']}")
        seen.add(item["name"])
        runs.append((item["name"], cfg))
    return runs


def sweep_default_variants(path):
    """Variants run when none are requested explicitly (None = all)."""
    return load_yaml(path).get("default_variants")


def resolve_path(path, root=REPO_ROOT):
    if path is None:
        return None
    p = Path(os.path.expanduser(str(path)))
    return p if p.is_absolute() else (Path(root) / p)


# --------------------------------------------------------------------------- #
# Files
# --------------------------------------------------------------------------- #
def list_images(directory):
    directory = Path(directory)
    return sorted(p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)


def sha256_of_names(paths):
    h = hashlib.sha256()
    for p in paths:
        h.update(Path(p).name.encode("utf-8") + b"\n")
    return h.hexdigest()


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=_json_default, allow_nan=False)
        f.write("\n")


def _json_default(obj):
    if isinstance(obj, Path):
        return str(obj)
    if hasattr(obj, "item"):  # numpy / torch scalars
        return obj.item()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def write_csv(path, rows, fieldnames=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# --------------------------------------------------------------------------- #
# Environment / provenance
# --------------------------------------------------------------------------- #
def now_iso():
    return _dt.datetime.now(_dt.timezone.utc).astimezone().isoformat(timespec="seconds")


def git_info(root=REPO_ROOT):
    def run(*args):
        try:
            return subprocess.check_output(["git", *args], cwd=root, stderr=subprocess.DEVNULL, text=True).strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    status = run("status", "--porcelain")  # git-ignored outputs (results/, logs/) are excluded
    lines = [] if not status else status.splitlines()
    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
        "tracked_changes": None if status is None else any(not l.startswith("??") for l in lines),
        # Uncommitted code means the commit hash alone does not identify what ran.
        "uncommitted_files": None if status is None else [l[3:] for l in lines[:50]],
    }


def package_versions(names=("torch", "diffusers", "transformers", "accelerate", "xformers", "pyiqa", "numpy", "Pillow")):
    try:
        from importlib.metadata import PackageNotFoundError, version
    except ImportError:  # pragma: no cover
        return {}
    out = {}
    for name in names:
        try:
            out[name] = version(name)
        except PackageNotFoundError:
            out[name] = None
    return out


def environment_info():
    info = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": package_versions(),
        "cuda_available": False,
    }
    try:
        import torch
    except ImportError:
        return info
    info["torch_cuda_build"] = torch.version.cuda
    info["cudnn"] = torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None
    info["cuda_available"] = torch.cuda.is_available()
    if torch.cuda.is_available():
        idx = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(idx)
        info["gpu_name"] = props.name
        info["gpu_total_vram_mb"] = round(props.total_memory / 2**20)
        info["gpu_capability"] = f"{props.major}.{props.minor}"
        free, total = torch.cuda.mem_get_info(idx)
        info["gpu_free_vram_mb_at_start"] = round(free / 2**20)
    return info


FINGERPRINT_BYTES = 16 * 2**20


def fingerprint(path, chunk=FINGERPRINT_BYTES):
    """sha256 of (size, first and last 16 MiB). Cheap identity check for multi-GB checkpoints on Drive;
    detects different or truncated files, but is not a full-content hash."""
    p = Path(path)
    size = p.stat().st_size
    h = hashlib.sha256(str(size).encode())
    with open(p, "rb") as f:
        h.update(f.read(chunk))
        if size > 2 * chunk:
            f.seek(size - chunk)
        h.update(f.read(chunk))
    return h.hexdigest()


def portable_relpath(path, base):
    """`path` relative to `base`, with '/' separators on every platform (for metadata, not for file access).

    Accepts str/Path or any PurePath flavour (e.g. PureWindowsPath), so the result does not depend on the OS
    the run was made on. Raises ValueError if `path` is not under `base`.
    """
    p = path if isinstance(path, PurePath) else Path(path)
    return p.relative_to(base).as_posix()


def file_record(path):
    p = Path(path)
    rec = {"path": str(p), "exists": p.exists()}
    if p.is_file():
        rec["size_bytes"] = p.stat().st_size
        rec["fingerprint"] = fingerprint(p)
    elif p.is_dir():
        files = sorted(f for f in p.rglob("*") if f.is_file())
        rec["n_files"] = len(files)
        rec["size_bytes"] = sum(f.stat().st_size for f in files)
        h = hashlib.sha256()
        for f in files:
            h.update(f"{f.relative_to(p).as_posix()}:{fingerprint(f)}\n".encode())
        rec["fingerprint"] = h.hexdigest()
    return rec
