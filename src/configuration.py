"""Load and validate the small YAML configuration used by every command."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    with config_path.open() as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Configuration is not a mapping: {config_path}")
    for key in ("data_root", "output_root", "targets", "methods"):
        if key not in config:
            raise ValueError(f"Missing required configuration key: {key}")

    base = config_path.parent
    config["config_path"] = config_path
    config["data_root"] = _resolve(base, config["data_root"])
    config["output_root"] = _resolve(base, config["output_root"])
    for target, settings in config["targets"].items():
        if "native_pdb" not in settings:
            raise ValueError(f"Target {target!r} has no native_pdb")
        settings["native_pdb"] = _resolve(base, settings["native_pdb"])

    config.setdefault("paths", {})
    config["paths"].setdefault("threaded_glob", "04_rosetta_thread/outputs/*.pdb")
    config["paths"].setdefault("openmm_glob", "05_openmm_wm/outputs/*_wm.pdb")
    config["paths"].setdefault("af3_ranked_glob", "07_af3/*_af3_ranked.tsv")
    config.setdefault("af3", {})
    config["af3"].setdefault("design_id_column", "name")
    config["af3"].setdefault("peptide_column", "peptide")
    config["af3"].setdefault("top_n", 10)
    config["af3"].setdefault("allow_unique_sequence_fallback", True)
    config.setdefault("analysis", {})
    defaults = {
        "peptide_length": 9,
        "contact_cutoff_angstrom": 4.5,
        "minimum_mapping_coverage": 0.95,
        "minimum_mapping_identity": 0.95,
        "minimum_receptor_alignment_ca": 30,
        "batch_size": 100,
    }
    for key, value in defaults.items():
        config["analysis"].setdefault(key, value)
    return config


def _resolve(base: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return (base / path).resolve() if not path.is_absolute() else path.resolve()


def condition_dir(config: dict, target: str, method: str) -> Path:
    return config["data_root"] / target / config["methods"][method]


def one_match(root: Path, pattern: str) -> Path:
    matches = sorted(root.glob(pattern))
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one {pattern!r} under {root}; found {len(matches)}")
    return matches[0].resolve()

