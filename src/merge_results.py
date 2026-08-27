"""Merge completed metric chunks without dropping QC failures."""
from __future__ import annotations

import argparse
import json

import pandas as pd

from .configuration import load_config


def merge(config: dict) -> pd.DataFrame:
    output = config["output_root"]
    manifest = pd.read_csv(output / "manifests/structural_manifest.tsv", sep="\t")
    chunk_ids = sorted(manifest["chunk_id"].unique())
    metric_paths = [output / "batches" / f"chunk_{int(chunk):05d}.metrics.parquet" for chunk in chunk_ids]
    mapping_paths = [output / "batches" / f"chunk_{int(chunk):05d}.mapping_qc.parquet" for chunk in chunk_ids]
    missing = [path for path in metric_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} metric chunks; first missing: {missing[0]}")

    metrics = pd.concat([pd.read_parquet(path) for path in metric_paths], ignore_index=True)
    expected_ids = set(manifest["row_id"].astype(int))
    observed_ids = set(metrics["row_id"].astype(int))
    if expected_ids != observed_ids or len(metrics) != len(manifest):
        raise ValueError(f"Manifest/result mismatch: expected={len(manifest)}, observed={len(metrics)}, missing={sorted(expected_ids-observed_ids)[:10]}")
    if metrics["row_id"].duplicated().any() or metrics.duplicated(["target", "method", "design_id"]).any():
        raise ValueError("Duplicate per-design rows in merged metrics")
    metrics = metrics.sort_values("row_id").reset_index(drop=True)

    final = output / "final"
    final.mkdir(parents=True, exist_ok=True)
    metrics.to_parquet(final / "structural_metrics_per_design.parquet", index=False)
    metrics.to_csv(final / "structural_metrics_per_design.tsv", sep="\t", index=False)
    existing_mapping_paths = [path for path in mapping_paths if path.is_file()]
    if existing_mapping_paths:
        mappings = pd.concat([pd.read_parquet(path) for path in existing_mapping_paths], ignore_index=True)
        mappings.to_parquet(final / "receptor_mapping_qc.parquet", index=False)
    report = {
        "manifest_rows": len(manifest),
        "merged_rows": len(metrics),
        "qc_pass_rows": int(metrics["qc_pass"].sum()),
        "qc_failed_rows": int((~metrics["qc_pass"].astype(bool)).sum()),
        "top10_flagged_rows": int(metrics["is_top10_af3"].sum()),
        "af3_coordinate_rows": int(metrics["structure_path"].str.endswith((".cif", ".mmcif")).sum()),
    }
    with (final / "merge_qc.json").open("w") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2))
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    merge(load_config(args.config))


if __name__ == "__main__":
    main()

