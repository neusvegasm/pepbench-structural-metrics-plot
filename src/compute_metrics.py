"""Calculate receptor-frame RMSD and contact Jaccards for one manifest chunk."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .configuration import load_config
from .metrics import analyze_complex
from .structure_io import infer_native_roles, parse_structure

BASE_COLUMNS = [
    "row_id", "chunk_id", "target", "method", "design_id", "sequence",
    "structure_path", "native_path", "coordinate_source", "is_top10_af3", "af3_rank",
]
METRIC_COLUMNS = [
    "peptide_bb_rmsd", "jaccard_total", "jaccard_hla", "jaccard_tcr",
    "native_contact_count_total", "generated_contact_count_total", "intersection_count_total", "union_count_total",
    "native_contact_count_hla", "generated_contact_count_hla", "intersection_count_hla", "union_count_hla",
    "native_contact_count_tcr", "generated_contact_count_tcr", "intersection_count_tcr", "union_count_tcr",
    "receptor_alignment_rmsd", "receptor_alignment_ca_count", "rotation_determinant",
]


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def process_chunk(config: dict, manifest: pd.DataFrame, chunk_id: int, force: bool = False) -> Path:
    batch_dir = config["output_root"] / "batches"
    batch_dir.mkdir(parents=True, exist_ok=True)
    output_path = batch_dir / f"chunk_{chunk_id:05d}.metrics.parquet"
    mapping_path = batch_dir / f"chunk_{chunk_id:05d}.mapping_qc.parquet"
    if output_path.is_file() and mapping_path.is_file() and not force:
        print(f"Already complete: {output_path}")
        return output_path

    work = manifest.loc[manifest["chunk_id"].eq(chunk_id)].copy()
    if work.empty:
        raise ValueError(f"Chunk {chunk_id} is absent from the manifest")
    native_cache, native_role_cache = {}, {}
    metric_rows, mapping_rows = [], []
    for row in work.itertuples(index=False):
        base = {column: getattr(row, column) for column in BASE_COLUMNS}
        try:
            if row.target not in native_cache:
                native = parse_structure(row.native_path)
                explicit = config["targets"][row.target].get("native_chain_roles")
                native_cache[row.target] = native
                native_role_cache[row.target] = infer_native_roles(
                    native, explicit, config["analysis"]["peptide_length"]
                )
            generated = parse_structure(row.structure_path)
            metrics, qc = analyze_complex(
                native_cache[row.target], native_role_cache[row.target], generated,
                row.sequence, config["analysis"],
            )
            metric_rows.append({**base, **metrics, "qc_pass": True, "qc_failure_reason": ""})
            mapping_rows.extend({**base, **item, "qc_pass": True, "qc_failure_reason": ""} for item in qc)
        except Exception as error:
            reason = f"{type(error).__name__}:{error}"
            metric_rows.append({**base, "qc_pass": False, "qc_failure_reason": reason})

    metrics_frame = pd.DataFrame(metric_rows)
    for column in METRIC_COLUMNS:
        if column not in metrics_frame:
            metrics_frame[column] = np.nan
    mapping_frame = pd.DataFrame(mapping_rows)
    _atomic_parquet(metrics_frame, output_path)
    _atomic_parquet(mapping_frame, mapping_path)
    print(f"Chunk {chunk_id}: {int(metrics_frame.qc_pass.sum())}/{len(metrics_frame)} passed QC")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--chunk-id", type=int)
    group.add_argument("--all", action="store_true", help="Process every chunk locally")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    manifest_path = config["output_root"] / "manifests/structural_manifest.tsv"
    manifest = pd.read_csv(manifest_path, sep="\t")
    chunk_ids = sorted(manifest["chunk_id"].unique()) if args.all else [args.chunk_id]
    for chunk_id in chunk_ids:
        process_chunk(config, manifest, int(chunk_id), args.force)


if __name__ == "__main__":
    main()

