"""Discover the repacked population and map AF3 ranks to OpenMM design IDs."""
from __future__ import annotations

import argparse
from pathlib import Path
import re

import pandas as pd

from .configuration import condition_dir, load_config, one_match

PEPTIDE_RE = re.compile(r"(?:^|_)([ACDEFGHIKLMNPQRSTVWY]{9})(?:_|$)")


def peptide_from_design_id(design_id: str, peptide_length: int = 9) -> str:
    matches = PEPTIDE_RE.findall(str(design_id))
    matches = [sequence for sequence in matches if len(sequence) == peptide_length]
    if len(matches) != 1:
        raise ValueError(f"Cannot extract one {peptide_length}-mer from design ID {design_id!r}: {matches}")
    return matches[0]


def match_af3_row(af3_id: str, sequence: str, repacked: pd.DataFrame,
                  allow_sequence_fallback: bool = True) -> tuple[pd.Series | None, str]:
    exact = repacked.loc[repacked["design_id"].eq(str(af3_id))]
    if len(exact) == 1:
        row = exact.iloc[0]
        if row["sequence"] != sequence:
            return None, "exact_id_sequence_mismatch"
        return row, "exact_design_id"
    if len(exact) > 1:
        return None, "ambiguous_exact_design_id"
    if not allow_sequence_fallback:
        return None, "no_exact_design_id"
    sequence_matches = repacked.loc[repacked["sequence"].eq(sequence)]
    if len(sequence_matches) == 1:
        return sequence_matches.iloc[0], "unique_sequence_fallback"
    if len(sequence_matches) > 1:
        return None, "ambiguous_sequence_fallback"
    return None, "no_openmm_match"


def build_manifest(config: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    population_rows, mapping_rows, condition_rows = [], [], []
    paths = config["paths"]
    af3_settings = config["af3"]
    peptide_length = config["analysis"]["peptide_length"]

    for target, target_settings in config["targets"].items():
        native_path = Path(target_settings["native_pdb"])
        if not native_path.is_file():
            raise FileNotFoundError(native_path)
        for method in config["methods"]:
            root = condition_dir(config, target, method)
            threaded_paths = sorted(root.glob(paths["threaded_glob"]))
            threaded_ids = {path.stem for path in threaded_paths}
            if not threaded_ids:
                raise FileNotFoundError(f"No Rosetta-threaded PDBs for {target}/{method}: {root / paths['threaded_glob']}")

            repacked_rows = []
            for pdb_path in sorted(root.glob(paths["openmm_glob"])):
                if not pdb_path.name.endswith("_wm.pdb"):
                    continue
                design_id = pdb_path.name[:-len("_wm.pdb")]
                if design_id not in threaded_ids:
                    continue
                repacked_rows.append({
                    "target": target,
                    "method": method,
                    "design_id": design_id,
                    "sequence": peptide_from_design_id(design_id, peptide_length),
                    "structure_path": str(pdb_path.resolve()),
                    "native_path": str(native_path.resolve()),
                    "coordinate_source": "OpenMM repacked/minimized PDB",
                })
            repacked = pd.DataFrame(repacked_rows)
            if repacked.empty:
                raise FileNotFoundError(f"No repacked PDB matched the threaded population for {target}/{method}")
            if repacked["design_id"].duplicated().any():
                raise ValueError(f"Duplicate repacked design IDs for {target}/{method}")

            ranked_path = one_match(root, paths["af3_ranked_glob"])
            ranked = pd.read_csv(ranked_path, sep="\t")
            id_column = af3_settings["design_id_column"]
            peptide_column = af3_settings["peptide_column"]
            missing = {id_column, peptide_column} - set(ranked.columns)
            if missing:
                raise ValueError(f"{ranked_path} lacks AF3 columns: {sorted(missing)}")
            top_n = int(af3_settings["top_n"])
            if len(ranked) < top_n:
                raise ValueError(f"{ranked_path} contains only {len(ranked)} rows; Top{top_n} requires {top_n}")

            local_mapping = []
            for rank, row in enumerate(ranked.iloc[:top_n].itertuples(index=False), 1):
                af3_id = str(getattr(row, id_column))
                sequence = str(getattr(row, peptide_column)).upper()
                if len(sequence) != peptide_length:
                    match, status = None, "invalid_peptide_length"
                else:
                    match, status = match_af3_row(
                        af3_id, sequence, repacked,
                        bool(af3_settings["allow_unique_sequence_fallback"]),
                    )
                local_mapping.append({
                    "target": target,
                    "method": method,
                    "af3_rank": rank,
                    "af3_design_id": af3_id,
                    "sequence": sequence,
                    "matched_design_id": match["design_id"] if match is not None else "",
                    "matched_structure_path": match["structure_path"] if match is not None else "",
                    "mapping_status": status,
                    "af3_ranked_table": str(ranked_path),
                })

            mapping = pd.DataFrame(local_mapping)
            nonempty = mapping["matched_design_id"].ne("")
            duplicated = mapping.loc[nonempty, "matched_design_id"].duplicated(keep=False)
            duplicated_ids = set(mapping.loc[nonempty].loc[duplicated, "matched_design_id"])
            if duplicated_ids:
                mask = mapping["matched_design_id"].isin(duplicated_ids)
                mapping.loc[mask, "mapping_status"] = "duplicate_openmm_match"
                mapping.loc[mask, ["matched_design_id", "matched_structure_path"]] = ""
            mapping_rows.extend(mapping.to_dict("records"))

            rank_by_design = {
                row.matched_design_id: int(row.af3_rank)
                for row in mapping.itertuples(index=False) if row.matched_design_id
            }
            repacked["is_top10_af3"] = repacked["design_id"].isin(rank_by_design)
            repacked["af3_rank"] = repacked["design_id"].map(rank_by_design).astype("Int64")
            population_rows.extend(repacked.to_dict("records"))
            condition_rows.append({
                "target": target,
                "method": method,
                "n_threaded": len(threaded_ids),
                "n_repacked_matched": len(repacked),
                "n_top10_expected": top_n,
                "n_top10_mapped": int(mapping["matched_design_id"].ne("").sum()),
            })

    manifest = pd.DataFrame(population_rows)
    manifest.insert(0, "row_id", range(len(manifest)))
    manifest["chunk_id"] = manifest["row_id"] // int(config["analysis"]["batch_size"])
    if manifest.duplicated(["target", "method", "design_id"]).any():
        raise ValueError("Duplicate target/method/design_id keys in manifest")
    if manifest["structure_path"].str.endswith((".cif", ".mmcif")).any():
        raise AssertionError("AF3 coordinates entered the structural manifest")
    return manifest, pd.DataFrame(mapping_rows), pd.DataFrame(condition_rows)


def write_manifest(config: dict) -> tuple[Path, Path]:
    manifest, mapping, conditions = build_manifest(config)
    output = config["output_root"] / "manifests"
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "structural_manifest.tsv"
    mapping_path = output / "af3_openmm_mapping.tsv"
    manifest.to_csv(manifest_path, sep="\t", index=False)
    mapping.to_csv(mapping_path, sep="\t", index=False)
    conditions.to_csv(output / "population_qc.tsv", sep="\t", index=False)
    chunks = manifest.groupby("chunk_id").agg(
        row_start=("row_id", "min"), row_end=("row_id", "max"), n_structures=("row_id", "size")
    ).reset_index()
    chunks.to_csv(output / "chunks.tsv", sep="\t", index=False)
    print(conditions.to_string(index=False))
    print(f"Wrote {len(manifest)} designs in {len(chunks)} chunks to {manifest_path}")
    return manifest_path, mapping_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    write_manifest(load_config(args.config))


if __name__ == "__main__":
    main()

