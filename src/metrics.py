"""Combine role mapping, contact comparison, RMSD and structural QC."""
from __future__ import annotations

import numpy as np

from .alignment import receptor_frame_peptide_rmsd
from .contacts import jaccard, residue_contact_set
from .structure_io import (RECEPTOR_ROLES, ParsedStructure, infer_generated_roles,
                           native_position_labels)


def analyze_complex(native: ParsedStructure, native_roles: dict, generated: ParsedStructure,
                    expected_peptide: str, settings: dict) -> tuple[dict, list[dict]]:
    roles, mappings = infer_generated_roles(native, native_roles, generated, expected_peptide)
    mapping_qc = []
    failures = []
    for role in RECEPTOR_ROLES:
        mapping = mappings[role]
        row = {"receptor_role": role, "generated_chain": roles[role],
               **{key: value for key, value in mapping.items() if key != "pairs"}}
        mapping_qc.append(row)
        if min(row["native_coverage"], row["generated_coverage"]) < settings["minimum_mapping_coverage"]:
            failures.append(f"{role}_mapping_coverage")
        if row["identity"] < settings["minimum_mapping_identity"]:
            failures.append(f"{role}_mapping_identity")

    observed_peptide = "".join(r.aa for r in generated.chains[roles["peptide"]])
    if len(observed_peptide) != settings["peptide_length"]:
        failures.append("generated_peptide_length")
    if observed_peptide != expected_peptide:
        failures.append("peptide_sequence_mismatch")
    if failures:
        raise ValueError(";".join(failures))

    native_labels = {role: {i: i + 1 for i in range(len(native.chains[native_roles[role]]))}
                     for role in RECEPTOR_ROLES}
    generated_labels = native_position_labels(generated, roles, mappings)
    cutoff = settings["contact_cutoff_angstrom"]
    native_hla = residue_contact_set(native, native_roles, ("hla",), native_labels, cutoff)
    native_tcr = residue_contact_set(native, native_roles, ("tcr_alpha", "tcr_beta"), native_labels, cutoff)
    generated_hla = residue_contact_set(generated, roles, ("hla",), generated_labels, cutoff)
    generated_tcr = residue_contact_set(generated, roles, ("tcr_alpha", "tcr_beta"), generated_labels, cutoff)

    contact_sets = {
        "hla": (native_hla, generated_hla),
        "tcr": (native_tcr, generated_tcr),
        "total": (native_hla | native_tcr, generated_hla | generated_tcr),
    }
    metrics = {}
    for interface, (native_set, generated_set) in contact_sets.items():
        comparison = jaccard(native_set, generated_set)
        for name, value in comparison.items():
            metrics[f"{name}_{interface}"] = value

    metrics.update(receptor_frame_peptide_rmsd(
        native, native_roles, generated, roles, mappings,
        minimum_alignment_ca=settings["minimum_receptor_alignment_ca"],
    ))
    required = [metrics["peptide_bb_rmsd"], metrics["jaccard_total"],
                metrics["jaccard_hla"], metrics["jaccard_tcr"]]
    if not np.isfinite(required).all():
        raise ValueError("nonfinite_primary_metric")
    return metrics, mapping_qc

