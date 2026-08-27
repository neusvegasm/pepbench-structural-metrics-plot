"""Rigid alignment and receptor-frame peptide RMSD."""
from __future__ import annotations

import numpy as np

from .structure_io import BACKBONE_ATOMS, RECEPTOR_ROLES, ParsedStructure


def kabsch(moving: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    moving_center = moving.mean(axis=0)
    target_center = target.mean(axis=0)
    u, _, vt = np.linalg.svd((moving - moving_center).T @ (target - target_center))
    correction = np.eye(3)
    correction[-1, -1] = np.sign(np.linalg.det(u @ vt))
    rotation = u @ correction @ vt
    translation = target_center - moving_center @ rotation
    return rotation, translation


def receptor_frame_peptide_rmsd(native: ParsedStructure, native_roles: dict,
                                generated: ParsedStructure, generated_roles: dict,
                                mappings: dict, minimum_alignment_ca: int = 30) -> dict:
    native_ca, generated_ca = [], []
    for role in RECEPTOR_ROLES:
        native_chain = native.chains[native_roles[role]]
        generated_chain = generated.chains[generated_roles[role]]
        for native_index, generated_index in mappings[role]["pairs"]:
            if "CA" in native_chain[native_index].atoms and "CA" in generated_chain[generated_index].atoms:
                native_ca.append(native_chain[native_index].atoms["CA"])
                generated_ca.append(generated_chain[generated_index].atoms["CA"])
    if len(native_ca) < minimum_alignment_ca:
        raise ValueError(f"Only {len(native_ca)} mapped receptor CA atoms; require {minimum_alignment_ca}")

    native_ca = np.asarray(native_ca)
    generated_ca = np.asarray(generated_ca)
    rotation, translation = kabsch(generated_ca, native_ca)
    native_peptide = native.chains[native_roles["peptide"]]
    generated_peptide = generated.chains[generated_roles["peptide"]]
    if len(native_peptide) != len(generated_peptide):
        raise ValueError("Native and generated peptide lengths differ")

    native_backbone, generated_backbone = [], []
    for position, (native_residue, generated_residue) in enumerate(zip(native_peptide, generated_peptide), 1):
        missing = [atom for atom in BACKBONE_ATOMS
                   if atom not in native_residue.atoms or atom not in generated_residue.atoms]
        if missing:
            raise ValueError(f"P{position} missing peptide backbone atoms: {missing}")
        for atom in BACKBONE_ATOMS:
            native_backbone.append(native_residue.atoms[atom])
            generated_backbone.append(generated_residue.atoms[atom])

    transformed = np.asarray(generated_backbone) @ rotation + translation
    target = np.asarray(native_backbone)
    peptide_rmsd = float(np.sqrt(np.mean(np.sum((transformed - target) ** 2, axis=1))))
    aligned_receptor = generated_ca @ rotation + translation
    receptor_rmsd = float(np.sqrt(np.mean(np.sum((aligned_receptor - native_ca) ** 2, axis=1))))
    return {
        "peptide_bb_rmsd": peptide_rmsd,
        "receptor_alignment_rmsd": receptor_rmsd,
        "receptor_alignment_ca_count": len(native_ca),
        "rotation_determinant": float(np.linalg.det(rotation)),
    }

