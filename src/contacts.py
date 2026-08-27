"""All-heavy-atom peptide/receptor residue contacts and Jaccard similarity."""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from .structure_io import ParsedStructure


def residue_contact_set(structure: ParsedStructure, roles: dict, receptor_roles: tuple[str, ...],
                        labels: dict, cutoff: float = 4.5) -> set[tuple]:
    peptide = structure.chains[roles["peptide"]]
    receptor_xyz, receptor_keys = [], []
    for role in receptor_roles:
        chain = roles[role]
        for index, residue in enumerate(structure.chains[chain]):
            for coordinate in residue.atoms.values():
                receptor_xyz.append(coordinate)
                receptor_keys.append((role, labels[role][index]))
    if not receptor_xyz:
        return set()
    tree = cKDTree(np.asarray(receptor_xyz))
    contacts = set()
    for peptide_position, residue in enumerate(peptide, 1):
        peptide_atoms = np.asarray(list(residue.atoms.values()))
        for nearby_atom_indices in tree.query_ball_point(peptide_atoms, cutoff):
            for receptor_atom_index in nearby_atom_indices:
                role, native_position = receptor_keys[receptor_atom_index]
                contacts.add((peptide_position, role, native_position))
    return contacts


def jaccard(native: set, generated: set) -> dict:
    intersection = len(native & generated)
    union = len(native | generated)
    return {
        "jaccard": intersection / union if union else 1.0,
        "native_contact_count": len(native),
        "generated_contact_count": len(generated),
        "intersection_count": intersection,
        "union_count": union,
    }

