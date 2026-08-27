from __future__ import annotations

import numpy as np

from src.alignment import receptor_frame_peptide_rmsd
from src.contacts import jaccard, residue_contact_set
from src.structure_io import (AA3, BACKBONE_ATOMS, STANDARD_AA3, ParsedStructure,
                              ResidueRecord, is_hydrogen)


def residue(chain, index, aa="A", x=0.0, y=None):
    y = index * 3.0 if y is None else y
    atoms = {atom: np.array([x, y, offset * 0.1])
             for offset, atom in enumerate(BACKBONE_ATOMS)}
    atoms["CB"] = np.array([x, y, 0.8])
    return ResidueRecord(chain, (" ", index + 1, " "), "ALA", aa, atoms)


def test_hydrogen_detection_includes_digit_prefixed_names():
    assert all(is_hydrogen(name) for name in ("H", "HA", "HB2", "1H", "2HB", "3HG2"))
    assert not is_hydrogen("CA")


def test_standard_and_common_modified_residue_names_are_normalized():
    assert len(STANDARD_AA3) == 20
    expected = {
        "HSD": "H", "HSE": "H", "HSP": "H",
        "ASH": "D", "GLH": "E", "LYN": "K", "CYM": "C", "CYX": "C",
        "MSE": "M", "SEP": "S", "TPO": "T", "PTR": "Y",
        "KCX": "K", "MLY": "K", "CSO": "C", "DLE": "L",
    }
    assert {name: AA3[name] for name in expected} == expected


def test_contact_cutoff_and_residue_pair_uniqueness():
    peptide = [residue("P", index, x=100.0) for index in range(9)]
    peptide[0].atoms = {"CA": np.array([0.0, 0.0, 0.0]), "CB": np.array([0.0, 0.0, 0.1])}
    near = residue("A", 0, x=4.499, y=0.0)
    far = residue("A", 1, x=4.501, y=0.0)
    near.atoms = {"CA": np.array([4.499, 0, 0]), "CB": np.array([4.499, 0, 0.1])}
    far.atoms = {"CA": np.array([4.501, 0, 0])}
    structure = ParsedStructure("synthetic", {"P": peptide, "A": [near, far]}, 0)
    contacts = residue_contact_set(structure, {"peptide": "P", "hla": "A"},
                                   ("hla",), {"hla": {0: 63, 1: 64}}, 4.5)
    assert contacts == {(1, "hla", 63)}


def test_jaccard_and_empty_set_behavior():
    comparison = jaccard({(1, "hla", 1), (2, "hla", 2)},
                         {(2, "hla", 2), (3, "hla", 3)})
    assert comparison["intersection_count"] == 1
    assert comparison["union_count"] == 3
    assert comparison["jaccard"] == 1 / 3
    assert jaccard(set(), set())["jaccard"] == 1.0


def test_receptor_frame_rmsd_is_invariant_to_complex_rigid_transform():
    roles = {"peptide": "P", "hla": "A", "tcr_alpha": "D", "tcr_beta": "E"}
    chains = {chain: [residue(chain, index, x=x) for index in range(12)]
              for chain, x in (("A", 0), ("D", 10), ("E", 20))}
    chains["P"] = [residue("P", index, x=5) for index in range(9)]
    native = ParsedStructure("native", chains, 0)
    theta = 0.7
    rotation = np.array([[np.cos(theta), -np.sin(theta), 0],
                         [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
    translation = np.array([5.0, -3.0, 2.0])
    moved = {
        chain: [ResidueRecord(item.chain, item.key, item.resname, item.aa,
                              {atom: xyz @ rotation + translation for atom, xyz in item.atoms.items()})
                for item in residues]
        for chain, residues in chains.items()
    }
    generated = ParsedStructure("generated", moved, 0)
    mappings = {role: {"pairs": list(zip(range(12), range(12)))}
                for role in ("hla", "tcr_alpha", "tcr_beta")}
    result = receptor_frame_peptide_rmsd(native, roles, generated, roles, mappings, 30)
    assert result["peptide_bb_rmsd"] < 1e-10
    assert result["receptor_alignment_rmsd"] < 1e-10
    assert result["receptor_alignment_ca_count"] == 36
