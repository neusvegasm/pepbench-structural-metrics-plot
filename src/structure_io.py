"""Protein structure parsing and biological chain-role assignment."""
from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations
from pathlib import Path
import re

import numpy as np
from Bio.Align import PairwiseAligner
from Bio.PDB import MMCIFParser, PDBParser

STANDARD_AA3 = {
    "ALA":"A", "ARG":"R", "ASN":"N", "ASP":"D", "CYS":"C", "GLN":"Q",
    "GLU":"E", "GLY":"G", "HIS":"H", "ILE":"I", "LEU":"L", "LYS":"K",
    "MET":"M", "PHE":"F", "PRO":"P", "SER":"S", "THR":"T", "TRP":"W",
    "TYR":"Y", "VAL":"V",
}

# Common residue names encountered in experimental structures and structures
# prepared by Rosetta, Amber, CHARMM, OpenMM, or related workflows. These are
# normalized to the parent sequence letter for receptor-chain correspondence.
# SEC and PYL retain their biological one-letter codes.
COMMON_AA3_ALIASES = {
    # Histidine protonation/tautomer names.
    "HID":"H", "HIE":"H", "HIP":"H", "HSD":"H", "HSE":"H", "HSP":"H",
    # Protonation or charge-state variants.
    "ASH":"D", "GLH":"E", "LYN":"K", "CYM":"C", "CYX":"C",
    # Genetically encoded/noncanonical residues.
    "MSE":"M", "SEC":"U", "PYL":"O", "FME":"M", "NLE":"L",
    # Common post-translational or crystallographic modifications.
    "SEP":"S", "TPO":"T", "PTR":"Y", "TYS":"Y", "HYP":"P", "PCA":"Q",
    "KCX":"K", "LLP":"K", "ALY":"K", "MLY":"K", "MLZ":"K", "M3L":"K",
    "CME":"C", "CSO":"C", "CSD":"C", "CSE":"C", "CSX":"C", "OCS":"C",
    "SCY":"C", "SMC":"C", "CIR":"R",
    # D-amino-acid residue names used by the PDB Chemical Component Dictionary.
    "DAL":"A", "DAR":"R", "DSN":"N", "DAS":"D", "DCY":"C", "DGN":"Q",
    "DGL":"E", "DHI":"H", "DIL":"I", "DLE":"L", "DLY":"K", "MED":"M",
    "DPN":"F", "DPR":"P", "DSG":"S", "DTH":"T", "DTR":"W", "DTY":"Y",
    "DVA":"V",
}

AA3 = {**STANDARD_AA3, **COMMON_AA3_ALIASES}
RECEPTOR_ROLES = ("hla", "tcr_alpha", "tcr_beta")
ALL_ROLES = ("hla", "b2m", "tcr_alpha", "tcr_beta")
BACKBONE_ATOMS = ("N", "CA", "C", "O")


@dataclass
class ResidueRecord:
    chain: str
    key: tuple
    resname: str
    aa: str
    atoms: dict[str, np.ndarray]


@dataclass
class ParsedStructure:
    path: str
    chains: dict[str, list[ResidueRecord]]
    skipped_nonprotein_residues: int


def is_hydrogen(atom_name: str, element: str = "") -> bool:
    if str(element).strip().upper() in {"H", "D"}:
        return True
    return re.sub(r"^\d+", "", str(atom_name).strip()).upper().startswith("H")


def parse_structure(path: str | Path) -> ParsedStructure:
    path = Path(path).expanduser().resolve()
    parser = MMCIFParser(QUIET=True) if path.suffix.lower() in {".cif", ".mmcif"} else PDBParser(QUIET=True)
    model = parser.get_structure(path.stem, str(path))[0]
    chains: dict[str, list[ResidueRecord]] = {}
    skipped = 0
    for chain in model:
        records = []
        for residue in chain:
            resname = residue.get_resname().strip().upper()
            if resname not in AA3 or "CA" not in residue:
                skipped += 1
                continue
            atoms = {}
            for atom in residue:
                if atom.is_disordered() and atom.get_altloc() not in {" ", "A"}:
                    continue
                if is_hydrogen(atom.get_name(), getattr(atom, "element", "")):
                    continue
                atoms[atom.get_name().strip()] = np.asarray(atom.coord, dtype=float)
            records.append(ResidueRecord(str(chain.id), tuple(residue.id), resname, AA3[resname], atoms))
        if records:
            chains[str(chain.id)] = records
    return ParsedStructure(str(path), chains, skipped)


def chain_sequence(structure: ParsedStructure, chain: str) -> str:
    return "".join(residue.aa for residue in structure.chains.get(chain, []))


def alignment_mapping(native_residues: list[ResidueRecord], generated_residues: list[ResidueRecord]) -> dict:
    native = "".join(r.aa for r in native_residues)
    generated = "".join(r.aa for r in generated_residues)
    aligner = PairwiseAligner(mode="global", match_score=2, mismatch_score=-1,
                              open_gap_score=-5, extend_gap_score=-0.5)
    alignment = aligner.align(native, generated)[0]
    pairs = []
    for (n0, n1), (g0, g1) in zip(*alignment.aligned):
        pairs.extend(zip(range(n0, n1), range(g0, g1)))
    identical = sum(native[i] == generated[j] for i, j in pairs)
    return {
        "pairs": pairs,
        "native_count": len(native),
        "generated_count": len(generated),
        "mapped_count": len(pairs),
        "native_coverage": len(pairs) / len(native) if native else np.nan,
        "generated_coverage": len(pairs) / len(generated) if generated else np.nan,
        "identity": identical / len(pairs) if pairs else np.nan,
    }


def infer_native_roles(structure: ParsedStructure, explicit: dict | None = None,
                       peptide_length: int = 9) -> dict[str, str]:
    if explicit:
        missing_roles = set((*ALL_ROLES, "peptide")) - set(explicit)
        missing_chains = set(explicit.values()) - set(structure.chains)
        if missing_roles or missing_chains:
            raise ValueError(f"Invalid native_chain_roles: missing roles={missing_roles}, missing chains={missing_chains}")
        if len(structure.chains[explicit["peptide"]]) != peptide_length:
            raise ValueError("Configured native peptide does not have the expected length")
        return dict(explicit)

    sequences = {chain: chain_sequence(structure, chain) for chain in structure.chains}
    peptides = [chain for chain, seq in sequences.items() if len(seq) == peptide_length]
    hla = [chain for chain, seq in sequences.items() if "SHSMRY" in seq and len(seq) >= 250]
    b2m = [chain for chain, seq in sequences.items() if ("IQRTP" in seq or "MIQRTP" in seq) and 80 <= len(seq) <= 130]
    if len(peptides) != 1 or len(hla) != 1 or len(b2m) != 1:
        raise ValueError(f"Cannot infer native roles: peptide={peptides}, HLA={hla}, b2m={b2m}; provide native_chain_roles")
    remaining = [chain for chain, seq in sequences.items()
                 if chain not in {peptides[0], hla[0], b2m[0]} and len(seq) >= 150]
    if len(remaining) != 2:
        raise ValueError(f"Cannot infer two native TCR chains: {remaining}; provide native_chain_roles")
    remaining.sort(key=lambda chain: len(sequences[chain]))
    return {"peptide": peptides[0], "hla": hla[0], "b2m": b2m[0],
            "tcr_alpha": remaining[0], "tcr_beta": remaining[1]}


def infer_generated_roles(native: ParsedStructure, native_roles: dict[str, str],
                          generated: ParsedStructure, expected_peptide: str) -> tuple[dict, dict]:
    exact_peptides = [chain for chain in generated.chains if chain_sequence(generated, chain) == expected_peptide]
    if len(exact_peptides) != 1:
        nine_mers = [chain for chain, residues in generated.chains.items() if len(residues) == len(expected_peptide)]
        if len(nine_mers) != 1:
            raise ValueError(f"Cannot uniquely identify expected peptide: exact={exact_peptides}, length-matched={nine_mers}")
        exact_peptides = nine_mers
    peptide_chain = exact_peptides[0]
    candidates = [chain for chain in generated.chains if chain != peptide_chain]
    if len(candidates) < 4:
        raise ValueError(f"Expected HLA, b2m, TCR alpha and TCR beta chains; found {candidates}")

    # Assign all four receptor roles jointly by sequence similarity. Generated
    # chain IDs are never assumed to match native IDs.
    best = None
    for chosen in permutations(candidates, 4):
        mappings = {}
        score = 0.0
        for role, chain in zip(ALL_ROLES, chosen):
            mapping = alignment_mapping(native.chains[native_roles[role]], generated.chains[chain])
            mappings[role] = mapping
            score += mapping["identity"] * min(mapping["native_coverage"], mapping["generated_coverage"])
        if best is None or score > best[0]:
            best = (score, chosen, mappings)
    roles = {"peptide": peptide_chain, **dict(zip(ALL_ROLES, best[1]))}
    return roles, best[2]


def native_position_labels(generated: ParsedStructure, roles: dict, mappings: dict) -> dict:
    labels = {}
    for role in RECEPTOR_ROLES:
        mapped = {generated_index: native_index + 1 for native_index, generated_index in mappings[role]["pairs"]}
        chain = roles[role]
        labels[role] = {i: mapped.get(i, f"unmapped_{chain}_{i + 1}")
                        for i in range(len(generated.chains[chain]))}
    return labels
