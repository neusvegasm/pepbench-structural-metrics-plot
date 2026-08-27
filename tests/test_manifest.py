from __future__ import annotations

import pandas as pd

from src.manifest import match_af3_row, peptide_from_design_id


def repacked():
    return pd.DataFrame([
        {"design_id": "target_model_AAAAAAAAA", "sequence": "AAAAAAAAA", "structure_path": "/a.pdb"},
        {"design_id": "target_model_CCCCCCCCC", "sequence": "CCCCCCCCC", "structure_path": "/c1.pdb"},
        {"design_id": "target_other_CCCCCCCCC", "sequence": "CCCCCCCCC", "structure_path": "/c2.pdb"},
    ])


def test_peptide_sequence_is_parsed_from_design_identity():
    assert peptide_from_design_id("target_model_ACDEFGHIK_0") == "ACDEFGHIK"


def test_exact_af3_match_is_authoritative_and_checks_sequence():
    row, status = match_af3_row("target_model_AAAAAAAAA", "AAAAAAAAA", repacked())
    assert status == "exact_design_id"
    assert row["structure_path"] == "/a.pdb"
    row, status = match_af3_row("target_model_AAAAAAAAA", "GGGGGGGGG", repacked())
    assert row is None and status == "exact_id_sequence_mismatch"


def test_sequence_fallback_rejects_ambiguity():
    row, status = match_af3_row("unknown", "CCCCCCCCC", repacked())
    assert row is None and status == "ambiguous_sequence_fallback"
    row, status = match_af3_row("unknown", "AAAAAAAAA", repacked())
    assert status == "unique_sequence_fallback"
    assert row["design_id"] == "target_model_AAAAAAAAA"

