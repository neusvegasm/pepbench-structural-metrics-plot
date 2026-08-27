# Structural metrics and AF3 Top10 plot

This is a small standalone workflow for calculating receptor-frame peptide
backbone RMSD, peptide–HLA/TCR contact Jaccard, and the final three-panel
RMSD-versus-Jaccard figure. It is intended to be understandable and runnable
without the rest of PepBench. This shared repository contains the production
`config.yaml`; both collaborators must have access to the configured GPFS input
paths.

AF3 coordinates are never used. AF3 ranking tables supply only the identities
of designs highlighted with stars. Every plotted coordinate comes from the
matching OpenMM-repacked `_wm.pdb`.

## Scientific population

For every configured target and method, the workflow discovers:

```text
TARGET/METHOD/04_rosetta_thread/outputs/*.pdb
TARGET/METHOD/05_openmm_wm/outputs/*_wm.pdb
TARGET/METHOD/07_af3/*_af3_ranked.tsv
```

The design ID is the OpenMM filename after removing only `_wm.pdb`. A repacked
PDB enters the background population only if that exact design ID also exists
in the Rosetta-threaded directory. Thus the population is all NetMHCpan-passing
designs with an available repacked structure, before later OpenMM score or
rankability filtering.

The first ten rows of each AF3-ranked TSV define Top10. Mapping to OpenMM uses:

1. exact design ID plus matching peptide sequence;
2. optionally, a unique sequence fallback;
3. rejection of missing, inconsistent, or ambiguous matches.

The complete mapping audit is saved rather than silently dropping failures.

## Metric definitions

Only recognized protein residues containing Cα are parsed. The residue mapping
includes the 20 canonical amino acids; common Rosetta/Amber/CHARMM protonation
names (`HID/HIE/HIP`, `HSD/HSE/HSP`, `ASH`, `GLH`, `LYN`, `CYM/CYX`); common
experimental modifications such as `MSE`, `SEP`, `TPO`, `PTR`, oxidized
cysteines, and modified lysines; and standard PDB D-amino-acid names. These are
normalized to their parent sequence letter for chain correspondence, except
that `SEC` and `PYL` retain `U` and `O`. Hydrogens and deuterium are excluded,
blank or `A` altloc is retained, and non-protein residues are skipped. A listed
residue must still contain Cα, preventing waters, ions, and ordinary ligands
from entering the protein contact set. Native roles may be inferred or declared explicitly.
Generated chain IDs are never trusted: HLA, β2-microglobulin, TCRα and TCRβ are
assigned jointly by global sequence alignment to the native chains. Generated
residues are labeled by their corresponding native sequence positions.

### Contacts

A contact is one unique:

```text
(peptide_position, receptor_role, native_receptor_position)
```

for which any peptide heavy atom lies within 4.5 Å of any receptor heavy atom.
Multiple atom contacts do not multiply the residue-pair count.

- HLA: peptide–HLA-heavy-chain contacts.
- TCR: peptide–TCRα plus peptide–TCRβ contacts.
- Total: union of HLA and TCR contacts.
- β2-microglobulin is excluded from these interfaces.

For native set `N` and generated set `G`:

```text
Jaccard = |N intersect G| / |N union G|
```

If both sets are empty, Jaccard is defined as 1.0 because the two empty sets are
identical. Native/generated/intersection/union counts are retained alongside
each Jaccard so this edge case remains auditable.

### Peptide backbone RMSD

For each generated complex, the workflow:

1. maps generated HLA, TCRα and TCRβ residues to native residues by sequence;
2. Kabsch-aligns their mapped Cα atoms onto the target-specific native receptor;
3. applies exactly that receptor-derived transform to the generated peptide;
4. calculates RMSD over P1–P9 `N`, `CA`, `C`, and `O` atoms.

The peptide is never independently fitted. `peptide_bb_rmsd` therefore measures
peptide placement and shape in the native receptor frame; it is not a
best-fitting peptide-only RMSD.

### QC

Failures remain in `structural_metrics_per_design` with `qc_pass = false` and a
nonempty `qc_failure_reason`. Checks include the expected 9-mer sequence,
mapping coverage and identity of at least 0.95 for HLA/TCRα/TCRβ, at least 30
mapped receptor Cα atoms, complete peptide backbone atoms, and finite primary
metrics. All thresholds are explicit in YAML.

## Installation

```bash
conda env create -f environment.yml
conda activate structural-metrics-plot
```

All commands below are run from the repository root.

## Current configuration

The tracked `config.yaml` currently analyzes:

- targets: `7pbc` and `7pdw`;
- methods: RF1, RF2, RF3, BindCraft, PepGLAD and UniMoMo;
- native structures under `shared/project_peptcr/mhc_i/input`;
- design results under `shared/project_peptcr/manuscript/mhc_i/result`;
- 9-mer peptides, 4.5 Å heavy-atom contacts, 95% mapping coverage and
  identity, and at least 30 receptor Cα alignment atoms;
- batches of 100 structures.

Targets and methods are mappings, so more can be added without changing Python
code. Relative paths are resolved relative to the YAML file. For native
structures that cannot be inferred from standard sequence characteristics,
provide `native_chain_roles` explicitly. Generated roles are still inferred by
sequence and do not inherit those chain IDs.

Because `config.yaml` is shared in this private repository, do not add passwords,
tokens, or credentials to it.

## Run locally

For a small smoke test or a machine where processing all structures locally is
appropriate:

```bash
python -m src.manifest --config config.yaml
python -m src.compute_metrics --config config.yaml --all
python -m src.merge_results --config config.yaml
python -m src.plot_results --config config.yaml
```

The first command is lightweight: it validates input discovery and AF3/OpenMM
identity matching and writes the manifest. The second command performs the
structural calculations and is the expensive step. Slurm is recommended for
the complete production population.

## Slurm

First build the manifest on the login node; this only inspects filenames and
small TSVs:

```bash
mkdir -p logs
python -m src.manifest --config config.yaml
N_CHUNKS=$(($(wc -l < output/manifests/chunks.tsv) - 1))
```

Submit the structural array and a dependent merge/plot job:

```bash
ARRAY_JOB=$(sbatch --parsable --array="0-$((N_CHUNKS-1))%8" \
  --export=ALL,CONFIG="$PWD/config.yaml",PYTHON="$(command -v python)" \
  slurm/compute_metrics_array.sbatch)

sbatch --dependency="afterok:${ARRAY_JOB%%;*}" \
  --export=ALL,CONFIG="$PWD/config.yaml",PYTHON="$(command -v python)" \
  slurm/merge_and_plot.sbatch
```

Adjust partition, memory, time and concurrency at submission if needed. No
Slurm submission is performed automatically by this repository.

After the merge completes, `notebooks/example_plot.ipynb` reads
`config.yaml` and the existing final metric table. It recreates the figure
without parsing PDB files or recomputing RMSD or contacts. To use another YAML
file, set `STRUCTURAL_METRICS_CONFIG` before starting Jupyter.

## Outputs

```text
OUTPUT_ROOT/
├── manifests/
│   ├── structural_manifest.tsv
│   ├── af3_openmm_mapping.tsv
│   ├── population_qc.tsv
│   └── chunks.tsv
├── batches/
│   ├── chunk_*.metrics.parquet
│   └── chunk_*.mapping_qc.parquet
├── final/
│   ├── structural_metrics_per_design.parquet
│   ├── structural_metrics_per_design.tsv
│   ├── receptor_mapping_qc.parquet
│   └── merge_qc.json
└── figures/
    ├── rmsd_vs_contact_jaccard.png
    ├── rmsd_vs_contact_jaccard.pdf
    ├── top10_matching_qc.tsv
    └── unmatched_top10_ids.tsv
```

The plot uses only QC-passing rows. Color identifies method, marker shape
identifies target, and stars identify exact `target + method + design_id`
matches to the AF3 Top10 mapping. The same RMSD scale and the 0–1 Jaccard scale
are used in all panels.
