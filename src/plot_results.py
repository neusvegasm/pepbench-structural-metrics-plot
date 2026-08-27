"""Create one three-panel RMSD-versus-Jaccard figure per design method."""
from __future__ import annotations

import argparse
from itertools import cycle

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import seaborn as sns

from .configuration import load_config

DEFAULT_METHOD_COLORS = {
    "rf1": "red", "rf2": "black", "rf3": "green",
    "pepglad": "blue", "bindcraft": "#E6B800", "unimomo": "pink",
}
METHOD_LABELS = {
    "rf1": "RF1", "rf2": "RF2", "rf3": "RF3",
    "bindcraft": "BindCraft", "pepglad": "PepGLAD", "unimomo": "UniMoMo",
}
MARKERS = ["o", "^", "s", "D", "P", "v", "X"]


def prepare_plot_data(config: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    output = config["output_root"]
    metrics = pd.read_parquet(output / "final/structural_metrics_per_design.parquet")
    population = metrics.loc[metrics["qc_pass"].fillna(False).astype(bool)].copy()
    if population.empty:
        raise ValueError("No QC-passing structures are available for plotting")
    if not population["coordinate_source"].eq("OpenMM repacked/minimized PDB").all():
        raise AssertionError("Plot population contains a non-OpenMM coordinate source")
    if population["structure_path"].str.endswith((".cif", ".mmcif")).any():
        raise AssertionError("AF3 coordinates entered the plot population")

    mapping = pd.read_csv(output / "manifests/af3_openmm_mapping.tsv", sep="\t", keep_default_na=False)
    expected = mapping[["target", "method", "af3_rank", "af3_design_id", "matched_design_id", "mapping_status"]].copy()
    available = population[["target", "method", "design_id"]].drop_duplicates()
    audit = expected.merge(
        available.rename(columns={"design_id": "matched_design_id"}).assign(metric_row_available=True),
        on=["target", "method", "matched_design_id"], how="left", validate="many_to_one",
    )
    valid_status = audit["mapping_status"].isin(["exact_design_id", "unique_sequence_fallback"])
    audit["matched"] = valid_status & audit["metric_row_available"].fillna(False).astype(bool)
    expected_counts = audit.groupby(["target", "method"]).size().rename("n_top10_expected")
    matched_counts = audit.loc[audit["matched"]].groupby(["target", "method"]).size().rename("n_top10_matched")
    qc = pd.concat([expected_counts, matched_counts], axis=1).fillna(0).astype(int).reset_index()
    unmatched = audit.loc[~audit["matched"]].drop(columns="metric_row_available")

    star_keys = audit.loc[audit["matched"], ["target", "method", "matched_design_id"]].rename(
        columns={"matched_design_id": "design_id"}
    )
    star_index = pd.MultiIndex.from_frame(star_keys)
    population["is_top10_af3"] = pd.MultiIndex.from_frame(
        population[["target", "method", "design_id"]]
    ).isin(star_index)
    return population, qc, unmatched


def _create_method_figure(config: dict, population: pd.DataFrame, method: str,
                          rmsd_limit: float):
    """Plot Total/HLA/TCR panels for one method using shared global limits."""
    methods = list(config["methods"])
    if method not in methods:
        raise ValueError(f"Unknown method {method!r}; expected one of {methods}")
    targets = list(config["targets"])
    palette = sns.color_palette("tab10", max(len(methods), 1))
    method_colors = {method: DEFAULT_METHOD_COLORS.get(method, palette[index])
                     for index, method in enumerate(methods)}
    target_markers = dict(zip(targets, cycle(MARKERS)))
    panels = [("Total contacts", "jaccard_total"),
              ("MHC contacts", "jaccard_hla"),
              ("TCR contacts", "jaccard_tcr")]

    sns.set_theme(style="whitegrid", context="notebook")
    figure, axes = plt.subplots(1, 3, figsize=(18, 5.8), sharex=True, sharey=True)
    method_population = population.loc[population["method"].eq(method)].copy()
    if method_population.empty:
        raise ValueError(f"No QC-passing structures are available for method {method!r}")
    for axis, (title, metric) in zip(axes, panels):
        for target in targets:
            group = method_population.loc[method_population["target"].eq(target)]
            axis.scatter(group["peptide_bb_rmsd"], group[metric], color=method_colors[method],
                         marker=target_markers[target], s=20, alpha=0.27,
                         edgecolors="none", zorder=1)
        selected = method_population.loc[method_population["is_top10_af3"]]
        axis.scatter(selected["peptide_bb_rmsd"], selected[metric], color=method_colors[method],
                     marker="*", s=210, alpha=1, edgecolors="black",
                     linewidths=1, zorder=10)
        axis.set(title=title, xlabel="Peptide backbone RMSD to native (Å)", ylim=(0, 1))

    for axis in axes:
        axis.set_xlim(0, rmsd_limit)
    axes[0].set_ylabel("Native-contact Jaccard")
    method_label = METHOD_LABELS.get(method, method)
    figure.suptitle(
        f"{method_label}: peptide RMSD versus native-contact recovery\n"
        "Top10 AF3-selected designs highlighted with stars",
        fontsize=16, y=1.04,
    )

    target_handles = [Line2D([0], [0], marker=target_markers[target], linestyle="None", markersize=8,
                             markerfacecolor=method_colors[method], markeredgecolor="black", label=target.upper())
                      for target in targets]
    top10_handle = Line2D([0], [0], marker="*", linestyle="None", markersize=14,
                          markerfacecolor=method_colors[method], markeredgecolor="black", label="Top10 AF3")
    figure.legend(handles=[*target_handles, top10_handle], loc="lower center",
                  bbox_to_anchor=(0.5, -0.08), ncol=len(target_handles) + 1)
    figure.tight_layout(rect=[0, 0.04, 1, 1])
    return figure


def create_method_figures(config: dict):
    """Return one comparable three-panel figure for every configured method."""
    population, qc, unmatched = prepare_plot_data(config)
    maximum = float(population["peptide_bb_rmsd"].max())
    rmsd_limit = max(2.0, float(np.ceil(maximum / 2) * 2))
    figures = {
        method: _create_method_figure(config, population, method, rmsd_limit)
        for method in config["methods"]
    }
    return figures, qc, unmatched, population


def save_plot(config: dict) -> None:
    figures, qc, unmatched, population = create_method_figures(config)
    figure_dir = config["output_root"] / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    qc.to_csv(figure_dir / "top10_matching_qc.tsv", sep="\t", index=False)
    unmatched.to_csv(figure_dir / "unmatched_top10_ids.tsv", sep="\t", index=False)
    for method, figure in figures.items():
        stem = f"rmsd_vs_contact_jaccard_{method}"
        figure.savefig(figure_dir / f"{stem}.png", dpi=300, bbox_inches="tight")
        figure.savefig(figure_dir / f"{stem}.pdf", bbox_inches="tight")
        plt.close(figure)
    print(qc.to_string(index=False))
    print(f"Background designs: {len(population)}; Top10 stars: {int(population.is_top10_af3.sum())}; unmatched: {len(unmatched)}")
    print(f"Saved {len(figures)} method-specific figures under {figure_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    save_plot(load_config(args.config))


if __name__ == "__main__":
    main()
