from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/braids_diagnosis_matplotlib")


def render_plots(rows: list[dict], output_dir: Path) -> None:
    if not rows:
        return
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    figures = output_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    depths = [row["depth"] for row in rows]

    def line(values, title, ylabel, filename, log_y=False):
        plt.figure(figsize=(9, 4.8))
        plt.plot(depths, values, linewidth=1.8)
        plt.title(title)
        plt.xlabel("Garside depth")
        plt.ylabel(ylabel)
        if log_y:
            plt.yscale("log")
        plt.grid(alpha=0.25)
        plt.tight_layout()
        plt.savefig(figures / filename, dpi=170)
        plt.close()

    line(
        [row["target_projlen"] for row in rows],
        "Known-kernel prefix projective length",
        "projlen",
        "target_projlen.png",
    )
    line(
        [max(1, row["paper_bucket_arrivals"]) for row in rows],
        "Target bucket arrival count",
        "arrivals",
        "paper_bucket_arrivals.png",
        log_y=True,
    )
    line(
        [max(1e-300, row["paper_step_survival_probability"]) for row in rows],
        "Paper-policy step survival probability",
        "probability",
        "paper_step_survival_probability.png",
        log_y=True,
    )
    line(
        [
            row["paper_cumulative_log10_survival"]
            if row["paper_cumulative_log10_survival"] is not None
            else -300.0
            for row in rows
        ],
        "Cumulative paper-policy survival probability",
        "log10 probability",
        "paper_cumulative_log10_survival.png",
    )
    line(
        [row["mcts_value_best_rank"] for row in rows],
        "Known prefix rank under surprise-MCTS value",
        "best rank",
        "mcts_value_rank.png",
        log_y=True,
    )

    plt.figure(figsize=(9, 4.8))
    for key, label in (
        ("crispr_endpoint_estimated_population_rank", "endpoint"),
        ("crispr_envelope_estimated_population_rank", "envelope"),
        ("crispr_collapse_estimated_population_rank", "collapse"),
        ("crispr_suffix_estimated_population_rank", "suffix"),
    ):
        values = [max(1, int(row.get(key, 1))) for row in rows]
        plt.plot(depths, values, label=label, linewidth=1.4)
    plt.yscale("log")
    plt.title("Estimated known-prefix rank in CRISPR V4 population")
    plt.xlabel("Garside depth")
    plt.ylabel("estimated population rank")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures / "crispr_sample_ranks.png", dpi=170)
    plt.close()
