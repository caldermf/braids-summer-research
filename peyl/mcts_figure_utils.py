import html
import json
from pathlib import Path


def read_jsonl(path):
    records = []
    path = Path(path)
    if not path.exists():
        return records
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_svg_line_plot(x_values, y_values, title, ylabel, out_path):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    width = 960
    height = 540
    left = 78
    right = 24
    top = 54
    bottom = 62
    plot_w = width - left - right
    plot_h = height - top - bottom

    if not x_values or not y_values:
        return

    x_min = min(x_values)
    x_max = max(x_values)
    y_min = min(y_values)
    y_max = max(y_values)
    if x_min == x_max:
        x_max = x_min + 1
    if y_min == y_max:
        pad = 1 if y_min == 0 else abs(y_min) * 0.1
        y_min -= pad
        y_max += pad

    def x_coord(x):
        return left + (float(x) - x_min) / (x_max - x_min) * plot_w

    def y_coord(y):
        return top + (y_max - float(y)) / (y_max - y_min) * plot_h

    points = " ".join(
        f"{x_coord(x):.2f},{y_coord(y):.2f}"
        for x, y in zip(x_values, y_values)
    )

    grid_lines = []
    tick_labels = []
    for i in range(6):
        frac = i / 5
        y = top + frac * plot_h
        value = y_max - frac * (y_max - y_min)
        grid_lines.append(
            f'<line x1="{left}" y1="{y:.2f}" x2="{width-right}" y2="{y:.2f}" '
            'stroke="#e5e7eb" stroke-width="1" />'
        )
        tick_labels.append(
            f'<text x="{left-10}" y="{y+4:.2f}" text-anchor="end" '
            'font-size="12" fill="#4b5563">'
            f"{value:.3g}</text>"
        )

    for i in range(6):
        frac = i / 5
        x = left + frac * plot_w
        value = x_min + frac * (x_max - x_min)
        grid_lines.append(
            f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{height-bottom}" '
            'stroke="#f3f4f6" stroke-width="1" />'
        )
        tick_labels.append(
            f'<text x="{x:.2f}" y="{height-bottom+24}" text-anchor="middle" '
            'font-size="12" fill="#4b5563">'
            f"{value:.0f}</text>"
        )

    title_escaped = html.escape(title)
    ylabel_escaped = html.escape(ylabel)
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="white"/>
  <text x="{width/2}" y="28" text-anchor="middle" font-size="20" font-family="Arial, sans-serif" fill="#111827">{title_escaped}</text>
  {"".join(grid_lines)}
  <line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" stroke="#111827" stroke-width="1.2"/>
  <line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" stroke="#111827" stroke-width="1.2"/>
  {"".join(tick_labels)}
  <text x="{width/2}" y="{height-16}" text-anchor="middle" font-size="14" font-family="Arial, sans-serif" fill="#111827">Iteration</text>
  <text transform="translate(18 {height/2}) rotate(-90)" text-anchor="middle" font-size="14" font-family="Arial, sans-serif" fill="#111827">{ylabel_escaped}</text>
  <polyline points="{points}" fill="none" stroke="#2563eb" stroke-width="2.2" stroke-linejoin="round" stroke-linecap="round"/>
</svg>
'''
    out_path.write_text(svg, encoding="utf-8")


def write_line_plot(x_values, y_values, title, ylabel, figures_dir, stem, plt=None):
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    if plt is not None:
        plt.figure(figsize=(8, 4.5))
        plt.plot(x_values, y_values, linewidth=1.5)
        plt.title(title)
        plt.xlabel("Iteration")
        plt.ylabel(ylabel)
        plt.grid(True, alpha=0.25)
        plt.tight_layout()
        plt.savefig(figures_dir / f"{stem}.png", dpi=160)
        plt.close()
    else:
        write_svg_line_plot(x_values, y_values, title, ylabel, figures_dir / f"{stem}.svg")


def render_surprise_beam_figures(run_dir, plt=None):
    run_dir = Path(run_dir)
    records = read_jsonl(run_dir / "iterations.jsonl")
    if not records:
        return []

    figures_dir = run_dir / "figures"
    iterations = [record["iteration"] for record in records]
    specs = [
        (
            "best_prefix_projlen",
            "Best reservoir prefix projective length per iteration",
            "Best reservoir prefix projlen",
            "best_prefix_projlen_per_iteration",
        ),
        (
            "best_projlen",
            "Best projective length over time",
            "Best projlen",
            "best_projlen_over_time",
        ),
        ("best_value", "Best value over time", "Best value", "best_value_over_time"),
        (
            "best_prefix_surprise",
            "Best prefix surprise per iteration",
            "Typical projlen minus observed projlen",
            "best_prefix_surprise_per_iteration",
        ),
        (
            "best_prefix_surprise_z",
            "Best prefix surprise z-score per iteration",
            "Surprise z-score",
            "best_prefix_surprise_z_per_iteration",
        ),
        ("path_depth", "Selected tree depth over time", "Tree depth", "selected_depth_over_time"),
        (
            "best_prefix_depth",
            "Best reservoir prefix depth per iteration",
            "Best prefix depth",
            "best_prefix_depth_per_iteration",
        ),
        ("kernel_hits_this_iteration", "Kernel hits per iteration", "Kernel hits", "kernel_hits_per_iteration"),
    ]

    written = []
    for key, title, ylabel, stem in specs:
        if key not in records[0]:
            continue
        y_values = [record[key] for record in records]
        write_line_plot(iterations, y_values, title, ylabel, figures_dir, stem, plt=plt)
        written.append(figures_dir / f"{stem}.{'png' if plt is not None else 'svg'}")
    return written
