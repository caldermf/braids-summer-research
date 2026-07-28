from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                rows.append(value)
    return rows


def savefig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = sorted({key for row in rows for key in row})
    lines = [",".join(keys)]
    for row in rows:
        out = []
        for key in keys:
            value = row.get(key, "")
            text = json.dumps(value, separators=(",", ":")) if isinstance(value, (dict, list)) else str(value)
            if "," in text or '"' in text:
                text = '"' + text.replace('"', '""') + '"'
            out.append(text)
        lines.append(",".join(out))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_training(model_root: Path, out_dir: Path) -> list[Path]:
    outputs: list[Path] = []
    histories = sorted(model_root.rglob("history.json")) if model_root.is_dir() else []
    for history_path in histories:
        history = read_json(history_path)
        if not isinstance(history, list) or not history:
            continue
        rel = history_path.parent.relative_to(model_root)
        prefix = out_dir / "training" / rel
        epochs = [int(row.get("epoch", idx + 1)) for idx, row in enumerate(history)]
        metrics = sorted(
            {
                key
                for row in history
                for split in ("train", "validation")
                if isinstance(row.get(split), dict)
                for key in row[split]
                if isinstance(row[split].get(key), (int, float))
            }
        )
        if metrics:
            cols = 2
            rows = math.ceil(len(metrics) / cols)
            plt.figure(figsize=(6.5 * cols, 4.0 * rows))
            for idx, metric in enumerate(metrics, start=1):
                plt.subplot(rows, cols, idx)
                for split, color in (("train", "#1f77b4"), ("validation", "#d62728")):
                    ys = [
                        row.get(split, {}).get(metric)
                        if isinstance(row.get(split), dict)
                        else None
                        for row in history
                    ]
                    xs = [x for x, y in zip(epochs, ys) if isinstance(y, (int, float))]
                    ys2 = [float(y) for y in ys if isinstance(y, (int, float))]
                    if xs:
                        plt.plot(xs, ys2, marker="o", linewidth=1.6, markersize=3, label=split, color=color)
                plt.xlabel("epoch")
                plt.ylabel(metric)
                plt.title(metric)
                plt.grid(alpha=0.25)
                plt.legend()
            path = prefix / "training_curves.png"
            savefig(path)
            outputs.append(path)

        generalization_rows: list[dict[str, Any]] = []
        for row in history:
            epoch = int(row.get("epoch", len(generalization_rows) + 1))
            train = row.get("train") if isinstance(row.get("train"), dict) else {}
            val = row.get("validation") if isinstance(row.get("validation"), dict) else {}
            out = {"epoch": epoch}
            for metric in metrics:
                if metric in train:
                    out[f"train_{metric}"] = train[metric]
                if metric in val:
                    out[f"validation_{metric}"] = val[metric]
                if metric in train and metric in val:
                    out[f"gap_{metric}"] = float(val[metric]) - float(train[metric])
            generalization_rows.append(out)
        write_csv(prefix / "training_history_flat.csv", generalization_rows)

        test_path = history_path.parent / "test_metrics.json"
        if test_path.exists():
            test = read_json(test_path)
            numeric = {k: float(v) for k, v in test.items() if isinstance(v, (int, float))}
            if numeric:
                plt.figure(figsize=(9, 4.8))
                names = list(numeric)
                vals = [numeric[k] for k in names]
                plt.bar(names, vals, color="#4c78a8")
                plt.xticks(rotation=30, ha="right")
                plt.title(f"test metrics: {rel}")
                plt.grid(axis="y", alpha=0.25)
                path = prefix / "test_metrics.png"
                savefig(path)
                outputs.append(path)
    return outputs


def collect_progress(run_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    best_pairs: list[dict[str, Any]] = []
    for path in sorted(run_root.rglob("progress.jsonl")):
        task = str(path.parent.relative_to(run_root))
        for idx, row in enumerate(read_jsonl(path), start=1):
            row2 = dict(row)
            row2["task"] = task
            row2["source_file"] = str(path)
            row2["record_index"] = idx
            rows.append(row2)
            bpl = row.get("best_projlen_by_length")
            if isinstance(bpl, dict):
                for length, projlen in bpl.items():
                    try:
                        best_pairs.append(
                            {
                                "task": task,
                                "source_file": str(path),
                                "record_index": idx,
                                "length": int(length),
                                "projlen": int(projlen),
                                "elapsed_seconds": row.get("elapsed_seconds"),
                                "phase": row.get("phase"),
                            }
                        )
                    except (TypeError, ValueError):
                        pass
            length = row.get("total_length", row.get("processed_length", row.get("next_length")))
            projlen = row.get("best_projlen", row.get("beam_best_projlen"))
            if isinstance(length, int) and isinstance(projlen, int):
                best_pairs.append(
                    {
                        "task": task,
                        "source_file": str(path),
                        "record_index": idx,
                        "length": int(length),
                        "projlen": int(projlen),
                        "elapsed_seconds": row.get("elapsed_seconds"),
                        "phase": row.get("phase"),
                    }
                )
            summaries = row.get("bucket_summaries")
            if isinstance(summaries, dict):
                for heuristic, summary in summaries.items():
                    if not isinstance(summary, dict):
                        continue
                    keys = summary.get("best_keys")
                    if not isinstance(keys, list) or not keys:
                        continue
                    first = keys[0]
                    key = first.get("key") if isinstance(first, dict) else None
                    if isinstance(key, list) and len(key) >= 2:
                        try:
                            best_pairs.append(
                                {
                                    "task": task,
                                    "source_file": str(path),
                                    "record_index": idx,
                                    "length": int(key[0]),
                                    "projlen": int(key[1]),
                                    "elapsed_seconds": row.get("elapsed_seconds"),
                                    "phase": "bucket_summary",
                                    "heuristic": heuristic,
                                }
                            )
                        except (TypeError, ValueError):
                            pass
    return rows, best_pairs


def aggregate_min_median(pairs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_length: dict[int, list[int]] = defaultdict(list)
    for row in pairs:
        if isinstance(row.get("length"), int) and isinstance(row.get("projlen"), int):
            by_length[row["length"]].append(row["projlen"])
    return [
        {
            "length": length,
            "min_projlen": min(vals),
            "median_projlen": median(vals),
            "max_projlen": max(vals),
            "count": len(vals),
        }
        for length, vals in sorted(by_length.items())
    ]


def plot_search_run(run_root: Path, out_dir: Path) -> list[Path]:
    outputs: list[Path] = []
    rows, pairs = collect_progress(run_root)
    if not rows:
        return outputs
    run_name = run_root.name
    prefix = out_dir / "search" / run_name
    write_csv(prefix / "progress_flat.csv", rows)
    write_csv(prefix / "best_projlen_pairs.csv", pairs)
    agg = aggregate_min_median(pairs)
    write_csv(prefix / "best_projlen_by_length.csv", agg)

    if agg:
        plt.figure(figsize=(10, 5.5))
        xs = [r["length"] for r in agg]
        plt.plot(xs, [r["min_projlen"] for r in agg], label="min across tasks", linewidth=2.2)
        plt.plot(xs, [r["median_projlen"] for r in agg], label="median", linewidth=1.6)
        plt.plot(xs, [r["max_projlen"] for r in agg], label="max", linewidth=1.0, alpha=0.7)
        plt.xlabel("Garside length")
        plt.ylabel("best projlen")
        plt.title(f"best projlen by length: {run_name}")
        plt.grid(alpha=0.25)
        plt.legend()
        path = prefix / "best_projlen_by_length.png"
        savefig(path)
        outputs.append(path)

    numeric_series = [
        ("live_braids", "live braids"),
        ("exact_evaluations", "exact evaluations"),
        ("expanded_states", "expanded states"),
        ("bucket_count", "bucket count"),
        ("selected_braids", "selected braids"),
        ("near_kernel_candidates", "near-kernel candidates"),
        ("scalar_identity_candidates", "scalar identity candidates"),
        ("target_match_candidates", "target match candidates"),
    ]
    for key, label in numeric_series:
        points = []
        for row in rows:
            x = row.get("total_length", row.get("processed_length", row.get("next_length")))
            y = row.get(key)
            if isinstance(x, int) and isinstance(y, (int, float)):
                points.append((x, float(y), row["task"]))
        if not points:
            continue
        by_length: dict[int, list[float]] = defaultdict(list)
        for x, y, _task in points:
            by_length[x].append(y)
        xs = sorted(by_length)
        plt.figure(figsize=(10, 5.5))
        plt.plot(xs, [median(by_length[x]) for x in xs], label="median", linewidth=2)
        plt.plot(xs, [min(by_length[x]) for x in xs], label="min", linewidth=1.2, alpha=0.8)
        plt.plot(xs, [max(by_length[x]) for x in xs], label="max", linewidth=1.2, alpha=0.8)
        plt.xlabel("Garside length")
        plt.ylabel(label)
        plt.title(f"{label} by length: {run_name}")
        plt.grid(alpha=0.25)
        plt.legend()
        path = prefix / f"{key}_by_length.png"
        savefig(path)
        outputs.append(path)

    heuristic_rows = [
        row for row in pairs if row.get("phase") == "bucket_summary" and isinstance(row.get("heuristic"), str)
    ]
    if heuristic_rows:
        by_h: dict[str, dict[int, list[int]]] = defaultdict(lambda: defaultdict(list))
        for row in heuristic_rows:
            by_h[row["heuristic"]][row["length"]].append(row["projlen"])
        plt.figure(figsize=(10, 5.5))
        for heuristic, by_length in sorted(by_h.items()):
            xs = sorted(by_length)
            ys = [min(by_length[x]) for x in xs]
            plt.plot(xs, ys, label=heuristic, linewidth=1.7)
        plt.xlabel("Garside length")
        plt.ylabel("best bucket score / projlen-like key")
        plt.title(f"heuristic bucket fronts: {run_name}")
        plt.grid(alpha=0.25)
        plt.legend()
        path = prefix / "heuristic_bucket_fronts.png"
        savefig(path)
        outputs.append(path)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot last-factor confusion and reservoir search reports")
    parser.add_argument("--model-root", action="append", default=[])
    parser.add_argument("--run-root", action="append", default=[])
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    outputs: list[Path] = []
    for root in args.model_root:
        outputs.extend(plot_training(Path(root), out_dir))
    for root in args.run_root:
        outputs.extend(plot_search_run(Path(root), out_dir))
    manifest = {
        "plots": [str(path) for path in outputs],
        "plot_count": len(outputs),
    }
    (out_dir / "plot_manifest.json").parent.mkdir(parents=True, exist_ok=True)
    (out_dir / "plot_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
