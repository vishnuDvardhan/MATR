#!/usr/bin/env python3
"""
Compare MATR eval logs across runs.

Usage:
  # Compare by result directory (auto-finds eval.log.txt):
  python compare_runs.py results/mr-sportsmr/sportsmoment-imagebind-3-* results/mr-sportsmr/sportsmoment-imagebind-4-*

  # With custom labels:
  python compare_runs.py \
    results/mr-sportsmr/sportsmoment-imagebind-3-imagebind-imagebind-2026_04_06_08:"IB-3 lr=1e-4 saliency=ON" \
    results/mr-sportsmr/sportsmoment-imagebind-4-imagebind-imagebind-2026_04_06_17:"IB-4 lr=5e-5 saliency=OFF"

  # Or pass eval.log.txt paths directly:
  python compare_runs.py path/to/run1/eval.log.txt:"Run 1" path/to/run2/eval.log.txt:"Run 2"
"""

import sys
import os
import json
import glob


def find_eval_log(path):
    if path.endswith("eval.log.txt") and os.path.isfile(path):
        return path
    candidate = os.path.join(path, "eval.log.txt")
    if os.path.isfile(candidate):
        return candidate
    # glob in case path has wildcard remnants
    matches = glob.glob(os.path.join(path, "**", "eval.log.txt"), recursive=True)
    if matches:
        return sorted(matches)[0]
    return None


def parse_log(log_path):
    rows = []
    with open(log_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                meta, metrics_json = line.split(" [Metrics] ", 1)
                metrics = json.loads(metrics_json)["brief"]
                epoch = meta.split("[Epoch] ")[1].split(" ")[0]
                loss_str = meta.split("[Loss] ")[1]
                toks = loss_str.split()
                losses = {toks[i]: float(toks[i + 1]) for i in range(0, len(toks) - 1, 2)}
                rows.append((epoch, losses, metrics))
            except Exception:
                continue
    return rows


def make_label(path):
    """Auto-generate a label from the directory name."""
    name = os.path.basename(path.rstrip("/"))
    # strip trailing date like -2026_04_06_08
    parts = name.rsplit("-", 2)
    if len(parts) == 3 and parts[2].replace("_", "").isdigit():
        name = parts[0]
    return name


def print_run(label, rows):
    print(f"\n=== {label} ===")
    header = f"{'Epoch':>6} | {'loss_f':>7} {'loss_g':>7} {'loss_s':>7} | {'R1@0.5':>7} {'R1@0.7':>7} {'mAP':>7} {'mIoU':>7}"
    print(header)
    print("-" * len(header))
    for epoch, losses, m in rows:
        loss_s = losses.get("loss_s_inter", 0) + losses.get("loss_s_intra", 0)
        r1_05 = m.get("MR-full-R1@0.5-key", 0)
        r1_07 = m.get("MR-full-R1@0.7-key", 0)
        mAP   = m.get("MR-full-mAP-key", 0)
        mIoU  = m.get("MR-full-mIoU-key", 0)
        print(f"{epoch:>6} | {losses['loss_f']:>7.4f} {losses['loss_g']:>7.4f} {loss_s:>7.4f} | "
              f"{r1_05:>7.2f} {r1_07:>7.2f} {mAP:>7.2f} {mIoU:>7.2f}")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    runs = []
    for arg in sys.argv[1:]:
        if ":" in arg:
            path, label = arg.split(":", 1)
        else:
            path = arg
            label = make_label(path)

        log_path = find_eval_log(path)
        if log_path is None:
            print(f"WARNING: could not find eval.log.txt in '{path}', skipping.")
            continue
        runs.append((label, log_path))

    if not runs:
        print("No valid runs found.")
        sys.exit(1)

    for label, log_path in runs:
        rows = parse_log(log_path)
        if not rows:
            print(f"\n=== {label} ===\n  (no eval entries yet)")
        else:
            print_run(label, rows)


if __name__ == "__main__":
    main()
