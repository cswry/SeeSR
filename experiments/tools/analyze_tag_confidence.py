"""Empirical check of Component A's confidence score c_i (no calibration is assumed).

For every DAPE tag that passed the upstream thresholds on the LR input (semantic_confidence.jsonl, written
by runs with semantic_confidence enabled), check whether the original RAM model also predicts it on the
HR image (ref_tags column of eval/semantic_per_image.csv, written by evaluation/semantic_tags.py).
Reports agreement rate per confidence bin and for tags kept / dropped by a given c_min.

RAM-on-HR tags are a model-based reference, not ground truth, and DAPE is distilled from RAM, so this
measures agreement, not correctness. If agreement does not rise with c_i, the confidence rule has no
empirical support on that dataset.

  python experiments/tools/analyze_tag_confidence.py --run <RESULTS>/RealSR/semantic_conf_ddpm50_s42 \\
      --reference <RESULTS>/RealSR/semantic_conf_ddpm50_s42/eval/semantic_per_image.csv --min_confidence 0.731
"""
import argparse
import csv
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.common import now_iso, write_csv, write_json  # noqa: E402

BIN_EDGES = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0000001]


def load_records(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_reference(path):
    with open(path, newline="", encoding="utf-8") as f:
        return {r["image"]: {t.strip() for t in r["ref_tags"].split(",") if t.strip()} for r in csv.DictReader(f)}


def analyse(records, reference, min_confidence):
    rows, kept, dropped, missing = [], [0, 0], [0, 0], []
    bins = [[0, 0] for _ in BIN_EDGES[:-1]]
    for rec in records:
        ref = reference.get(rec["image"])
        if ref is None:
            missing.append(rec["image"])
            continue
        for tag, p, tau, c in rec["tag_scores"]:
            hit = tag in ref
            for k in range(len(bins)):
                if BIN_EDGES[k] <= c < BIN_EDGES[k + 1]:
                    bins[k][0] += hit
                    bins[k][1] += 1
            target = kept if c > min_confidence else dropped
            target[0] += hit
            target[1] += 1
    for k, (hits, n) in enumerate(bins):
        rows.append({"c_low": BIN_EDGES[k], "c_high": min(BIN_EDGES[k + 1], 1.0), "n_tags": n,
                     "agreement_with_ram_hr": hits / n if n else None})
    rate = lambda hn: hn[0] / hn[1] if hn[1] else None  # noqa: E731
    return rows, {"min_confidence": min_confidence, "kept": {"n": kept[1], "agreement": rate(kept)},
                  "dropped": {"n": dropped[1], "agreement": rate(dropped)}, "images_without_reference": missing}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True, help="run folder containing semantic_confidence.jsonl")
    p.add_argument("--reference", required=True, help="semantic_per_image.csv with a ref_tags column")
    p.add_argument("--min_confidence", type=float, default=0.7310585786300049)
    p.add_argument("--output_dir", help="default: <run>/eval")
    a = p.parse_args(argv)
    jsonl = Path(a.run) / "semantic_confidence.jsonl"
    if not jsonl.is_file():
        print(f"ERROR: {jsonl} not found (run a variant with semantic_confidence enabled)", file=sys.stderr)
        return 2
    rows, summary = analyse(load_records(jsonl), load_reference(a.reference), a.min_confidence)
    out = Path(a.output_dir) if a.output_dir else Path(a.run) / "eval"
    write_csv(out / "tag_confidence_bins.csv", rows)
    write_json(out / "tag_confidence_summary.json", {"created_at": now_iso(), "run": str(Path(a.run).resolve()),
                                                     "reference": str(Path(a.reference).resolve()),
                                                     "note": "agreement with RAM on HR, not ground truth",
                                                     "bins": rows, **summary})
    for r in rows:
        ag = "–" if r["agreement_with_ram_hr"] is None else f"{r['agreement_with_ram_hr']:.3f}"
        print(f"c in [{r['c_low']:.2f}, {r['c_high']:.2f}): n={r['n_tags']:5d} agreement={ag}")
    print(f"kept (c > {a.min_confidence:.3f}): {summary['kept']}; dropped: {summary['dropped']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
