# Pilot artifacts: Canon_001_LR4 (single image, 512 × 512)

Saved outputs of the three V2.1 single-image pilot runs, copied unmodified from the Colab results folder
(`results/` on Google Drive), except that absolute Colab/Drive paths in the JSON files were replaced by
`<DRIVE_ROOT>` / `<REPO>` (`pilot_report.py sanitize`).

| Folder | Run (Drive) | Files |
|---|---|---|
| `original/` | `results/realsr_single_image_pilot/` | `run_metadata.json`, `per_image.csv`, `sr.png` (= `sr/sample00/Canon_001_LR4.png`), `paired_per_image.csv` and `paired_summary.json` (= `metrics/`) |
| `heuristic/` | `results/realsr_heuristic_pilot/realsr_heuristic_pilot/` (identical to `FINAL_HEURISTIC_PILOT_BACKUP/`) | same |
| `learned/` | `results/realsr_lornet_learned_pilot/realsr_lornet_learned_pilot/` | same |
| `three_way_pilot_comparison.csv` | `results/final_pilot_comparison/` | the combined table |

Not copied: the reliability maps (`observation_maps.npz`, `spatial_maps.npz`), JSONL logs, prompt text files and the
failed `__replaced_*` debugging runs; they remain in the Drive folder. The LR and GT images are dataset files and are
not redistributed.

```bash
python experiments/tools/pilot_report.py check     # tables in README.md and docs/results.md == CSVs; no private paths
python experiments/tools/pilot_report.py figures --crop 300 0 192   # assets/figures/pilot_metrics.svg, assets/comparisons/canon_001.png
```

Test-set evidence for the input (SHA-256 of LR and GT, training-data search) is tabulated in `docs/results.md`;
the tool's JSON from that Colab run is not included here.

One image: these show that the pipeline runs end to end with both reliability pathways; they are not benchmark
results, and the three runs differ in more than the method (see `docs/results.md`).
