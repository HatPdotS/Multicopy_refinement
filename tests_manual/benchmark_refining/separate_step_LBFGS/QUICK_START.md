# Weight Screening - Quick Start

## What This Does

Systematically tests 25 different combinations of restraints and ADP weights to find optimal parameters for your refinement.

## Quick Start

### 1. Run Screening (4-6 hours)
```bash
cd /das/work/p17/p17490/Peter/Library/multicopy_refinement/tests_manual/benchmark_refining/separate_step_LBFGS
sbatch screen_weights.py
```

### 2. Check Progress
```bash
tail -f screening_results/weight_screening.out
```

### 3. Analyze Results (after completion)
```bash
python analyze_screening.py
cat screening_results/screening_summary.txt
```

## Weight Combinations Tested

**25 combinations total** (5 × 5 grid on log scale):

- **Restraints**: 0.20, 0.40, 0.80, 1.60, 2.00
- **ADP**: 0.20, 0.40, 0.80, 1.60, 2.00

Each combination runs **10 cycles** of alternating ADP and XYZ refinement.

## Output Files

All saved to: `screening_results/`

- `weight_screening_results.json` - Complete data
- `screening_summary.txt` - Text summary with best weights
- `heatmap_rfree_final.png` - Heatmap showing final Rfree
- `heatmap_delta_rfree.png` - Heatmap showing improvements
- `comparison_plots.png` - Multiple comparison charts
- `screened_model_r*_a*.pdb` - Refined models (one per combination)

## What to Look For

In `screening_summary.txt`, check:

1. **BEST FINAL RFREE** - Lowest absolute Rfree
2. **BEST IMPROVEMENT** - Largest negative delta (most reliable)

Use those weights in your production refinements!

## Example Result

```
BEST IMPROVEMENT:
   Restraints weight: 0.8000
   ADP weight: 0.4000
   Initial Rfree: 0.3200
   Final Rfree: 0.2450
   Delta Rfree: -0.0750  ← 7.5% improvement!
```

→ Use `restraints=0.8, adp=0.4` for future refinements

## Notes

- Safe to interrupt - results saved after each combination
- Failed combinations are logged but don't stop the run
- Each combination starts fresh (no carry-over effects)
- Weights are auto-scaled based on gradient magnitudes

## Full Documentation

See `SCREENING_README.md` for detailed information.
