# Weight Screening Scripts

## Overview

These scripts screen different combinations of restraints and ADP weights to find optimal parameters for crystallographic refinement.

## Files

### 1. `screen_weights.py`
Main screening script that:
- Tests 25 weight combinations (5 restraints × 5 ADP values)
- Uses log scale: 0.2 to 2.0 for both parameters
- Runs 10 refinement cycles per combination
- Saves results to JSON and PDB files

### 2. `analyze_screening.py`
Analysis and visualization script that creates:
- Heatmaps of final Rfree and Delta Rfree
- Comparison plots
- Summary text table with best combinations

## Usage

### Running the Screening

Submit the screening job:
```bash
sbatch /das/work/p17/p17490/Peter/Library/multicopy_refinement/tests_manual/benchmark_refining/separate_step_LBFGS/screen_weights.py
```

This will:
- Test all 25 weight combinations
- Save results to `screening_results/weight_screening_results.json`
- Save refined models for each combination
- Take approximately 4-6 hours depending on system

### Analyzing the Results

After screening completes, analyze the results:
```bash
python /das/work/p17/p17490/Peter/Library/multicopy_refinement/tests_manual/benchmark_refining/separate_step_LBFGS/analyze_screening.py
```

Or specify a custom results file:
```bash
python analyze_screening.py /path/to/weight_screening_results.json
```

This creates:
- `heatmap_rfree_final.png` - Heatmap of final Rfree values
- `heatmap_delta_rfree.png` - Heatmap of Rfree improvements
- `comparison_plots.png` - Multiple comparison plots
- `screening_summary.txt` - Detailed text summary

## Weight Parameters

### Restraints Weight (5 values, log scale 0.2-2.0):
- **0.200** - Very weak geometry restraints
- **0.400** - Weak geometry restraints
- **0.800** - Moderate geometry restraints
- **1.600** - Strong geometry restraints  
- **2.000** - Very strong geometry restraints

### ADP Weight (5 values, log scale 0.2-2.0):
- **0.200** - Very weak B-factor regularization
- **0.400** - Weak B-factor regularization
- **0.800** - Moderate B-factor regularization
- **1.600** - Strong B-factor regularization
- **2.000** - Very strong B-factor regularization

## Output Files

### JSON Results (`weight_screening_results.json`)
Contains for each combination:
- `restraints_weight` - Weight used for geometry restraints
- `adp_weight` - Weight used for ADP regularization
- `rfree_initial` - Initial Rfree value
- `rfree_final` - Final Rfree after refinement
- `delta_rfree` - Change in Rfree (negative = improvement)
- `rwork_initial` / `rwork_final` - Work set R-factors
- `model_file` - Path to refined PDB file
- `status` - success or failed
- `n_cycles` - Number of refinement cycles run

### PDB Files
Each successful combination saves a refined model:
- Format: `screened_model_r{restraints}_a{adp}.pdb`
- Example: `screened_model_r0.800_a0.400.pdb`

### Summary Table (`screening_summary.txt`)
Text file containing:
- Complete results sorted by final Rfree
- Best final Rfree combination
- Best improvement combination
- Statistics for all runs

### Visualizations
- **`heatmap_rfree_final.png`** - Shows final Rfree for all weight combinations
- **`heatmap_delta_rfree.png`** - Shows improvement (darker = better)
- **`comparison_plots.png`** - 4-panel figure with various comparisons

## Interpreting Results

### What to Look For:

1. **Best Final Rfree**: Lowest absolute Rfree value
   - Indicates best overall fit to data
   - May have high initial Rfree

2. **Best Improvement**: Largest negative delta Rfree
   - Shows which weights gave best refinement
   - Most reliable for future refinements

3. **Trends**:
   - Sweet spot usually in middle of range
   - Extremes (very high/low weights) often perform poorly
   - Look for "valley" in heatmap

### Example Interpretation:

If best results are:
- Restraints: 0.8
- ADP: 0.4

This suggests:
- Moderate geometry restraints work best
- Lighter B-factor regularization is optimal
- Use these weights for production refinements

## Customization

### Change Weight Range

Edit `screen_weights.py`, line ~136:
```python
restraints_weights = np.logspace(np.log10(0.2), np.log10(2.0), 5)
adp_weights = np.logspace(np.log10(0.2), np.log10(2.0), 5)
```

Change `0.2`, `2.0`, or `5` as needed.

### Change Number of Cycles

Edit `screen_weights.py`, line ~295:
```python
results = screen_weights(mtz, pdb, outdir, n_cycles=10)
```

More cycles = better convergence but longer runtime.

### Test Different Structure

Edit `screen_weights.py`, lines ~287-289:
```python
mtz = '/path/to/your.mtz'
pdb = '/path/to/your.pdb'
```

## Computational Requirements

- **CPU cores**: 16 (adjust with `#SBATCH -c`)
- **Memory**: ~4-8 GB per combination
- **Time**: ~15-20 minutes per combination
- **Total**: 4-6 hours for 25 combinations
- **Disk**: ~100 MB per combination (for PDB files)

## Notes

- Each combination uses a fresh refinement object (no carry-over)
- Weights are automatically scaled based on gradient magnitudes
- Failed combinations are logged but don't stop the screening
- Results are saved after each combination (safe for interruption)

## Troubleshooting

### Job Dies Before Completion
Results are saved incrementally - check `weight_screening_results.json` for partial results.

### Out of Memory
Reduce number of CPU cores or run fewer combinations at once.

### Poor Results for All Weights
- Check initial structure quality
- Verify MTZ and PDB files are compatible
- Try adjusting weight range (may need lower or higher values)

## Example Workflow

```bash
# 1. Submit screening job
sbatch screen_weights.py

# 2. Monitor progress
tail -f screening_results/weight_screening.out

# 3. After completion, analyze results
python analyze_screening.py

# 4. View results
cat screening_results/screening_summary.txt
open screening_results/heatmap_rfree_final.png

# 5. Use best weights for production
# (Update your refinement script with optimal values)
```
