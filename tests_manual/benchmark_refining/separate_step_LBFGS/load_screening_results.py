#!/das/work/p17/p17490/CONDA/muticopy_refinement/bin/python
"""
Load weight screening results into a pandas DataFrame for analysis
"""

import json
import pandas as pd
import sys
import os


def load_screening_results(json_file):
    """
    Load screening results from JSON file into a pandas DataFrame
    
    Args:
        json_file: Path to weight_screening_results.json
        
    Returns:
        pandas DataFrame with all screening results
    """
    # Load JSON
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # Extract results list
    results = data['results']
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Add some computed columns for convenience
    if 'rfree_final' in df.columns and 'rfree_initial' in df.columns:
        df['rfree_improvement_pct'] = (df['delta_rfree'] / df['rfree_initial'] * 100)
    
    if 'rwork_final' in df.columns and 'rwork_initial' in df.columns:
        df['rwork_improvement_pct'] = (df['delta_rwork'] / df['rwork_initial'] * 100)
    
    # Add metadata as attributes
    df.attrs['timestamp'] = data.get('timestamp', 'N/A')
    df.attrs['n_combinations'] = data.get('n_combinations', len(results))
    
    return df


def print_summary(df):
    """Print summary statistics of the screening results"""
    print("=" * 80)
    print("SCREENING RESULTS SUMMARY")
    print("=" * 80)
    print(f"\nTimestamp: {df.attrs.get('timestamp', 'N/A')}")
    print(f"Total combinations: {len(df)}")
    
    # Filter successful results
    successful = df[df['status'] == 'success']
    failed = df[df['status'] == 'failed']
    
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")
    
    if len(successful) == 0:
        print("\nNo successful results to analyze!")
        return
    
    # Basic statistics
    print("\n" + "-" * 80)
    print("RFREE STATISTICS (successful runs only)")
    print("-" * 80)
    print(f"Initial Rfree: {successful['rfree_initial'].mean():.4f} ± {successful['rfree_initial'].std():.4f}")
    print(f"Final Rfree:   {successful['rfree_final'].mean():.4f} ± {successful['rfree_final'].std():.4f}")
    print(f"Best Rfree:    {successful['rfree_final'].min():.4f}")
    print(f"Worst Rfree:   {successful['rfree_final'].max():.4f}")
    print(f"Mean improvement: {successful['delta_rfree'].mean():.4f} ({successful['rfree_improvement_pct'].mean():.1f}%)")
    
    # Best results
    print("\n" + "-" * 80)
    print("TOP 5 RESULTS (by final Rfree)")
    print("-" * 80)
    top5 = successful.nsmallest(5, 'rfree_final')[['combo_id', 'restraints_weight', 'adp_weight', 
                                                     'rfree_final', 'delta_rfree', 'rwork_final']]
    print(top5.to_string(index=False))
    
    # Best improvement
    print("\n" + "-" * 80)
    print("TOP 5 IMPROVEMENTS (by delta Rfree)")
    print("-" * 80)
    best_improvement = successful.nsmallest(5, 'delta_rfree')[['combo_id', 'restraints_weight', 'adp_weight', 
                                                                 'rfree_initial', 'rfree_final', 'delta_rfree']]
    print(best_improvement.to_string(index=False))
    
    print("\n" + "=" * 80)


def export_to_csv(df, output_file):
    """Export DataFrame to CSV file"""
    # Select relevant columns
    columns_to_export = [
        'combo_id', 'restraints_weight', 'adp_weight',
        'rwork_initial', 'rfree_initial',
        'rwork_final', 'rfree_final',
        'delta_rwork', 'delta_rfree',
        'rfree_improvement_pct', 'rwork_improvement_pct',
        'n_cycles', 'status'
    ]
    
    # Export only columns that exist
    export_cols = [col for col in columns_to_export if col in df.columns]
    df[export_cols].to_csv(output_file, index=False)
    print(f"Exported to CSV: {output_file}")


def main():
    """Main function"""
    # Default path or from command line
    if len(sys.argv) > 1:
        json_file = sys.argv[1]
    else:
        json_file = '/das/work/p17/p17490/Peter/Library/multicopy_refinement/tests_manual/benchmark_refining/separate_step_LBFGS/screening_results/weight_screening_results.json'
    
    if not os.path.exists(json_file):
        print(f"Error: File not found: {json_file}")
        sys.exit(1)
    
    print(f"Loading: {json_file}")
    
    # Load data
    df = load_screening_results(json_file)
    
    # Print summary
    print_summary(df)
    
    # Export to CSV
    outdir = os.path.dirname(json_file)
    csv_file = os.path.join(outdir, 'screening_results.csv')
    export_to_csv(df, csv_file)
    
    # Return DataFrame for interactive use
    return df


if __name__ == '__main__':
    df = main()
    
    # Print some usage tips
    print("\n" + "=" * 80)
    print("USAGE IN PYTHON/IPYTHON:")
    print("=" * 80)
    print("from load_screening_results import load_screening_results")
    print("df = load_screening_results('path/to/weight_screening_results.json')")
    print("")
    print("# Filter successful results")
    print("successful = df[df['status'] == 'success']")
    print("")
    print("# Sort by final Rfree")
    print("df_sorted = successful.sort_values('rfree_final')")
    print("")
    print("# Get best result")
    print("best = successful.loc[successful['rfree_final'].idxmin()]")
    print("print(f\"Best: restraints={best['restraints_weight']}, adp={best['adp_weight']}\")")
    print("")
    print("# Plot results")
    print("import matplotlib.pyplot as plt")
    print("plt.scatter(successful['restraints_weight'], successful['rfree_final'])")
    print("plt.xlabel('Restraints Weight'); plt.ylabel('Final Rfree')")
    print("plt.xscale('log'); plt.show()")
    print("=" * 80)
