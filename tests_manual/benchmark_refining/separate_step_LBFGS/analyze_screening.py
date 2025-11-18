#!/das/work/p17/p17490/CONDA/muticopy_refinement/bin/python
"""
Analyze and visualize weight screening results
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import sys
import os


def load_results(results_file):
    """Load screening results from JSON"""
    with open(results_file, 'r') as f:
        data = json.load(f)
    return data


def plot_heatmap(results, metric='rfree_final', outdir='.'):
    """Create heatmap of results"""
    
    # Extract successful results
    successful = [r for r in results if r['status'] == 'success']
    
    if not successful:
        print("No successful results to plot!")
        return
    
    # Get unique weight values
    restraints_weights = sorted(set(r['restraints_weight'] for r in successful))
    adp_weights = sorted(set(r['adp_weight'] for r in successful))
    
    # Create 2D array for heatmap
    n_restraints = len(restraints_weights)
    n_adp = len(adp_weights)
    values = np.full((n_adp, n_restraints), np.nan)
    
    # Fill in values
    for result in successful:
        i = adp_weights.index(result['adp_weight'])
        j = restraints_weights.index(result['restraints_weight'])
        values[i, j] = result[metric]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot heatmap
    im = ax.imshow(values, aspect='auto', cmap='viridis', origin='lower')
    
    # Set ticks
    ax.set_xticks(range(n_restraints))
    ax.set_yticks(range(n_adp))
    ax.set_xticklabels([f'{w:.3f}' for w in restraints_weights])
    ax.set_yticklabels([f'{w:.3f}' for w in adp_weights])
    
    # Labels
    ax.set_xlabel('Restraints Weight', fontsize=12)
    ax.set_ylabel('ADP Weight', fontsize=12)
    
    metric_names = {
        'rfree_final': 'Final Rfree',
        'delta_rfree': 'Delta Rfree',
        'rwork_final': 'Final Rwork',
        'delta_rwork': 'Delta Rwork'
    }
    ax.set_title(f'Weight Screening: {metric_names.get(metric, metric)}', fontsize=14)
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(metric_names.get(metric, metric), fontsize=12)
    
    # Add text annotations
    for i in range(n_adp):
        for j in range(n_restraints):
            if not np.isnan(values[i, j]):
                text = ax.text(j, i, f'{values[i, j]:.4f}',
                             ha="center", va="center", color="white", fontsize=8)
    
    plt.tight_layout()
    
    # Save figure
    output_file = f'{outdir}/heatmap_{metric}.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved heatmap to: {output_file}")
    plt.close()


def plot_comparison(results, outdir='.'):
    """Create comparison plots"""
    
    successful = [r for r in results if r['status'] == 'success']
    
    if not successful:
        print("No successful results to plot!")
        return
    
    # Sort by final Rfree
    successful_sorted = sorted(successful, key=lambda x: x['rfree_final'])
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Final Rfree vs combination
    ax = axes[0, 0]
    x = range(len(successful))
    rfree_values = [r['rfree_final'] for r in successful_sorted]
    ax.plot(x, rfree_values, 'o-', markersize=4)
    ax.axhline(y=successful_sorted[0]['rfree_initial'], color='r', linestyle='--', label='Initial Rfree')
    ax.set_xlabel('Combination (sorted by final Rfree)')
    ax.set_ylabel('Final Rfree')
    ax.set_title('Final Rfree for All Combinations')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Delta Rfree vs combination
    ax = axes[0, 1]
    delta_values = [r['delta_rfree'] for r in successful_sorted]
    colors = ['green' if d < 0 else 'red' for d in delta_values]
    ax.bar(x, delta_values, color=colors, alpha=0.6)
    ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    ax.set_xlabel('Combination (sorted by final Rfree)')
    ax.set_ylabel('Delta Rfree')
    ax.set_title('Rfree Change (negative = improvement)')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 3: Restraints weight vs Rfree
    ax = axes[1, 0]
    restraints_wts = [r['restraints_weight'] for r in successful]
    rfree_vals = [r['rfree_final'] for r in successful]
    ax.scatter(restraints_wts, rfree_vals, alpha=0.6, s=50)
    ax.set_xlabel('Restraints Weight')
    ax.set_ylabel('Final Rfree')
    ax.set_title('Restraints Weight vs Final Rfree')
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)
    
    # Plot 4: ADP weight vs Rfree
    ax = axes[1, 1]
    adp_wts = [r['adp_weight'] for r in successful]
    ax.scatter(adp_wts, rfree_vals, alpha=0.6, s=50)
    ax.set_xlabel('ADP Weight')
    ax.set_ylabel('Final Rfree')
    ax.set_title('ADP Weight vs Final Rfree')
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_file = f'{outdir}/comparison_plots.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved comparison plots to: {output_file}")
    plt.close()


def create_summary_table(results, outdir='.'):
    """Create a text summary table"""
    
    successful = [r for r in results if r['status'] == 'success']
    
    if not successful:
        print("No successful results!")
        return
    
    output_file = f'{outdir}/screening_summary.txt'
    
    with open(output_file, 'w') as f:
        f.write("=" * 100 + "\n")
        f.write("WEIGHT SCREENING SUMMARY\n")
        f.write("=" * 100 + "\n\n")
        
        f.write(f"Total combinations tested: {len(results)}\n")
        f.write(f"Successful: {len(successful)}\n")
        f.write(f"Failed: {len(results) - len(successful)}\n\n")
        
        # All results table
        f.write("-" * 100 + "\n")
        f.write("ALL RESULTS (sorted by final Rfree):\n")
        f.write("-" * 100 + "\n")
        f.write(f"{'ID':<6} {'Restraints':<12} {'ADP':<12} {'Rfree Init':<14} "
                f"{'Rfree Final':<14} {'Delta':<12} {'Rwork Final':<14}\n")
        f.write("-" * 100 + "\n")
        
        successful_sorted = sorted(successful, key=lambda x: x['rfree_final'])
        
        for result in successful_sorted:
            f.write(f"{result['combo_id']:<6} "
                   f"{result['restraints_weight']:<12.4f} "
                   f"{result['adp_weight']:<12.4f} "
                   f"{result['rfree_initial']:<14.4f} "
                   f"{result['rfree_final']:<14.4f} "
                   f"{result['delta_rfree']:<+12.4f} "
                   f"{result['rwork_final']:<14.4f}\n")
        
        # Best combinations
        f.write("\n" + "=" * 100 + "\n")
        f.write("BEST COMBINATIONS\n")
        f.write("=" * 100 + "\n\n")
        
        # Best final Rfree
        best_final = successful_sorted[0]
        f.write("1. BEST FINAL RFREE:\n")
        f.write(f"   Combination ID: {best_final['combo_id']}\n")
        f.write(f"   Restraints weight: {best_final['restraints_weight']:.4f}\n")
        f.write(f"   ADP weight: {best_final['adp_weight']:.4f}\n")
        f.write(f"   Initial Rfree: {best_final['rfree_initial']:.4f}\n")
        f.write(f"   Final Rfree: {best_final['rfree_final']:.4f}\n")
        f.write(f"   Delta Rfree: {best_final['delta_rfree']:+.4f}\n")
        f.write(f"   Final Rwork: {best_final['rwork_final']:.4f}\n\n")
        
        # Best improvement
        best_improvement = min(successful, key=lambda x: x['delta_rfree'])
        f.write("2. BEST IMPROVEMENT:\n")
        f.write(f"   Combination ID: {best_improvement['combo_id']}\n")
        f.write(f"   Restraints weight: {best_improvement['restraints_weight']:.4f}\n")
        f.write(f"   ADP weight: {best_improvement['adp_weight']:.4f}\n")
        f.write(f"   Initial Rfree: {best_improvement['rfree_initial']:.4f}\n")
        f.write(f"   Final Rfree: {best_improvement['rfree_final']:.4f}\n")
        f.write(f"   Delta Rfree: {best_improvement['delta_rfree']:+.4f}\n")
        f.write(f"   Final Rwork: {best_improvement['rwork_final']:.4f}\n\n")
        
        f.write("=" * 100 + "\n")
    
    print(f"Saved summary table to: {output_file}")


def main():
    if len(sys.argv) > 1:
        results_file = sys.argv[1]
    else:
        results_file = '/das/work/p17/p17490/Peter/Library/multicopy_refinement/tests_manual/benchmark_refining/separate_step_LBFGS/screening_results/weight_screening_results.json'
    
    if not os.path.exists(results_file):
        print(f"Error: Results file not found: {results_file}")
        sys.exit(1)
    
    print(f"Loading results from: {results_file}")
    data = load_results(results_file)
    results = data['results']
    
    outdir = os.path.dirname(results_file)
    
    print(f"\nFound {len(results)} combinations")
    print(f"Timestamp: {data.get('timestamp', 'N/A')}")
    
    # Create visualizations
    print("\nCreating visualizations...")
    plot_heatmap(results, metric='rfree_final', outdir=outdir)
    plot_heatmap(results, metric='delta_rfree', outdir=outdir)
    plot_comparison(results, outdir=outdir)
    create_summary_table(results, outdir=outdir)
    
    print("\nAnalysis complete!")


if __name__ == '__main__':
    main()
