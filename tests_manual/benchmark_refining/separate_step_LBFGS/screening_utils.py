"""
Simple utilities for loading and analyzing weight screening results
"""

import json
import pandas as pd


def load_results(json_file):
    """
    Load screening results JSON into a pandas DataFrame
    
    Args:
        json_file (str): Path to weight_screening_results.json
        
    Returns:
        pd.DataFrame: Results with columns for weights, R-factors, etc.
        
    Example:
        >>> df = load_results('screening_results/weight_screening_results.json')
        >>> best = df.loc[df['rfree_final'].idxmin()]
        >>> print(f"Best: R={best['restraints_weight']:.3f}, A={best['adp_weight']:.3f}")
    """
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    df = pd.DataFrame(data['results'])
    
    # Add computed columns
    if 'rfree_final' in df.columns:
        df['rfree_improvement_pct'] = (df['delta_rfree'] / df['rfree_initial']) * 100
    if 'rwork_final' in df.columns:
        df['rwork_improvement_pct'] = (df['delta_rwork'] / df['rwork_initial']) * 100
    
    # Store metadata
    df.attrs['timestamp'] = data.get('timestamp')
    df.attrs['n_combinations'] = len(df)
    
    return df


def get_best(df, metric='rfree_final', status='success'):
    """
    Get the best result based on a metric
    
    Args:
        df (pd.DataFrame): Results DataFrame
        metric (str): Column to optimize ('rfree_final', 'delta_rfree', etc.)
        status (str): Filter by status ('success', 'failed', or None for all)
        
    Returns:
        pd.Series: Best result row
        
    Example:
        >>> best = get_best(df, metric='rfree_final')
        >>> print(f"Best final Rfree: {best['rfree_final']:.4f}")
    """
    filtered = df[df['status'] == status] if status else df
    
    if len(filtered) == 0:
        raise ValueError(f"No results with status='{status}'")
    
    # For delta metrics, minimize (most negative = best improvement)
    # For absolute metrics, minimize (lowest = best)
    return filtered.loc[filtered[metric].idxmin()]


def get_top_n(df, n=5, metric='rfree_final', status='success'):
    """
    Get top N results based on a metric
    
    Args:
        df (pd.DataFrame): Results DataFrame
        n (int): Number of top results to return
        metric (str): Column to optimize
        status (str): Filter by status
        
    Returns:
        pd.DataFrame: Top N results
        
    Example:
        >>> top5 = get_top_n(df, n=5, metric='rfree_final')
        >>> print(top5[['restraints_weight', 'adp_weight', 'rfree_final']])
    """
    filtered = df[df['status'] == status] if status else df
    
    if len(filtered) == 0:
        raise ValueError(f"No results with status='{status}'")
    
    return filtered.nsmallest(n, metric)


def summary_stats(df, status='success'):
    """
    Get summary statistics for screening results
    
    Args:
        df (pd.DataFrame): Results DataFrame
        status (str): Filter by status
        
    Returns:
        dict: Summary statistics
        
    Example:
        >>> stats = summary_stats(df)
        >>> print(f"Mean improvement: {stats['mean_rfree_improvement']:.4f}")
    """
    filtered = df[df['status'] == status] if status else df
    
    if len(filtered) == 0:
        return {}
    
    return {
        'n_total': len(df),
        'n_successful': len(df[df['status'] == 'success']),
        'n_failed': len(df[df['status'] == 'failed']),
        'rfree_initial_mean': filtered['rfree_initial'].mean(),
        'rfree_final_mean': filtered['rfree_final'].mean(),
        'rfree_final_std': filtered['rfree_final'].std(),
        'rfree_final_min': filtered['rfree_final'].min(),
        'rfree_final_max': filtered['rfree_final'].max(),
        'mean_rfree_improvement': filtered['delta_rfree'].mean(),
        'best_rfree_improvement': filtered['delta_rfree'].min(),
    }


def print_stats(df, status='success'):
    """
    Print formatted summary statistics
    
    Args:
        df (pd.DataFrame): Results DataFrame
        status (str): Filter by status
        
    Example:
        >>> print_stats(df)
    """
    stats = summary_stats(df, status)
    
    if not stats:
        print(f"No results with status='{status}'")
        return
    
    print("=" * 60)
    print("SCREENING RESULTS SUMMARY")
    print("=" * 60)
    print(f"Total combinations: {stats['n_total']}")
    print(f"Successful: {stats['n_successful']}")
    print(f"Failed: {stats['n_failed']}")
    print()
    print("Rfree Statistics:")
    print(f"  Initial (mean): {stats['rfree_initial_mean']:.4f}")
    print(f"  Final (mean):   {stats['rfree_final_mean']:.4f} ± {stats['rfree_final_std']:.4f}")
    print(f"  Final (best):   {stats['rfree_final_min']:.4f}")
    print(f"  Final (worst):  {stats['rfree_final_max']:.4f}")
    print()
    print("Improvements:")
    print(f"  Mean:  {stats['mean_rfree_improvement']:.4f}")
    print(f"  Best:  {stats['best_rfree_improvement']:.4f}")
    print("=" * 60)


def export_csv(df, output_file):
    """
    Export DataFrame to CSV
    
    Args:
        df (pd.DataFrame): Results DataFrame
        output_file (str): Output CSV filename
        
    Example:
        >>> export_csv(df, 'screening_results.csv')
    """
    columns = [
        'combo_id', 'restraints_weight', 'adp_weight',
        'rfree_initial', 'rfree_final', 'delta_rfree',
        'rwork_initial', 'rwork_final', 'delta_rwork',
        'rfree_improvement_pct', 'n_cycles', 'status'
    ]
    
    # Only export columns that exist
    export_cols = [col for col in columns if col in df.columns]
    df[export_cols].to_csv(output_file, index=False)
    print(f"Exported to: {output_file}")


# Quick access functions
def load_and_print(json_file):
    """Load results and print summary in one call"""
    df = load_results(json_file)
    print_stats(df)
    return df


if __name__ == '__main__':
    # Example usage
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python screening_utils.py <json_file>")
        print("Example: python screening_utils.py screening_results/weight_screening_results.json")
        sys.exit(1)
    
    json_file = sys.argv[1]
    df = load_and_print(json_file)
    
    print("\nTop 5 results:")
    top5 = get_top_n(df, n=5)
    print(top5[['combo_id', 'restraints_weight', 'adp_weight', 'rfree_final', 'delta_rfree']])
