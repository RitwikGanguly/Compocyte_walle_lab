"""
Plot v1 vs v2.0 comparison figures for 4 TIL datasets.

Generates 4 figures (one per dataset), each with:
- Left panel: Speedup (Train + Total) for v1/v2_cpu/v2_gpu
- Right panel: Per-level accuracy AND coverage (grouped bars)

Output: comparison/figs/{dataset}_v1_vs_v2_comparison.{png,pdf}
"""

import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

# Nature Biotechnology style
plt.rcParams.update({
    'font.family': 'Arial',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 8,
    'figure.dpi': 300,
    'axes.grid': False,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.linewidth': 0.8,
})

# Color scheme
COLORS = {
    'v1': '#888888',      # gray (baseline)
    'v2_cpu': '#4477AA',  # blue
    'v2_gpu': '#228833'   # green
}

DATASETS = ['steele', 'che', 'wang', 'bassez']
LEVELS = ['Level_1_pred', 'Level_2_pred', 'Level_3_pred', 
          'Level_4_pred', 'Level_5_pred', 'Level_6_pred', 'Level_7_pred']
LEVEL_LABELS = ['L1', 'L2', 'L3', 'L4', 'L5', 'L6', 'L7']

def load_dataset(dataset):
    """Load comparison JSON for a dataset."""
    path = Path('comparison/result/pretraining/compare_{dataset}.json'.format(dataset=dataset))
    with open(path) as f:
        return json.load(f)

def plot_speedup_panel(ax, data, dataset):
    """Left panel: speedup comparison (Train + Total)."""
    speedups = data['speedup_vs_v1']
    
    # Extract speedup values
    train_speedups = [1.0, speedups['v2_cpu']['train'], speedups['v2_gpu']['train']]
    total_speedups = [1.0, speedups['v2_cpu']['total'], speedups['v2_gpu']['total']]
    
    # Bar positions
    x = np.arange(2)
    width = 0.25
    
    # Plot train speedups (group 1)
    bars_train_v1 = ax.bar(x[0] - width, train_speedups[0], width, color=COLORS['v1'],
                           edgecolor='black', linewidth=0.5, label='v1.0 (CPU)')
    bars_train_v2cpu = ax.bar(x[0], train_speedups[1], width, color=COLORS['v2_cpu'],
                              edgecolor='black', linewidth=0.5, label='v2.0 (CPU)')
    bars_train_v2gpu = ax.bar(x[0] + width, train_speedups[2], width, color=COLORS['v2_gpu'],
                              edgecolor='black', linewidth=0.5, label='v2.0 (GPU)')
    
    # Plot total speedups (group 2)
    bars_total_v1 = ax.bar(x[1] - width, total_speedups[0], width, color=COLORS['v1'],
                           edgecolor='black', linewidth=0.5)
    bars_total_v2cpu = ax.bar(x[1], total_speedups[1], width, color=COLORS['v2_cpu'],
                              edgecolor='black', linewidth=0.5)
    bars_total_v2gpu = ax.bar(x[1] + width, total_speedups[2], width, color=COLORS['v2_gpu'],
                              edgecolor='black', linewidth=0.5)
    
    # Add value labels on bars
    for bars in [bars_train_v1, bars_train_v2cpu, bars_train_v2gpu,
                 bars_total_v1, bars_total_v2cpu, bars_total_v2gpu]:
        for bar in bars:
            height = bar.get_height()
            if height > 1.0:
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                       f'{height:.1f}×', ha='center', va='bottom', fontsize=8)
    
    # Reference line at 1.0×
    ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
    
    # Formatting
    ax.set_xlabel('Phase', fontsize=10)
    ax.set_ylabel('Speedup (×v1.0)', fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(['Training', 'Total'])
    ax.set_ylim(0, max(total_speedups) * 1.2)
    ax.legend(loc='upper left', frameon=False, fontsize=8)
    ax.set_title(f'Dataset: {dataset.capitalize()}', fontsize=11, fontweight='bold', pad=10)

def plot_accuracy_coverage_panel(ax_acc, ax_cov, data):
    """Right panel: per-level accuracy and coverage."""
    correctness = data['correctness_vs_truth']
    
    # Extract accuracy and coverage for all 3 frameworks
    accuracy = {
        'v1': [correctness['v1']['per_level_vs_truth'][lvl]['accuracy'] * 100 for lvl in LEVELS],
        'v2_cpu': [correctness['v2_cpu']['per_level_vs_truth'][lvl]['accuracy'] * 100 for lvl in LEVELS],
        'v2_gpu': [correctness['v2_gpu']['per_level_vs_truth'][lvl]['accuracy'] * 100 for lvl in LEVELS]
    }
    
    coverage = {
        'v1': [correctness['v1']['per_level_vs_truth'][lvl]['coverage'] * 100 for lvl in LEVELS],
        'v2_cpu': [correctness['v2_cpu']['per_level_vs_truth'][lvl]['coverage'] * 100 for lvl in LEVELS],
        'v2_gpu': [correctness['v2_gpu']['per_level_vs_truth'][lvl]['coverage'] * 100 for lvl in LEVELS]
    }
    
    x = np.arange(len(LEVELS))
    width = 0.25
    
    # Accuracy subplot
    bars_acc_v1 = ax_acc.bar(x - width, accuracy['v1'], width, color=COLORS['v1'],
                             edgecolor='black', linewidth=0.5, label='v1.0')
    bars_acc_v2cpu = ax_acc.bar(x, accuracy['v2_cpu'], width, color=COLORS['v2_cpu'],
                                edgecolor='black', linewidth=0.5, label='v2.0 (CPU)')
    bars_acc_v2gpu = ax_acc.bar(x + width, accuracy['v2_gpu'], width, color=COLORS['v2_gpu'],
                                edgecolor='black', linewidth=0.5, label='v2.0 (GPU)')
    
    ax_acc.set_ylabel('Accuracy (%)', fontsize=10)
    ax_acc.set_ylim(0, 110)
    ax_acc.set_xticks(x)
    ax_acc.set_xticklabels(LEVEL_LABELS, rotation=0)
    ax_acc.legend(loc='lower left', frameon=False, fontsize=7)
    ax_acc.set_title('Per-Level Accuracy', fontsize=10, pad=5)
    
    # Add value labels (only if accuracy < 95 to avoid clutter)
    for bars in [bars_acc_v1, bars_acc_v2cpu, bars_acc_v2gpu]:
        for bar in bars:
            height = bar.get_height()
            if height < 95:
                ax_acc.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                           f'{height:.0f}', ha='center', va='bottom',
                           fontsize=6, rotation=90)
    
    # Coverage subplot
    bars_cov_v1 = ax_cov.bar(x - width, coverage['v1'], width, color=COLORS['v1'],
                             edgecolor='black', linewidth=0.5)
    bars_cov_v2cpu = ax_cov.bar(x, coverage['v2_cpu'], width, color=COLORS['v2_cpu'],
                                edgecolor='black', linewidth=0.5)
    bars_cov_v2gpu = ax_cov.bar(x + width, coverage['v2_gpu'], width, color=COLORS['v2_gpu'],
                                edgecolor='black', linewidth=0.5)
    
    ax_cov.set_ylabel('Coverage (%)', fontsize=10)
    ax_cov.set_xlabel('Hierarchy Level', fontsize=10)
    ax_cov.set_ylim(0, 110)
    ax_cov.set_xticks(x)
    ax_cov.set_xticklabels(LEVEL_LABELS, rotation=0)
    ax_cov.set_title('Per-Level Coverage', fontsize=10, pad=5)
    
    # Add value labels
    for bars in [bars_cov_v1, bars_cov_v2cpu, bars_cov_v2gpu]:
        for bar in bars:
            height = bar.get_height()
            if height > 5:  # only label if coverage > 5%
                ax_cov.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                           f'{height:.0f}', ha='center', va='bottom',
                           fontsize=6, rotation=90)

def plot_dataset(dataset):
    """Generate figure for one dataset."""
    print(f'Plotting {dataset}...')
    data = load_dataset(dataset)
    
    # Create figure with 1x2 grid
    fig = plt.figure(figsize=(14, 5))
    
    # Grid spec: left panel (speedup) takes 1/3 width, right panel takes 2/3
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1],
                          hspace=0.4, wspace=0.3)
    
    # Left panel: speedup
    ax_speedup = fig.add_subplot(gs[:, 0])  # spans both rows
    plot_speedup_panel(ax_speedup, data, dataset)
    
    # Right panel: accuracy (top) and coverage (bottom)
    ax_accuracy = fig.add_subplot(gs[0, 1])
    ax_coverage = fig.add_subplot(gs[1, 1])
    plot_accuracy_coverage_panel(ax_accuracy, ax_coverage, data)
    
    # Add dataset metadata as annotation
    n_train = data['time']['v1']['n_train']
    n_test = data['time']['v1']['n_test']
    metadata_text = f'n_train={n_train:,}  n_test={n_test:,}'
    fig.text(0.02, 0.02, metadata_text, fontsize=7, style='italic', color='gray')
    
    # Save
    out_dir = Path('comparison/figs')
    out_dir.mkdir(exist_ok=True)
    
    fig.savefig(out_dir / f'{dataset}_v1_vs_v2_comparison.png', dpi=300, bbox_inches='tight')
    fig.savefig(out_dir / f'{dataset}_v1_vs_v2_comparison.pdf', bbox_inches='tight')
    
    plt.close(fig)
    print(f'  ✓ Saved {dataset} figures')

if __name__ == '__main__':
    print('Generating v1 vs v2.0 comparison figures...\n')
    for dataset in DATASETS:
        plot_dataset(dataset)
    print('\nAll figures saved to comparison/figs/')
