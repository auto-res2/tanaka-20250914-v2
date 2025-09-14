"""
src/utils/figures.py - Plotting and visualization utilities
-----------------------------------------------------------
Implements plotting functions for experiment results visualization.
"""
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import List, Union, Optional

def save_line(x_data: List[float], 
              y_data_list: List[List[float]], 
              labels: List[str],
              xlabel: str, 
              ylabel: str, 
              title: str, 
              output_path: Union[str, Path],
              figsize: tuple = (10, 6)) -> None:
    """Save a line plot with multiple series."""
    plt.figure(figsize=figsize)
    
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p']
    
    for i, (y_data, label) in enumerate(zip(y_data_list, labels)):
        color = colors[i % len(colors)]
        marker = markers[i % len(markers)]
        plt.plot(x_data, y_data, label=label, color=color, marker=marker, linewidth=2, markersize=6)
    
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[figure] Saved plot to: {output_path}")

def save_scatter(x_data: List[float], 
                 y_data: List[float], 
                 labels: List[str],
                 xlabel: str, 
                 ylabel: str, 
                 title: str, 
                 output_path: Union[str, Path],
                 figsize: tuple = (10, 6)) -> None:
    """Save a scatter plot."""
    plt.figure(figsize=figsize)
    
    colors = plt.cm.get_cmap('tab10')(np.linspace(0, 1, len(labels)))
    
    for i, (x, y, label) in enumerate(zip(x_data, y_data, labels)):
        plt.scatter(x, y, label=label, color=colors[i], s=100, alpha=0.7)
    
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[figure] Saved scatter plot to: {output_path}")

def save_bar(categories: List[str], 
             values: List[float], 
             xlabel: str, 
             ylabel: str, 
             title: str, 
             output_path: Union[str, Path],
             figsize: tuple = (10, 6)) -> None:
    """Save a bar plot."""
    plt.figure(figsize=figsize)
    
    bars = plt.bar(categories, values, color='skyblue', alpha=0.8, edgecolor='navy')
    
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(values)*0.01,
                f'{value:.2f}', ha='center', va='bottom', fontsize=10)
    
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[figure] Saved bar plot to: {output_path}")
