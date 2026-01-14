"""Visualization utilities for experiment results."""

import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional
import numpy as np


# Set default style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


def plot_truthfulness_by_category(
    categories: list[str],
    truthfulness_scores: list[float],
    title: str = "Truthfulness by Category",
    figsize: tuple = (12, 6),
    save_path: Optional[str] = None
) -> plt.Figure:
    """Plot truthfulness scores by category.
    
    Args:
        categories: List of category names.
        truthfulness_scores:  Corresponding truthfulness scores. 
        title: Plot title.
        figsize: Figure size. 
        save_path: Path to save the figure (optional).
        
    Returns: 
        matplotlib Figure object.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Sort by score
    sorted_data = sorted(zip(categories, truthfulness_scores), key=lambda x: x[1])
    cats, scores = zip(*sorted_data)
    
    colors = sns.color_palette("RdYlGn", len(cats))
    bars = ax.barh(cats, scores, color=colors)
    
    ax.set_xlabel("Truthfulness Score")
    ax.set_title(title)
    ax.set_xlim(0, 1)
    
    # Add value labels
    for bar, score in zip(bars, scores):
        ax.text(
            score + 0.02, bar.get_y() + bar.get_height() / 2,
            f"{score:.2f}", va="center", fontsize=9
        )
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    
    return fig


def plot_perturbation_impact(
    perturbation_types: list[str],
    baseline_scores: list[float],
    perturbed_scores:  list[float],
    metric_name: str = "F1 Score",
    title: str = "Impact of Perturbations on Model Performance",
    figsize: tuple = (10, 6),
    save_path: Optional[str] = None
) -> plt.Figure:
    """Plot comparison of baseline vs perturbed performance. 
    
    Args:
        perturbation_types: List of perturbation type names.
        baseline_scores: Scores without perturbation.
        perturbed_scores:  Scores with perturbation.
        metric_name: Name of the metric being plotted.
        title: Plot title. 
        figsize: Figure size.
        save_path: Path to save the figure (optional).
        
    Returns:
        matplotlib Figure object.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    x = np.arange(len(perturbation_types))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, baseline_scores, width, label="Baseline", color="#2ecc71")
    bars2 = ax.bar(x + width/2, perturbed_scores, width, label="Perturbed", color="#e74c3c")
    
    ax.set_xlabel("Perturbation Type")
    ax.set_ylabel(metric_name)
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(perturbation_types, rotation=45, ha="right")
    ax.legend()
    ax.set_ylim(0, 1)
    
    # Add value labels
    def add_labels(bars):
        for bar in bars: 
            height = bar.get_height()
            ax.annotate(
                f"{height:.2f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center", va="bottom", fontsize=8
            )
    
    add_labels(bars1)
    add_labels(bars2)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    
    return fig


def plot_failure_distribution(
    failure_types:  list[str],
    counts: list[int],
    title: str = "Distribution of Reasoning Failures",
    figsize: tuple = (10, 8),
    save_path: Optional[str] = None
) -> plt.Figure:
    """Plot distribution of failure types as a pie chart. 
    
    Args: 
        failure_types: List of failure type names.
        counts: Corresponding counts for each failure type. 
        title: Plot title.
        figsize: Figure size. 
        save_path:  Path to save the figure (optional).
        
    Returns: 
        matplotlib Figure object. 
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Filter out zero counts
    non_zero = [(ft, c) for ft, c in zip(failure_types, counts) if c > 0]
    if not non_zero: 
        ax.text(0.5, 0.5, "No failures detected", ha="center", va="center")
        return fig
    
    types, cnts = zip(*non_zero)
    
    colors = sns.color_palette("Set2", len(types))
    explode = [0.05] * len(types)
    
    wedges, texts, autotexts = ax.pie(
        cnts, 
        labels=types, 
        autopct="%1.1f%%",
        explode=explode,
        colors=colors,
        shadow=True,
        startangle=90
    )
    
    ax.set_title(title)
    
    # Make percentage text more readable
    for autotext in autotexts:
        autotext.set_fontsize(10)
        autotext.set_fontweight("bold")
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    
    return fig


def plot_robustness_heatmap(
    models: list[str],
    perturbation_types:  list[str],
    robustness_matrix: list[list[float]],
    title:  str = "Model Robustness Across Perturbation Types",
    figsize: tuple = (10, 8),
    save_path:  Optional[str] = None
) -> plt.Figure:
    """Plot heatmap of robustness scores. 
    
    Args:
        models:  List of model names. 
        perturbation_types: List of perturbation type names.
        robustness_matrix: 2D list of robustness scores [models x perturbations]. 
        title: Plot title.
        figsize: Figure size. 
        save_path: Path to save the figure (optional).
        
    Returns: 
        matplotlib Figure object.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    im = ax.imshow(robustness_matrix, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)
    
    # Set ticks
    ax.set_xticks(np.arange(len(perturbation_types)))
    ax.set_yticks(np.arange(len(models)))
    ax.set_xticklabels(perturbation_types, rotation=45, ha="right")
    ax.set_yticklabels(models)
    
    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Robustness Score", rotation=-90, va="bottom")
    
    # Add text annotations
    for i in range(len(models)):
        for j in range(len(perturbation_types)):
            score = robustness_matrix[i][j]
            text_color = "white" if score < 0.5 else "black"
            ax.text(j, i, f"{score:.2f}", ha="center", va="center", color=text_color)
    
    ax.set_title(title)
    
    plt.tight_layout()
    
    if save_path: 
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    
    return fig


def plot_self_verification_improvement(
    categories: list[str],
    before_verification: list[float],
    after_verification: list[float],
    title: str = "Impact of Self-Verification Prompts",
    figsize: tuple = (12, 6),
    save_path: Optional[str] = None
) -> plt.Figure:
    """Plot improvement from self-verification.
    
    Args: 
        categories: List of category or condition names.
        before_verification: Scores before self-verification.
        after_verification:  Scores after self-verification.
        title: Plot title.
        figsize: Figure size.
        save_path: Path to save the figure (optional).
        
    Returns:
        matplotlib Figure object.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    x = np.arange(len(categories))
    
    # Plot lines connecting before and after
    for i in range(len(categories)):
        color = "#2ecc71" if after_verification[i] > before_verification[i] else "#e74c3c"
        ax.plot(
            [i, i], 
            [before_verification[i], after_verification[i]], 
            color=color, 
            linewidth=2,
            alpha=0.7
        )
    
    # Plot points
    ax.scatter(x, before_verification, s=100, c="#3498db", label="Before Verification", zorder=5)
    ax.scatter(x, after_verification, s=100, c="#9b59b6", label="After Verification", zorder=5)
    
    ax.set_xlabel("Category / Condition")
    ax.set_ylabel("Accuracy Score")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=45, ha="right")
    ax.legend()
    ax.set_ylim(0, 1)
    
    # Add improvement annotations
    for i in range(len(categories)):
        improvement = after_verification[i] - before_verification[i]
        if abs(improvement) > 0.01: 
            y_pos = max(before_verification[i], after_verification[i]) + 0.03
            ax.annotate(
                f"{improvement: +.2f}",
                xy=(i, y_pos),
                ha="center",
                fontsize=9,
                color="#2ecc71" if improvement > 0 else "#e74c3c"
            )
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    
    return fig


def plot_reasoning_chain_analysis(
    step_numbers: list[int],
    accuracy_at_step: list[float],
    error_rate_at_step:  list[float],
    title: str = "Reasoning Chain Analysis",
    figsize: tuple = (10, 6),
    save_path: Optional[str] = None
) -> plt.Figure:
    """Plot how accuracy changes through reasoning steps.
    
    Args: 
        step_numbers:  List of step numbers (1, 2, 3, ...).
        accuracy_at_step: Accuracy at each reasoning step.
        error_rate_at_step: Error rate at each step.
        title: Plot title.
        figsize: Figure size.
        save_path: Path to save the figure (optional).
        
    Returns:
        matplotlib Figure object.
    """
    fig, ax1 = plt.subplots(figsize=figsize)
    
    color1 = "#2ecc71"
    ax1.set_xlabel("Reasoning Step")
    ax1.set_ylabel("Accuracy", color=color1)
    line1 = ax1.plot(step_numbers, accuracy_at_step, color=color1, marker="o", linewidth=2, label="Accuracy")
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.set_ylim(0, 1)
    
    ax2 = ax1.twinx()
    color2 = "#e74c3c"
    ax2.set_ylabel("Error Rate", color=color2)
    line2 = ax2.plot(step_numbers, error_rate_at_step, color=color2, marker="s", linewidth=2, linestyle="--", label="Error Rate")
    ax2.tick_params(axis="y", labelcolor=color2)
    ax2.set_ylim(0, 1)
    
    # Combine legends
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="upper right")
    
    ax1.set_title(title)
    ax1.set_xticks(step_numbers)
    
    plt.tight_layout()
    
    if save_path: 
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    
    return fig


class ExperimentVisualizer:
    """Unified visualizer for experiment results."""
    
    def __init__(self, results_dir: str = "paper/figures"):
        """Initialize the visualizer.
        
        Args: 
            results_dir:  Directory to save figures.
        """
        self.results_dir = results_dir
        self.figures = []
    
    def create_full_report(
        self,
        metrics_data: dict,
        failure_data: dict,
        save_figures: bool = True
    ) -> list[plt.Figure]:
        """Create a full visual report from experiment data.
        
        Args:
            metrics_data: Dictionary with metrics results. 
            failure_data:  Dictionary with failure analysis results.
            save_figures: Whether to save figures to disk.
            
        Returns:
            List of matplotlib Figure objects.
        """
        figures = []
        
        # Create individual plots based on available data
        if "by_category" in metrics_data:
            fig = plot_truthfulness_by_category(
                categories=list(metrics_data["by_category"].keys()),
                truthfulness_scores=list(metrics_data["by_category"].values()),
                save_path=f"{self.results_dir}/truthfulness_by_category.png" if save_figures else None
            )
            figures.append(fig)
        
        if "perturbation_impact" in metrics_data: 
            impact = metrics_data["perturbation_impact"]
            fig = plot_perturbation_impact(
                perturbation_types=list(impact.keys()),
                baseline_scores=[v["baseline"] for v in impact.values()],
                perturbed_scores=[v["perturbed"] for v in impact.values()],
                save_path=f"{self.results_dir}/perturbation_impact.png" if save_figures else None
            )
            figures.append(fig)
        
        if "failure_counts" in failure_data:
            fig = plot_failure_distribution(
                failure_types=list(failure_data["failure_counts"].keys()),
                counts=list(failure_data["failure_counts"].values()),
                save_path=f"{self.results_dir}/failure_distribution.png" if save_figures else None
            )
            figures.append(fig)
        
        self.figures = figures
        return figures
    
    def show_all(self):
        """Display all created figures."""
        plt.show()