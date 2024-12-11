import matplotlib.pyplot as plt
import matplotlib as mpl
from typing import List, Tuple
import pandas as pd
import numpy as np
import wandb


def fetch_model_runs(model_id: int) -> List[pd.DataFrame]:
    """Fetch all runs for a specific model_id from wandb"""
    api = wandb.Api()
    runs = api.runs(
        "olafercik/mario_shpeed",  # Your wandb project
        filters={"display_name": f"model_{model_id}"},
    )

    dfs = []
    for run in runs:
        run_history = run.history()
        # Convert run history to pandas DataFrame
        df = pd.DataFrame(run_history)
        df["run_id"] = run.id
        dfs.append(df)

    return dfs


def setup_publication_plot(
    figsize: Tuple[float, float] = (5.5, 4),
) -> Tuple[plt.Figure, plt.Axes]:
    """Setup publication-quality plot with LaTeX styling"""
    plt.style.use("seaborn-v0_8-paper")
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "figure.figsize": figsize,
            "figure.dpi": 300,
            "savefig.dpi": 600,
            "text.usetex": True,
            "axes.grid": True,
            "grid.alpha": 0.3,
        }
    )

    return plt.subplots()


def plot_max_x(
    dfs: List[pd.DataFrame], model_id: int, window: int = 50, save: bool = True
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot max_x progression with smoothing"""
    fig, ax = setup_publication_plot()

    for i, df in enumerate(dfs):
        # Apply smoothing
        max_x_smooth = df["max_x"].rolling(window=window, min_periods=1).mean()

        ax.plot(
            df["episode_count"],
            max_x_smooth,
            alpha=0.8,
            label=f"Run {i+1}",
            linewidth=1.0,
        )

    ax.set_title(f"Model {model_id} Maximum Position Progress")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Maximum Position (x)")

    if len(dfs) > 1:
        ax.legend(loc="lower right")

    plt.tight_layout()

    if save:
        fig.savefig(
            f"model_{model_id}_max_x.eps",
            format="eps",
            bbox_inches="tight",
            pad_inches=0.02,
        )

    return fig, ax


def main():
    model_id = int(input("Enter model ID to analyze: "))
    print(f"Fetching run data for model {model_id}...")
    dfs = fetch_model_runs(model_id)

    if not dfs:
        print(f"No runs found for model {model_id}")
        return

    plot_max_x(dfs, model_id)
    print(f"Plot saved as model_{model_id}_max_x.eps")


if __name__ == "__main__":
    main()
