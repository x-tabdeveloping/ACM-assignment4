from collections import defaultdict
from pathlib import Path
from typing import Iterable

import jax
import jax.numpy as jnp
import joblib
import numpyro.distributions as dist
import plotly.express as px
import plotly.graph_objects as go
from jax.scipy.stats import gaussian_kde
from plotly.subplots import make_subplots

from simulation_study import simulate_experiment
from utils.agents import naive_bayes, simulate_behaviour
from utils.plots import plot_forest, plot_predictives


def load_result_files(paths: Iterable[Path]) -> dict[str, list]:
    paths = list(paths)
    file_ids = jnp.array([int(sf.stem.split("_")[1]) for sf in paths])
    order = jnp.argsort(file_ids)
    data = defaultdict(list)
    for i_file in order:
        exp_data = joblib.load(paths[i_file])
        for key, value in exp_data.items():
            data[key].append(value)
    return data


def main():
    figures_dir = Path("figures")
    figures_dir.mkdir(exist_ok=True)

    sim_files = list(Path("results/simulations/").glob("simulation_*.joblib"))
    data = load_result_files(sim_files)

    fig = plot_forest(data["samples"], jnp.array(data["lr"]))
    fig = fig.update_traces(showlegend=False)
    fig = fig.update_layout(width=800, height=300)
    fig.write_image(figures_dir.joinpath("simulation_parameter_recovery.png"), scale=2)
    fig.show()

    fig = plot_predictives(
        data["prior_predictive"],
        data["posterior_predictive"],
        data["choices"],
        names=[f"lr={lr:.2f}" for lr in data["lr"]],
    )
    fig = fig.update_layout(width=1200, height=400)
    fig.write_image(figures_dir.joinpath("simulation_predictives.png"), scale=2)
    fig.show()

    n_grid_points = 200
    n_cols = 7
    n_data = len(data["prior_predictive"]) - 1
    n_rows = n_data // n_cols
    if n_data % n_cols:
        n_rows += 1
    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=[f"lr={lr:.2f}" for lr in data["lr"]][1:],
        horizontal_spacing=0.02,
        vertical_spacing=0.15,
    )
    color_scheme = px.colors.qualitative.Dark24
    for i, posterior in enumerate(data["samples"][1:]):
        color = color_scheme[i]
        posterior_lr = jnp.ravel(posterior["lr"])
        grid = jnp.linspace(0, 2, n_grid_points)
        fig.add_scatter(
            x=grid,
            y=jnp.exp(dist.HalfNormal(1.0).log_prob(grid)),
            line=dict(color=color, width=2, dash="dash"),
            name="Prior",
            showlegend=i == 0,
            col=i % n_cols + 1,
            row=i // n_cols + 1,
        )
        fig.add_scatter(
            x=grid,
            y=gaussian_kde(posterior_lr).pdf(grid),
            line=dict(color=color, width=2),
            name="Posterior",
            showlegend=i == 0,
            col=i % n_cols + 1,
            row=i // n_cols + 1,
        )
    fig = fig.update_yaxes(visible=False)
    fig = fig.update_layout(
        template="plotly_white",
        margin=dict(t=20, l=10, b=10, r=10),
        width=1000,
        height=400,
    )
    fig.write_image(
        figures_dir.joinpath("simulation_prior_posterior_update.png"), scale=2
    )
    fig.show()

    fig = go.Figure()
    key = jax.random.key(42)
    key, subkey = jax.random.split(key)
    n_rules = 5
    n_trials_per_rule = 100
    xs, ys = simulate_experiment(
        subkey, n_rules=n_rules, n_trials_per_rule=n_trials_per_rule
    )
    lrs = jnp.linspace(0, 1, 15)
    colors = px.colors.sample_colorscale(
        px.colors.get_colorscale("Viridis"), lrs.tolist()
    )
    for color, lr in zip(colors, lrs):
        agent = naive_bayes.add_input(xs, ys)
        key, subkey = jax.random.split(key)
        cs, agent_trace = simulate_behaviour(
            subkey,
            agent,
            parameters=dict(lr=lr),
        )
        wins = jnp.all(ys == cs, axis=1)
        trials = jnp.arange(len(wins))
        win_rate = jnp.cumsum(wins) / (trials + 1)
        fig = fig.add_scatter(
            x=trials,
            y=win_rate,
            name=f"lr={lr:.2f}",
            line=dict(color=color),
            showlegend=False,
        )
    for i in range(n_rules):
        if i != 0:
            fig = fig.add_vline(
                x=i * 100,
                line=dict(
                    width=2,
                    color="black",
                    dash="dash",
                ),
                annotation_text=f"Rule {i+1}",
            )
    # I add a dummy trace so I can display the color scale
    fig = fig.add_heatmap(opacity=0.0, coloraxis="coloraxis", z=[[0, 1], [0, 1]])
    fig = fig.update_coloraxes(
        colorbar_title="Learning rate",
        colorscale="Viridis",
        showscale=True,
        cmin=0.0,
        cmax=1.0,
    )
    fig = fig.update_yaxes(title="Win rate", range=(0, 1))
    fig = fig.update_xaxes(title="Trial")
    fig = fig.update_layout(
        template="plotly_white",
        margin=dict(t=20, l=10, b=10, r=10),
        width=1000,
        height=400,
    )
    fig.write_image(figures_dir.joinpath("simulation_behaviour.png"), scale=2)
    fig.show()


if __name__ == "__main__":
    main()
