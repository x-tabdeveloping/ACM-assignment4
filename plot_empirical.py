from pathlib import Path

import jax.numpy as jnp
import numpyro.distributions as dist
import plotly.express as px
import plotly.graph_objects as go
from jax.scipy.stats import gaussian_kde
from plotly.subplots import make_subplots

from plot_simulations import load_result_files
from utils.plots import plot_predictives


def main():
    figures_dir = Path("figures")
    figures_dir.mkdir(exist_ok=True)
    res_files = Path("results/empirical").glob("res_*.joblib")
    data = load_result_files(res_files)

    n_grid_points = 200
    n_cols = 8
    n_data = len(data["prior_predictive"])
    n_rows = n_data // n_cols
    if n_data % n_cols:
        n_rows += 1
    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=[f"ID={sid}" for sid in data["subject_id"]],
        horizontal_spacing=0.02,
        vertical_spacing=0.2,
    )
    color_scheme = px.colors.qualitative.Dark24
    for i, posterior in enumerate(data["samples"]):
        color = color_scheme[i]
        posterior_lr = jnp.ravel(posterior["lr"])
        grid = jnp.linspace(0, 3, n_grid_points)
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
        height=300,
    )
    fig.write_image(figures_dir.joinpath("emp_prior_posterior_update.png"), scale=2)
    fig.show()

    fig = go.Figure()
    lrs = jnp.array([jnp.mean(jnp.ravel(samples["lr"])) for samples in data["samples"]])
    colors = px.colors.sample_colorscale("Viridis", (lrs / lrs.max()).tolist())
    for i in range(len(data["samples"])):
        ys = data["ys"][i]
        cs = data["choices"][i]
        color = colors[i]
        wins = jnp.all(ys == cs, axis=1)
        trials = jnp.arange(len(wins))
        win_rate = jnp.cumsum(wins) / (trials + 1)
        fig = fig.add_scatter(
            x=trials,
            y=win_rate,
            name=data["subject_id"][i],
            line=dict(color=color),
            showlegend=False,
        )
    fig = fig.add_heatmap(opacity=0.0, coloraxis="coloraxis", z=[[0, 1], [0, 1]])
    fig = fig.update_coloraxes(
        colorbar_title="Estimated learning rate",
        colorscale="Viridis",
        showscale=True,
        cmin=0.0,
        cmax=float(lrs.max()),
    )
    fig = fig.update_yaxes(title="Win rate", range=(0, 1))
    fig = fig.update_xaxes(title="Trial")
    fig = fig.update_layout(
        template="plotly_white",
        margin=dict(t=20, l=10, b=10, r=10),
        width=1000,
        height=400,
    )
    fig.write_image(figures_dir.joinpath("emp_behaviour.png"), scale=2)
    fig.show()

    fig = plot_predictives(
        data["prior_predictive"],
        data["posterior_predictive"],
        data["choices"],
        names=[f"ID={sid}" for sid in data["subject_id"]],
    )
    fig = fig.update_layout(width=1400, height=400)
    fig.write_image(figures_dir.joinpath("emp_ppc.png"), scale=2)
    fig.show()


if __name__ == "__main__":
    main()
