from collections import defaultdict
from pathlib import Path

import jax.numpy as jnp
import joblib
import numpyro.distributions as dist
import plotly.express as px
from jax.scipy.stats import gaussian_kde
from plotly.subplots import make_subplots

from utils.plots import plot_predictives, plot_recovery

figures_dir = Path("figures")
figures_dir.mkdir(exist_ok=True)

sim_files = list(Path("results/").glob("simulation_*.joblib"))
file_ids = jnp.array([int(sf.stem.split("_")[1]) for sf in sim_files])
order = jnp.argsort(file_ids)
data = defaultdict(list)
for i_file in order:
    exp_data = joblib.load(sim_files[i_file])
    for key, value in exp_data.items():
        data[key].append(value)


fig = plot_recovery(data["samples"], jnp.array(data["lr"]))
fig = fig.update_traces(showlegend=False)
fig = fig.update_layout(width=800, height=300)
fig.show()

fig = plot_predictives(
    data["prior_predictive"], data["posterior_predictive"], data["choices"], data["lr"]
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
    template="plotly_white", margin=dict(t=20, l=10, b=10, r=10), width=1000, height=400
)
fig.show()
