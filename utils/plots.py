import firetruck as ft
import jax.numpy as jnp
import plotly.graph_objects as go
from numpyro import diagnostics
from plotly.subplots import make_subplots
from tqdm import tqdm


def plot_forest(posterior_samples, true_lrs=None) -> go.Figure:
    fig = go.Figure()
    for i, samples in enumerate(posterior_samples):
        lr_median = jnp.median(samples["lr"])
        lr_lower, lr_upper = diagnostics.hpdi(samples["lr"], prob=0.95)
        if true_lrs is not None:
            x = true_lrs[i]
        else:
            x = i
        fig.add_scatter(
            x0=x,
            y=[lr_median],
            error_y=dict(
                type="data",
                symmetric=False,
                array=[lr_upper - lr_median],
                arrayminus=[lr_median - lr_lower],
                width=0,
                thickness=3,
            ),
            showlegend=False,
            mode="markers",
            marker=dict(size=12),
        )
    if true_lrs is not None:
        fig.add_scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            line=dict(color="black", dash="dash", width=2.0),
            name="True Learning Rate",
        )
    fig = fig.update_layout(template="plotly_white", margin=dict(t=20, b=0, l=0, r=0))
    return fig


def plot_predictives(
    prior_predictives: list,
    posterior_predictives: list,
    choices: list,
    names: list,
):
    model_names = names
    subplot_titles = []
    for t in ["prior", "posterior"]:
        for model_name in model_names:
            subplot_titles.append(f"{model_name}")
    fig = make_subplots(
        rows=2,
        cols=len(prior_predictives),
        subplot_titles=subplot_titles,
        horizontal_spacing=0.02,
        vertical_spacing=0.1,
    )
    i_model = 0
    for model_name in tqdm(model_names, desc="Adding subplots for all models"):
        subfig = ft.plot_predictive_check(
            prior_predictives[i_model],
            obs=jnp.argmax(choices[i_model], axis=-1).astype(int),
        )
        for trace in subfig.data:
            trace.showlegend = (i_model == 0) and trace.showlegend
            fig.add_trace(
                trace,
                col=i_model + 1,
                row=1,
            )
        subfig = ft.plot_predictive_check(
            posterior_predictives[i_model],
            obs=jnp.argmax(choices[i_model], axis=-1).astype(int),
        )
        for trace in subfig.data:
            trace.showlegend = False
            fig.add_trace(trace, col=i_model + 1, row=2)
        i_model += 1
    fig = fig.update_yaxes(matches="y", visible=False)
    fig = fig.update_yaxes(visible=True, col=1, row=1, title="Prior Predictive")
    fig = fig.update_yaxes(visible=True, col=1, row=2, title="Posterior Predictive")
    fig = fig.update_xaxes(visible=False)
    fig = fig.update_layout(
        template="plotly_white",
        barmode="overlay",
        margin=dict(l=10, r=10, t=30, b=10),
        font=dict(size=16),
    )
    return fig
