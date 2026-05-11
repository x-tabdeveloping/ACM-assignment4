from pathlib import Path

import jax
import jax.numpy as jnp
import joblib
import numpyro.distributions as dist
from tqdm import tqdm

from utils.agents import naive_bayes, sample_predictives, simulate_behaviour


def match_rule(x, rule):
    result = True
    if jnp.any(rule == -1):
        result = result and jnp.all(x[rule == -1] == 0)
    if jnp.any(rule == 1):
        result = result and jnp.all(x[rule == 1] == 1)
    return result


def simulate_experiment(rng_key, n_trials_per_rule=100, n_features=5, n_rules=5):
    key = rng_key
    key, subkey = jax.random.split(key)
    # -1 means feature has to be negative, 0 means ignore, 1 means it has to be positive
    rules = jax.random.choice(
        subkey,
        jnp.array([-1, 0, 1]),
        p=jnp.array([0.15, 0.7, 0.15]),
        shape=(n_rules, n_features),
    )
    key, subkey = jax.random.split(key)
    feature_probs = dist.Beta(
        jnp.ones((len(rules), n_features)), jnp.ones((len(rules), n_features))
    ).sample(subkey)
    xs = []
    ys = []
    for rule, p in zip(rules, feature_probs):
        for i in range(n_trials_per_rule):
            key, subkey = jax.random.split(key)
            x = dist.Bernoulli(probs=p).sample(subkey)
            matches_rule = match_rule(x, rule)
            y = jax.nn.one_hot(int(matches_rule), 2)
            ys.append(y)
            xs.append(x)
    xs = jnp.stack(xs)
    ys = jnp.stack(ys)
    return xs, ys


def main():
    res_dir = Path("results/simulations/")
    res_dir.mkdir(exist_ok=True, parents=True)
    key = jax.random.key(0)
    lrs = jnp.linspace(0, 1.0, 15)
    i = 0
    for lr in tqdm(
        lrs, desc="Fitting models with different learning rates.", disable=True
    ):
        key, subkey = jax.random.split(key)
        xs, ys = simulate_experiment(subkey)
        agent = naive_bayes.add_input(xs, ys)
        key, subkey = jax.random.split(key)
        cs, agent_trace = simulate_behaviour(
            subkey,
            agent,
            parameters=dict(lr=lr),
        )
        key, subkey = jax.random.split(key)
        mcmc = agent.condition_on(cs).sample_posterior(subkey)
        print(f"True learning rate = {lr:.2f}")
        key, subkey = jax.random.split(key)
        prior_predictive, posterior_predictive = sample_predictives(
            subkey, agent, mcmc.get_samples()
        )
        data = dict(
            lr=lr,
            samples=mcmc.get_samples(),
            choices=cs,
            xs=xs,
            ys=ys,
            prior_predictive=prior_predictive,
            posterior_predictive=posterior_predictive,
        )
        joblib.dump(data, res_dir.joinpath(f"simulation_{i}.joblib"))
        i += 1
    print("DONE")


if __name__ == "__main__":
    main()
