from itertools import islice
from pathlib import Path

import jax
import jax.numpy as jnp
import joblib
import numpy as np
import numpyro
import pandas as pd
from jax.nn import one_hot

from utils.agents import naive_bayes, sample_predictives

numpyro.set_host_device_count(4)


def preprocess_data(subject_data: pd.DataFrame, n_features=5):
    subject_data = subject_data.sort_values("trial")
    ys, xs, cs = [], [], []
    for _, row in subject_data.iterrows():
        digits = [int(digit) for digit in str(row["stimulus"])]
        x = np.pad(digits, pad_width=(n_features - len(digits), 0), constant_values=0)
        xs.append(x)
        choice = 0 if row["choice"] == "ignore" else 1
        y = choice if row["correct"] == 1 else 1 - choice
        cs.append(choice)
        ys.append(y)
    xs = jnp.stack(xs)
    ys = one_hot(jnp.array(ys), num_classes=2)
    cs = one_hot(jnp.array(cs), num_classes=2)
    return xs, ys, cs


def main():
    res_dir = Path("results/empirical/")
    res_dir.mkdir(exist_ok=True, parents=True)
    data = pd.read_csv("dat/TrainingAlienData.csv")
    # shuffling dataframe
    data = data.sample(frac=1, random_state=42)
    key = jax.random.key(0)
    subjects = list(islice(data.groupby("unique_id"), 15))
    i = 1
    for s_id, s_data in subjects:
        print(f"===============SUBJECT {s_id} ({i}/{len(subjects)})==================")
        xs, ys, cs = preprocess_data(s_data)
        agent = naive_bayes.add_input(xs, ys)
        key, subkey = jax.random.split(key)
        key, subkey = jax.random.split(key)
        mcmc = agent.condition_on(cs).sample_posterior(subkey)
        key, subkey = jax.random.split(key)
        prior_predictive, posterior_predictive = sample_predictives(
            subkey, agent, mcmc.get_samples()
        )
        data = dict(
            subject_id=s_id,
            samples=mcmc.get_samples(),
            choices=cs,
            xs=xs,
            ys=ys,
            prior_predictive=prior_predictive,
            posterior_predictive=posterior_predictive,
        )
        joblib.dump(data, res_dir.joinpath(f"res_{s_id}.joblib"))
        i += 1
    print("DONE")


if __name__ == "__main__":
    main()
