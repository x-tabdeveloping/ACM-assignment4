import firetruck as ft
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
import plotly.express as px
from jax.scipy.special import logsumexp
from numpyro.handlers import seed, substitute, trace

numpyro.set_host_device_count(4)


def trace_conditionals(xs, labels, init_conc, lr):
    def _update(c, d):
        c = c.at[d["l"]].add(lr * d["x"])
        return c, c

    _, concs = jax.lax.scan(_update, init_conc, {"x": xs, "l": labels})
    concs = jnp.insert(concs, 0, init_conc, axis=0)[:-1]
    return concs


def trace_marginals(ys, init_conc, lr):
    def _update(c, y):
        c += lr * y
        return c, c

    _, concs = jax.lax.scan(_update, init_conc, ys)
    concs = jnp.insert(concs, 0, init_conc, axis=0)[:-1]
    return concs


def simulate_behaviour(rng_key, agent, parameters):
    agent = seed(substitute(agent, parameters), rng_key)
    agent_trace = trace(agent).get_trace()
    return agent_trace["obs"]["value"], agent_trace


def trace_feature_beliefs(xs, ys, lr):
    n_trials, n_features = xs.shape
    n_labels = ys.shape[1]
    c0 = jnp.ones((2, n_labels, n_features))
    labels = jnp.argmax(ys, axis=1)

    def _update(c, d):
        c = c.at[0, d["l"]].add(lr * d["x"])
        c = c.at[1, d["l"]].add(lr * (1 - d["x"]))
        return c, c

    # cs has shape (n_trials, [a, b], n_labels, n_features)
    _, cs = jax.lax.scan(_update, c0, {"x": xs, "l": labels})
    # Inserting c0 to 0 place so that we don't leak information about
    # the next trial
    cs = jnp.insert(cs, 0, c0, axis=0)[:-1]
    return cs


def feature_log_prob(cs, xs):

    def get_naive_prob(label_c):
        # label_c has shape (n_labels, n_trials, [a, b], n_features)
        # transposing and decomposing to ([a,b], n_labels, n_trials, n_features)
        a, b = jnp.transpose(label_c, (1, 0, 2))
        return jnp.sum(dist.BetaBinomial(a, b).log_prob(xs), axis=1)

    # Transposing to (n_labels, n_trials, [a, b], n_features), then running it into vmap
    feature_prob = jax.vmap(get_naive_prob)(jnp.transpose(cs, (2, 0, 1, 3)))
    return feature_prob.T


@ft.compact
def naive_bayes(self, xs, ys):
    n_labels = ys.shape[1]
    self.lr = dist.HalfNormal(1.0)
    self.feature_beliefs = trace_feature_beliefs(
        xs,
        ys,
        lr=self.lr,
    )
    self.label_beliefs = trace_marginals(
        ys,
        init_conc=jnp.ones(n_labels),
        lr=self.lr,
    )
    self.feature_log_p = feature_log_prob(self.feature_beliefs, xs)
    self.label_log_p = jax.vmap(
        lambda c: dist.DirichletMultinomial(c).log_prob(jnp.eye(n_labels))
    )(self.label_beliefs)
    log_prob = self.feature_log_p + self.label_log_p
    norm = logsumexp(log_prob, axis=1)
    self.log_p = log_prob - norm[:, jnp.newaxis]
    return dist.Multinomial(logits=self.log_p)
