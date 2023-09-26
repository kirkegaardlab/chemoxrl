import jax
import jax.numpy as jnp

import chemoxrl
from chemoxrl.cell import Cell


def random_walk_policy(n=4):
    options = jnp.linspace(0, n, endpoint=False) / n

    def policy(rng, _, h):
        dtheta = jax.random.choice(rng, options)
        return (jnp.array([dtheta]), h)

    return jax.jit(jax.vmap(policy, in_axes=(None, 0, 0))), 1


def greedy_policy(env_params, eps=0.1, steps=1, kernel="lin", adapt=False, gamma=1.0,):
    M = env_params.n_receptors
    phi = jnp.arange(M) * 2 * jnp.pi / M

    hidden_size = steps * M

    if kernel == "exp":
        K = jnp.exp(-gamma*jnp.arange(steps))[:, jnp.newaxis]
    else:
        K = jnp.ones((steps, 1))

    interp = jax.vmap(lambda x, c: jnp.interp(phi, x, c, period=(2 * jnp.pi)), in_axes=(None, 0))

    def policy(rng, s, h):
        m = s[:M]
        h = jnp.reshape(h, (steps, M))
        if adapt:
            theta = phi - s[-1] * jnp.pi
            h = interp(theta, h)

        h = jnp.roll(h, 1, axis=0)
        h = h.at[0].set(m)
        weights = jnp.mean(h*K, axis=0)
        ex = jnp.sum(jnp.cos(phi) * weights)
        ey = jnp.sum(jnp.sin(phi) * weights)
        dtheta = jnp.arctan2(ey, ex) / jnp.pi
        dtheta = jnp.clip(dtheta, -eps, eps)
        a = jnp.array([dtheta])
        return a, h.flatten()

    _select_action_fn = jax.vmap(policy, in_axes=(None, 0, 0))
    _select_action_fn = jax.jit(_select_action_fn)
    return _select_action_fn, hidden_size


def rppo_policy(params, deterministic=False, spatial=False, memory=False, recurrent=False):
    model = chemoxrl.ActorCritic(Cell().num_actions, spatial=spatial, memory=memory, recurrent=recurrent)

    def policy(rng, s, h):
        outs = model.apply(params, s[None], h)
        _, (mu, scale), h = jax.tree_util.tree_map(lambda x: x[-1], outs)
        a = jax.random.normal(rng, shape=mu.shape) * scale + mu
        a = jax.lax.select(deterministic, mu, a)
        return a, h

    return jax.jit(policy)
