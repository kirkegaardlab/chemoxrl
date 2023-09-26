import jax
import jax.numpy as jnp
import numpy as np

from chemoxrl.cell import MultiCell

def evaluate_policy(model, num_envs, seed):
    rng, rng_reset = jax.random.split(jax.random.PRNGKey(seed))
    eval_envs = MultiCell(model.env_params, num_envs)

    @jax.jit
    def transition_step(carry, rng):
        s, env_state, h = carry
        rng_action, rng_step = jax.random.split(rng, 2)
        a, h = model.policy(rng_action, s, h)
        next_s, next_env_state, _, done, __ = eval_envs.step(rng_step, env_state, a)
        h = jnp.where(done[:, None], jnp.zeros_like(h), h)
        return (next_s, next_env_state, h), done

    obs, env_state = eval_envs.reset(rng_reset)
    h = jnp.zeros((*obs.shape[:-1], model.hidden_size))
    rngs = jax.random.split(rng, model.env_params.max_steps_in_episode)
    _, dones = jax.lax.scan(transition_step, init=(obs, env_state, h), xs=rngs)

    # Compute the arrival times of the simulations
    last_step = jnp.argmax(dones, axis=0)
    times = last_step[last_step > 0] * model.env_params.dt

    # Compute the chemotatic efficiency of the policy
    rate = np.sqrt(model.env_params.decay_rate / model.env_params.diffusion_coeff)
    delta = -np.log(0.9) / rate
    d_init = np.sqrt(np.sum(env_state.x**2, axis=-1))
    efficiencies = (d_init - delta) / (model.env_params.speed * times)

    return times, efficiencies
