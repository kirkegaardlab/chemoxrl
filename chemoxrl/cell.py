from functools import partial
from typing import Union

from flax.struct import dataclass
import jax
import jax.numpy as jnp


@dataclass
class EnvParams:
    max_steps_in_episode: int = 256
    radius: float = 4  # Radius of the cell (μm).
    n_receptors: int = 8  # Number of discrete receptors along surface.
    speed: float = 4  # Swimming speed (μm/s).
    rotational_diffusion: float = 0.025  # Rotational diffusion constant (1/s).
    decay_rate: float = 0.01  # Average decay rate for the simulation.
    diffusion_coeff: float = 100  # Average diffusion coefficient for the simulation.
    dt: float = 0.1  # Time step of the simulation.
    C_min: int = 4 # Minimum value of the exponent for the concentration.
    C_max: int = 5 # Maximum value of the exponent for the concentratino.


@dataclass
class CellState:
    step: int  # Step count in the episode.
    x: jax.Array  # Coordinates of the cell.
    theta: jax.Array  # Orientation of the cell.
    cum_reward: float  # Cumulative reward at R(t<=time)
    N: Union[jax.Array, int]  # Number of particles
    d_init: jax.Array  # Initial distance to the center (useful for reward)


def polar_to_cartesian(r, phi):
    x = r * jnp.cos(phi)
    y = r * jnp.sin(phi)
    return jnp.array([x, y])


def cartesian_to_polar(x, y):
    r = jnp.sqrt(x**2 + y**2)
    phi = jnp.arctan2(y, x)
    return r, phi


def gradient(env_params):
    return jnp.sqrt(env_params.decay_rate / env_params.diffusion_coeff)


class Cell:
    @partial(jax.jit, static_argnums=(0, 2))
    def reset(self, rng, env_params):
        rng_media, rng_cell = jax.random.split(rng, 2)
        Cs = jnp.logspace(env_params.C_min, env_params.C_max, num=100, dtype=jnp.int32)
        N = jax.random.choice(rng_cell, Cs, shape=())

        x, theta = self._init_cell(rng_cell, env_params)
        d_init = jnp.hypot(x[0], x[1])
        state = CellState(step=0, x=x, theta=theta, cum_reward=0.0, N=N, d_init=d_init)

        obs = self._get_obs(rng_media, state, env_params)
        obs = jnp.concatenate((obs, jnp.zeros(1)))
        return obs, state

    @partial(jax.jit, static_argnums=(0, 4))
    def step(self, rng, state, action, env_params):
        # Steps returns values until a done happens. Then it returns 0s.
        # until the next self.reset is called.
        obs_st, state_st, reward, done = self._step(rng, state, action, env_params)
        state_re = jax.tree_util.tree_map(lambda x: x*0, state_st)
        state = jax.tree_util.tree_map(lambda x, y: jax.lax.select(done, x, y), state_re, state_st)
        obs_re = jnp.zeros(self.observation_space(env_params).shape)
        obs = jax.lax.select(done, obs_re, obs_st)
        return obs, state, reward, done, {}

    def _init_cell(self, rng, env_params):
        rng_theta, rng_r, rng_phi = jax.random.split(rng, num=3)
        theta = jax.random.uniform(rng_theta) * 2 * jnp.pi
        percentile = jax.random.uniform(rng_r, minval=0.3, maxval=0.5)
        r = -jnp.log(1 - percentile) / gradient(env_params)
        phi = jax.random.uniform(rng_phi) * 2 * jnp.pi
        x = polar_to_cartesian(r, phi)
        return x, theta

    def _step(self, rng, state, action, env_params):
        rng_o, rng_a = jax.random.split(rng, 2)
        noise = jax.random.normal(rng_a) * jnp.sqrt(2 * env_params.rotational_diffusion * env_params.dt)
        dtheta = action[0] * jnp.pi + noise
        theta = (state.theta + dtheta) % (2 * jnp.pi)
        v = polar_to_cartesian(env_params.speed, theta)
        x = state.x + v * env_params.dt
        state = state.replace(x=x, theta=theta)

        obs = self._get_obs(rng_o, state, env_params)
        state = state.replace(step=state.step + 1)

        final_reward = self._get_reward(state, env_params)
        has_reached = jnp.hypot(x[0], x[1]) <= (-jnp.log(0.9) / gradient(env_params))
        done = (state.step >= env_params.max_steps_in_episode) | has_reached
        reward = jax.lax.select(done, final_reward, 0.0)
        state = state.replace(cum_reward=state.cum_reward + reward)

        obs = jnp.concatenate((obs, action))
        return obs, state, reward, done

    def _get_obs(self, rng, state, env_params):
        M = env_params.n_receptors
        a = env_params.radius

        angles = jnp.arange(M) * ((2 * jnp.pi) / M) + state.theta
        receptors = state.x + a * jnp.array([jnp.cos(angles), jnp.sin(angles)]).T
        sensor_area = (a * jnp.sin(jnp.pi / M)) ** 2 * jnp.pi
        B = (state.N / (2 * jnp.pi)) * sensor_area  # integration constant
        rate = gradient(env_params)

        @jax.vmap
        def detect(rng, xi):
            d = jnp.hypot(xi[0], xi[1])
            c = rate * jnp.exp(-rate * d)
            M_avg = B * c
            M = jax.random.poisson(rng, M_avg)
            m = jnp.log(M + 1)
            return m

        m = detect(jax.random.split(rng, M), receptors)
        return m

    def _get_reward(self, state, env_params):
        max_steps = env_params.max_steps_in_episode
        d = jnp.hypot(state.x[0], state.x[1])
        d_min = -jnp.log(0.9) / gradient(env_params)
        distance_reward = jnp.clip((d_min - d) / (state.d_init - d_min), -1.0, 0.0)
        time_reward = jnp.clip((max_steps - state.step) / max_steps, 0.0, 1.0)
        return distance_reward + time_reward

    @property
    def num_actions(self):
        return 1

    def observation_space(self, env_params):
        return jnp.empty(shape=(env_params.n_receptors + 1,))


class MultiCell:
    def __init__(self, env_params, n_envs):
        self.env = Cell()
        self.env_params = env_params
        self.n_envs = n_envs
        self.num_actions = self.env.num_actions
        self.observation_space = self.env.observation_space

    @partial(jax.jit, static_argnums=0)
    def reset(self, rng):
        rngs = jax.random.split(rng, self.n_envs)
        return jax.vmap(self.env.reset, in_axes=(0, None))(rngs, self.env_params)

    @partial(jax.jit, static_argnums=0)
    def step(self, rng, env_state, actions):
        rngs = jax.random.split(rng, self.n_envs)
        batched_step = jax.vmap(self.env.step, in_axes=(0, 0, 0, None))
        return batched_step(rngs, env_state, actions, self.env_params)
