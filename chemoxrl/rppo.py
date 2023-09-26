from functools import partial
from typing import NamedTuple

from flax.core.frozen_dict import freeze
import flax.linen as nn
from flax.training import checkpoints
from flax.training.train_state import TrainState
import jax
import jax.numpy as jnp
import optax
from tqdm import tqdm

from chemoxrl.cell import MultiCell


class ExperimentConfig(NamedTuple):
    n_train_envs: int = 256  # Number of train environments in parallel.
    total_steps: int = int(10e6)  # Number of train steps.
    n_steps: int = 301  # Number of steps per environment before training.
    max_grad_norm: float = 0.5  # Global norm to clip gradients by.
    eval_interval: int = 1_000_000  # Number of steps until an evaluation is done.
    n_eval_envs: int = 2048  # Number of parallel envs to evaluate on.
    learning_rate: float = 3e-4  # Learning rate.
    n_epochs: int = 4  # Number of epochs at each training step.
    n_minibatch: int = 4  # Number of minibatches to split the buffer into.
    clip_eps: float = 0.2  # Surrogate clipping loss.
    entropy_coeff: float = 0.00  # Entropy loss coefficient
    critic_coeff: float = 0.5  # Value loss coefficient
    discount: float = 0.99  # Discount factor for the GAE calculation.
    gae_lambda: float = 0.95  # "GAE lambda"
    logdir: str = "./logs/"  # Path to store the logs.
    load: str = ""  # Load a previous policy before training.
    seed: int = 118  # Random state seed.
    hidden_cells: int = 25 # Neurons in the recurrent network.
    spatial: bool = True
    memory: bool = True
    recurrent: bool = True


class Rollout(NamedTuple):
    states: jax.Array
    actions: jax.Array
    rewards: jax.Array
    dones: jax.Array
    log_probs: jax.Array
    values: jax.Array
    hidden_state: jax.Array
    mask: jax.Array


class MemoryCell(nn.Module):
    decay: float = 0.1

    @nn.vmap
    @nn.compact
    def __call__(self, h, x):
        x = jnp.concatenate((x, x.mean(keepdims=True)))
        h = h.reshape(-1, 3)

        k = self.decay
        h_new = jnp.empty_like(h)
        h_new = h_new.at[:, 0].set(k * x + (1 - k) * h[:, 0])
        h_new = h_new.at[:, 1].set(k * h[:, 0] + (1 - k) * h[:, 1])
        h_new = h_new.at[:, 2].set(k * h[:, 1] + (1 - k) * h[:, 2])

        # Normalize
        x_normalized = x.at[:-1].add(-x[-1])
        h_new_normalized = h_new.at[:-1, :].add(-h_new[-1, :])
        h_new_normalized = h_new_normalized.at[-1, 0].add(-x[-1])
        h_new_normalized = h_new_normalized.at[-1, 1].add(-h_new[-1, 0])
        h_new_normalized = h_new_normalized.at[-1, 2].add(-h_new[-1, 1])

        h_new = h_new.flatten()
        h_new_normalized = h_new_normalized.flatten()
        x_input = jnp.concatenate((x_normalized, h_new_normalized))
        return h_new, (h_new, x_input)


class ActorCritic(nn.Module):
    num_output_units: int
    num_hidden_units: int = 64
    num_hidden_layers: int = 2
    min_std: float = 0.05
    max_std: float = 1.0
    spatial: bool = True
    memory: bool = False
    recurrent: bool = False

    @nn.compact
    def __call__(self, xs, h_init):
        init_fn = nn.initializers.orthogonal(scale=jnp.sqrt(2))
        init_fn_actor = nn.initializers.orthogonal(scale=0.01)

        if not self.spatial:
            # All input sensors report the mean (last element is previous action)
            xs = xs.at[..., :-1].set(xs[..., :-1].mean(axis=-1, keepdims=True))

        if self.memory:
            if self.recurrent:
                F = nn.scan(nn.GRUCell, variable_broadcast="params", split_rngs={"params": False})
                _, hs = F()(h_init, xs)
                x_input  = hs
            else:
                xs = xs[..., :-1] # remove the action from the input.
                F = nn.scan(MemoryCell)
                _, (hs, x_input) = F()(h_init, xs)
        else:
            hs = nn.Dense(h_init.shape[-1], kernel_init=init_fn, name='feature_ext')(xs)
            x_input = hs

        # Critic network
        x_v = x_input
        for _ in range(self.num_hidden_layers):
            x_v = nn.Dense(self.num_hidden_units*2, kernel_init=init_fn)(x_v)
            x_v = nn.tanh(x_v)
        value = nn.Dense(1, kernel_init=nn.initializers.orthogonal())(x_v)

        # Actor Network
        x_a = x_input
        for _ in range(self.num_hidden_layers):
            x_a = nn.Dense(self.num_hidden_units, kernel_init=init_fn)(x_a)
            x_a = nn.tanh(x_a)

        mu = nn.Dense(self.num_output_units, kernel_init=init_fn_actor)(x_a)
        log_scale = nn.Dense(self.num_output_units, kernel_init=init_fn_actor)(x_a)
        scale = jax.lax.clamp(self.min_std, jax.nn.softplus(-0.5 + log_scale), self.max_std)
        return value, (mu, scale), hs


def loss_fn(params, apply_fn, minibatch, eps=0.2, entropy_coeff=0.001, vf_coeff=0.5):
    s, a, logp_old, target, A, h, mask = minibatch
    values, (mu, scale), _ = apply_fn(params, s, h[0])
    values = values[..., 0]
    num_entries = jnp.sum(mask)

    # Compute Clipped Surrogate Loss on the policy.
    # Normalize the advantage seems to help.
    Am = A * mask
    Amean = jnp.sum(Am) / num_entries
    Astd = jnp.sqrt(jnp.sum(Am**2) / num_entries - Amean**2)
    A = (A - Amean) / (Astd + 1e-8)
    logp = jax.scipy.stats.norm.logpdf(a, loc=mu, scale=scale).sum(-1)
    ratio = jnp.exp(logp - logp_old)
    policy_loss = jnp.minimum(ratio * A, jnp.clip(ratio, 1.0 - eps, 1.0 + eps) * A)
    policy_loss = -jnp.sum(mask * policy_loss) / num_entries

    # Critic MSE loss.
    value_loss = jnp.sum(mask * (values - target) ** 2) / num_entries

    # Entropy of normal distribution is H = -p*ln(p) = 1/2 ln(e*2π*σ^2)
    entropy = jnp.sum(0.5 + 0.5 * jnp.log(2 * jnp.pi) + jnp.log(scale), axis=-1)
    entropy_loss = jnp.sum(mask * entropy) / num_entries

    # Compute KL divergence (approximation)
    approx_kl_div = jnp.sum(mask * ((ratio - 1) - (logp - logp_old))) / num_entries

    loss = policy_loss + vf_coeff * value_loss - entropy_coeff * entropy_loss
    aux = (loss, policy_loss, value_loss, entropy_loss, approx_kl_div)
    return loss, aux


@partial(jax.jit, static_argnums=(3, 4))
def train_step(rng, train_state, batch, n_epochs=8, n_minibatch=8):

    buffer_size = batch[0].shape[1]

    grad_fn = jax.grad(loss_fn, has_aux=True)

    def epoch_step(state, rng):
        def batch_step(state, chosen):
            minibatch = jax.tree_util.tree_map(lambda x: x[:, chosen], batch)
            grads, metrics = grad_fn(state.params, state.apply_fn, minibatch)
            state = state.apply_gradients(grads=grads)
            return state, metrics

        batch_indices = jax.random.permutation(rng, buffer_size)
        batch_indices = batch_indices.reshape(n_minibatch, -1)
        state, metrics = jax.lax.scan(batch_step, init=state, xs=batch_indices)
        return state, jax.tree_util.tree_map(jnp.mean, metrics)

    rngs = jax.random.split(rng, n_epochs)
    train_state, metrics = jax.lax.scan(epoch_step, init=train_state, xs=rngs)
    return train_state, jax.tree_util.tree_map(jnp.mean, metrics)


def calculate_gae(values, rewards, dones, discount=0.99, A_lambda=0.95):
    def body_fn(A, x):
        next_value, done, value, reward = x
        value_diff = discount * next_value * (1 - done) - value
        delta = reward + value_diff
        A = delta + discount * A_lambda * (1 - done) * A
        return A, A

    xs = (values[1:], dones[:-1], values[:-1], rewards[:-1])
    num_envs = values.shape[1]
    _, gae = jax.lax.scan(body_fn, jnp.zeros(num_envs), xs, reverse=True)
    gae = jnp.pad(gae, pad_width=((0, 1), (0, 0)))
    return gae


@partial(jax.jit, static_argnums=(1, 2, 3))
def collect_batch(buffer, discount=0.99, A_lambda=0.95, permutate=True):
    gae = calculate_gae(buffer.values, buffer.rewards, buffer.dones, discount, A_lambda)
    target = gae + buffer.values
    batch = (buffer.states, buffer.actions, buffer.log_probs, target, gae, buffer.hidden_state, buffer.mask)

    # Remove the last element of the batch since it doesn't have advantage to compute.
    batch = jax.tree_util.tree_map(lambda x: x[:-1], batch)

    if permutate:
        batch = jax.tree_util.tree_map(lambda x: x.reshape(-1, *x.shape[2:]), batch)
        batch = jax.tree_util.tree_map(lambda x: x[None, ...], batch)

    return batch


def init_train_state(rng, env, env_params, config):
    model = ActorCritic(num_output_units=env.num_actions,
                        spatial=config.spatial,
                        memory=config.memory,
                        recurrent=config.recurrent)

    obs_shape = env.observation_space(env_params).shape
    dummy_x = jnp.ones((1, 1, *obs_shape))
    hidden_state = jnp.zeros((1, config.hidden_cells))
    params = model.init(rng, dummy_x, hidden_state)

    if config.load:
        # Overwrite the initial parameters with the loaded ones but 
        # leave the scale layer as initialised.
        ckpt = checkpoints.restore_checkpoint(config.load, target=None)
        new_params = {**params, **ckpt['params']}
        new_params['params']['Dense_6'] = params['params']['Dense_6']
        params = freeze(new_params)

    tx = optax.adam(config.learning_rate, eps=1e-7)
    return TrainState.create(apply_fn=model.apply, params=params, tx=tx)


@partial(jax.jit, static_argnums=(1, 2, 3, 4))
def evaluate(state, env_params, num_envs=1024, hidden_size=24, deterministic=True):
    eval_envs = MultiCell(env_params, num_envs)
    eval_rng = jax.random.PRNGKey(0)

    def transition_step(carry, rng):
        s, env_state, h = carry
        rng_action, rng_step = jax.random.split(rng, 2)
        out = state.apply_fn(state.params, s[None, ...], h)
        _, (mu, scale), next_h = jax.tree_util.tree_map(lambda x: x[-1], out)
        a = jax.random.normal(rng_action, shape=mu.shape) * scale + mu
        a = jax.lax.select(deterministic, mu, a)
        next_s, next_env_state, r, done, __ = eval_envs.step(rng_step, env_state, a)
        return (next_s, next_env_state, next_h), (r, done)

    obs, env_state = eval_envs.reset(eval_rng)
    h = jnp.zeros((*obs.shape[:-1], hidden_size))
    rngs = jax.random.split(eval_rng, env_params.max_steps_in_episode)
    _, (rewards, dones) = jax.lax.scan(transition_step, (obs, env_state, h), rngs)

    end_step = jax.vmap(lambda x: jnp.argwhere(x, size=1)[...,0], in_axes=1)(dones)[...,0]
    mean_reward = rewards[end_step, jnp.arange(num_envs)].mean()
    reached_percentage = (end_step < env_params.max_steps_in_episode-1).mean()
    arrival_time = end_step * env_params.dt
    return mean_reward, reached_percentage, arrival_time


def train_loop(rng, config, env_params, checkpointer):

    # Initialise the batched environment.
    envs = MultiCell(env_params, config.n_train_envs)

    # Initialise the network
    rng, rng_init = jax.random.split(rng, num=2)
    train_state = init_train_state(rng_init, envs, env_params, config)

    @jax.jit
    def run_episodes(train_state, rng):

        def step(carry, rng):
            train_state, obs, env_state, h, mask = carry
            rng_action, rng_step = jax.random.split(rng, 2)
            out = train_state.apply_fn(train_state.params, obs[None, ...], h)
            value, (mu, scale), next_h = jax.tree_util.tree_map(lambda x: x[-1], out)
            action = jax.random.normal(rng_action, shape=mu.shape) * scale + mu
            log_prob = jax.scipy.stats.norm.logpdf(action, mu, scale).sum(-1)
            next_obs, next_env_state, reward, done, _ = envs.step(rng_step, env_state, action)
            rollout = Rollout(obs, action, reward, done, log_prob, value[..., 0], h, 1-mask)
            return (train_state, next_obs, next_env_state, next_h, mask | done), rollout

        rng, rng_reset = jax.random.split(rng, 2)
        rngs = jax.random.split(rng, env_params.max_steps_in_episode+1)

        obs, env_state = envs.reset(rng_reset)
        hidden_state = jnp.zeros(shape=(*obs.shape[:-1], config.hidden_cells))
        mask = jnp.zeros(config.n_train_envs) > 0
        init = (train_state, obs, env_state, hidden_state, mask)
        _, buffer = jax.lax.scan(step, init, rngs)
        return buffer

    step_size = int(config.n_train_envs * config.n_steps)
    eval_interval = max(1, int(config.total_steps / 100 // step_size))
    pbar = tqdm(total=config.total_steps, dynamic_ncols=True)

    steps_passed = 0
    for step in range(config.total_steps // step_size):

        # Perform n_steps for each of the num_train_envs in parallel.
        buffer = run_episodes(train_state, jax.random.fold_in(rng, step))
        steps_passed += step_size

        # Collect the buffer from the rollout and perform a training step corresponding
        # to n_minibatch*n_epochs gradient descent steps with the current policy.
        rng_train, rng = jax.random.split(rng, 2)
        batch = collect_batch(buffer, config.discount, config.gae_lambda, not config.memory)
        train_state, metrics = train_step(rng_train, train_state, batch, config.n_epochs, config.n_minibatch)
        metrics = jax.tree_util.tree_map(jnp.mean, metrics)

        logging_step = int(step * step_size)
        checkpointer.writer.scalar("train/loss", metrics[0], logging_step)
        checkpointer.writer.scalar("train/policy_loss", metrics[1], logging_step)
        checkpointer.writer.scalar("train/critic_loss", metrics[2], logging_step)
        checkpointer.writer.scalar("train/entropy_loss", metrics[3], logging_step)
        checkpointer.writer.scalar("train/kl_divergence", metrics[4], logging_step)

        if step % eval_interval == 0:
            eval_metrics = evaluate(train_state, env_params, config.n_eval_envs, config.hidden_cells, False)
            metrics = jax.tree_util.tree_map(jnp.mean, eval_metrics)
            checkpointer.writer.scalar("eval/reward", metrics[0], logging_step)
            checkpointer.writer.scalar("eval/reached", metrics[1], logging_step)
            checkpointer.writer.scalar("eval/ftp", metrics[2], logging_step)
            pbar.set_description(f"R: {metrics[0]:.2f}({metrics[1]:.2f})")
            pbar.update(steps_passed)
            steps_passed = 0

            ckpt = {"params": train_state.params}
            checkpoints.save_checkpoint(checkpointer.dir, target=ckpt, step=logging_step, keep_every_n_steps=int(1e6), keep=10)

    return train_state
