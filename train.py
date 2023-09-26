import argparse
from collections import namedtuple
import json
import pathlib
import time
import dataclasses

from flax.metrics import tensorboard
import jax

from chemoxrl import rppo
from chemoxrl.cell import EnvParams


parser = argparse.ArgumentParser(description="Script to run experiment.")

# ExperimentConfig parameters
parser.add_argument("--n_train_envs", type=int, default=4096)
parser.add_argument("--total_steps", type=int, default=int(1e9))
parser.add_argument("--n_steps", type=int, default=256)
parser.add_argument("--max_grad_norm", type=float, default=0.5)
parser.add_argument("--n_eval_envs", type=int, default=4096)
parser.add_argument("--learning_rate", type=float, default=3e-4)
parser.add_argument("--n_epochs", type=int, default=8)
parser.add_argument("--n_minibatch", type=int, default=8)
parser.add_argument("--clip_eps", type=float, default=0.2)
parser.add_argument("--entropy_coeff", type=float, default=0.01)
parser.add_argument("--critic_coeff", type=float, default=0.5)
parser.add_argument("--discount", type=float, default=1.0)
parser.add_argument("--gae_lambda", type=float, default=1.0)
parser.add_argument("--logdir", type=str, default="./logs/")
parser.add_argument("--load", type=str, default="")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--hidden_cells", type=int, default=25)
parser.add_argument("--memory", type=bool, default=False, action=argparse.BooleanOptionalAction)
parser.add_argument("--spatial", type=bool, default=True, action=argparse.BooleanOptionalAction)
parser.add_argument("--recurrent", default=True, action=argparse.BooleanOptionalAction)

# EnvParams parameters
parser.add_argument("--max_steps_in_episode", type=int, default=256)
parser.add_argument("--radius", type=float, default=1.0)
parser.add_argument("--n_receptors", type=int, default=5)
parser.add_argument("--speed", type=float, default=5)
parser.add_argument("--rotational_diffusion", type=float, default=0.025)
parser.add_argument("--decay_rate", type=float, default=0.01)
parser.add_argument("--diffusion_coeff", type=float, default=100)
parser.add_argument("--dt", type=float, default=0.1)
parser.add_argument("--C_min", type=int, default=4)
parser.add_argument("--C_max", type=int, default=5)


# Training flags
parser.add_argument("--noise", type=bool, default=True, action=argparse.BooleanOptionalAction)
args = parser.parse_args()

env_args = {k: v for k, v in vars(args).items() if k in EnvParams.__dict__}
if not args.noise:
    env_args["C_max"] = 9
    env_args["C_min"] = 8
env_params = EnvParams(**env_args)
print(env_params)

ec_args = {k: v for k, v in vars(args).items() if k in rppo.ExperimentConfig.__dict__}
if args.memory and not args.recurrent:
    ec_args['hidden_cells'] = 3 * (args.n_receptors+1)
config = rppo.ExperimentConfig(**ec_args)
print(config._asdict())

Checkpointer = namedtuple("Checkpointer", ["dir", "writer"])

def init_logger(config, params):
    # Create the directory with the logs and checkpoints.
    # Define the experiment name.
    cell_type = f"S{'T' if config.spatial else 'F'}_M{'T' if config.memory else 'F'}"
    radius = f"PPO_R{int(params.radius*10):02d}".replace(".", "-")
    timestamp = time.strftime("%Y%m%d-%H%M")
    experiment_name = f"{radius}_{cell_type}_{config.n_train_envs}_{timestamp}_{config.seed}"
    model_dir = pathlib.Path(config.logdir) / experiment_name

    # Initialise the tensorboard logger and same the config files.
    summary_writer = tensorboard.SummaryWriter(log_dir=model_dir)
    with open(model_dir / "env_params.json", "w") as f:
        f.write(json.dumps(dataclasses.asdict(params), indent=4))
    with open(model_dir / "config.json", "w") as f:
        f.write(json.dumps(config._asdict(), indent=4))
    print(f"Writting logs to: {model_dir}")
    return Checkpointer(model_dir, summary_writer)

# NOTE: Apparently jax needs to run before tensorboard.SummaryWriter.
# otherwise there is some libcudnn error.
rng = jax.random.PRNGKey(config.seed)
ckpt_manager = init_logger(config, env_params)

rppo.train_loop(rng, config, env_params, ckpt_manager)
