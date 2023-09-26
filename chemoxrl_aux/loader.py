from dataclasses import asdict
import json
import pathlib
from typing import Callable, List, NamedTuple

from flax.training import checkpoints
import pick
from chemoxrl import EnvParams, ExperimentConfig
from chemoxrl_aux.policies import greedy_policy, rppo_policy


class CkptModel(NamedTuple):
    name: str
    label: str
    env_params: EnvParams
    hidden_size: int
    policy: Callable


def make_pickable(model: CkptModel):
    env_params = asdict(model.env_params)
    _model = {"name": model.name, "label": model.label, "env_params": env_params}
    return _model


MODEL_LABELS = {
    (True, True, True): "Combined",
    (True, False, False): "Spatial",
    (True, False, True): "Spatial",
    (False, True, True): "Temporal",
    (False, True, False): "Temporal(Markovian)",
    (True, True, False): "Combined (Markovian)",
}


def load_ckpt(ckpt, deterministic=True, max_steps=0) -> CkptModel:
    path = pathlib.Path(ckpt)

    with open(path / "env_params.json", "r") as f:
        args = json.load(f)

    with open(path / "config.json", "r") as f:
        args.update(json.load(f))

    env_args = {k: v for k, v in args.items() if k in EnvParams.__dict__}
    if max_steps > 0:
        env_args["max_steps_in_episode"] = max_steps
    env_params = EnvParams(**env_args)
    ec_args = {k: v for k, v in args.items() if k in ExperimentConfig.__dict__}
    config = ExperimentConfig(**ec_args)
    params = checkpoints.restore_checkpoint(path, target=None)["params"]
    name = path.name
    label = MODEL_LABELS[(config.spatial, config.memory, config.recurrent)]
    policy = rppo_policy(params, deterministic, config.spatial, config.memory, config.recurrent)
    return CkptModel(name, label, env_params, config.hidden_cells, policy)


def load_options(checkpoints_dir, pattern="*"):
    root = pathlib.Path(checkpoints_dir)
    directories = list(root.glob(pattern))
    directories = sorted(directories, key=lambda f: f.stat().st_ctime, reverse=True)
    labels = [f"{o.stem} ({len(list(o.glob('checkpoint_*')))} checkpoints)" for o in directories]
    return directories, labels


def load_multi_interactive(base_dir, pattern="*", select_all=False, **kwargs) -> List[CkptModel]:
    directories, labels = load_options(base_dir, pattern)
    if select_all:
        selected = directories
    else:
        selected = pick.pick(labels, title="Choose checkpoint(s) to load", multiselect=True)
        selected = [directories[s[1]] for s in selected]
    results = [load_ckpt(s, **kwargs) for s in selected]
    return results


def load_multi(base_dir, pattern="*", **kwargs) -> List[CkptModel]:
    selected, labels = load_options(base_dir, pattern)
    results = [load_ckpt(s, **kwargs) for s in selected]
    return results


def load_greedy(env_params, step, epsilon, kernel, adapt):
    name = f"greedy_{kernel}_{epsilon:.4f}_{step:04d}" + "_adapt" if adapt else ""
    label = rf"{kernel}{'(adaptive)' if adapt else ''} T={step:1d} eps={epsilon:.2f}"
    policy, hidden_size = greedy_policy(
        env_params, steps=step, eps=epsilon, kernel=kernel, adapt=adapt
    )
    model = CkptModel(name, label, env_params, hidden_size, policy)
    return model
