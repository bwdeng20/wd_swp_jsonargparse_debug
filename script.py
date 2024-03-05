import wandb
import os
import collections
import random
from typing import Optional, Dict, Any
from jsonargparse import CLI, capture_parser
from pprint import pprint
from lightning.pytorch import seed_everything

Worker = collections.namedtuple("Worker", ("queue", "process"))
WorkerInitData = collections.namedtuple(
    "WorkerInitData", ("num", "sweep_id", "sweep_run_name", "config")
)
WorkerDoneData = collections.namedtuple("WorkerDoneData", ("val_accuracy"))

PROJECT = "ggg"


class FakeResultGenerator:
    def __init__(self, mu=0, sigma=1):
        self.mu = mu
        self.sigma = sigma

    def generate(self):
        return random.normalvariate(self.mu, self.sigma)

    def __repr__(self):
        return f"Fake(mu={self.mu}, sigma={self.sigma})"


def reset_wandb_env():
    exclude = {
        "WANDB_PROJECT",
        "WANDB_ENTITY",
        "WANDB_API_KEY",
    }
    for key in os.environ.keys():
        if key.startswith("WANDB_") and key not in exclude:
            del os.environ[key]


def train(gen: FakeResultGenerator, num, sweep_id, sweep_run_name, config):
    run_name = f'{sweep_run_name}-{num}'
    run = wandb.init(
        project=PROJECT,
        group=sweep_id,
        job_type=sweep_run_name,
        name=run_name,
        config=config,
        reinit=True
    )
    val_accuracy = gen.generate()
    run.log(dict(val_accuracy=val_accuracy))
    run.finish()
    return val_accuracy


def multirun_train_main(gen: FakeResultGenerator = FakeResultGenerator(mu=0, sigma=1),
                        raw_cfg_dict: Optional[Dict[str, Any]] = None):
    num_folds = 3

    sweep_run = wandb.init(project=PROJECT)
    sweep_id = sweep_run.sweep_id or "unknown"
    sweep_url = sweep_run.get_sweep_url()
    project_url = sweep_run.get_project_url()
    sweep_run.config.update(raw_cfg_dict, allow_val_change=True)
    sweep_group_url = f'{project_url}/groups/{sweep_id}'
    sweep_run.notes = sweep_group_url
    sweep_run.save()
    sweep_run_name = sweep_run.name or sweep_run.id or "unknown_2"
    sweep_run_id = sweep_run.id
    sweep_run.finish()
    wandb.sdk.wandb_setup._setup(_reset=True)

    metrics = []
    for num in range(num_folds):
        reset_wandb_env()
        result = train(
            gen,
            sweep_id=sweep_id,
            num=num,
            sweep_run_name=sweep_run_name,
            config=dict(sweep_run.config),
        )
        metrics.append(result)

    # resume the sweep run
    sweep_run = wandb.init(project=PROJECT, id=sweep_run_id, resume="must")
    # log metric to sweep run
    sweep_run.log(dict(mean_val_accuracy=sum(metrics) / len(metrics)))
    sweep_run.finish()

    print("*" * 40)
    print("Sweep URL:       ", sweep_url)
    print("Sweep Group URL: ", sweep_group_url)
    print("*" * 40)


def objective_main():
    def main_cli():
        return CLI(multirun_train_main)

    parser = capture_parser(main_cli)
    cfg = parser.parse_args()
    init = parser.instantiate_classes(cfg)
    init_dict = init.as_dict()

    cfg_dict = cfg.as_dict()
    print(">================= CFG ========================")
    pprint(cfg_dict)
    print("<================= CFG ========================")
    print(">================= INIT ========================")
    pprint(init_dict)
    print("<================= INIT ========================")

    config_fps = cfg_dict["config"]
    cfg_dict["cwd"] = config_fps[0]._cwd  # noqa
    cfg_dict["config"] = [str(fp) for fp in config_fps]

    multirun_train_main(init_dict["gen"], cfg_dict)


if __name__ == "__main__":
    seed_everything(2024)
    objective_main()
