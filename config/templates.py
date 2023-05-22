from dataclasses import dataclass
import time

@dataclass
class BaseConfig:
    algorithm: str = "vpg"
    env_name: str = "CartPole-v1"
    policy_lr: float = 1e-4
    hidden_size: int = 256
    batch_size: int = 64 
    device: str = "cuda"
    compile: bool = False
    wandb_log: bool = False
    wandb_project: str = "project"
    wandb_run_name: str = "run_" + str(time.time())
    out_dir: str = "outputs"
    log_interval: int = 1
    eval_episodes: int = 3
    eval_interval: int = 3
    init_from = 'scratch' # 'scratch' or 'resume'

@dataclass
class VPGConfig(BaseConfig):
    num_epochs: int = 100

