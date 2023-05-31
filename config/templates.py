from dataclasses import dataclass
import time

@dataclass
class BaseConfig:
    algorithm: str = "vpg"
    env_name: str = "CartPole-v1"
    policy_lr: float = 1e-3
    policy_hidden_n_layers: int = 1
    policy_hidden_sizes: tuple[int, ...] | list[int] = (64, 32)
    value_hidden_n_layers: int = 1
    value_hidden_sizes: tuple[int, ...] | list[int] = (64, 32)
    batch_size: int = 64 
    n_frame_stack: int = 1
    bias: bool = True
    log_interval: int = 1
    n_eval_epochs: int = 4
    eval_interval: int = 10
    log_sig_min: float = -20
    log_sig_max: float = 2
    epsilon: float = 1e-8
    device: str = "cuda"
    compile: bool = True
    dtype: 'str' = 'float32' # 'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16 (used for autocasting)
    init_from: str = 'scratch' # 'scratch' or 'resume'
    out_dir: str = "outputs"
    wandb_log: bool = False
    wandb_project: str = "project"
    wandb_run_name: str = "run_" + str(time.time())

@dataclass
class VPGConfig(BaseConfig):
    algorithm: str = "vpg"
    num_epochs: int = 100

@dataclass
class TRPOConfig(BaseConfig):
    algorithm: str = "trpo"
    delta: float = 0.01
    gamma: float = 0.995
    backtrack_coeff: float = 0.8
    backtrack_iters: int = 10
    cg_iters: int = 10
    max_kl_divergence: float = 0.01
    damping_coeff: float = 0.1
    value_lr: float = 0.001
    n_epochs: int = 1000
    batch_size: int = 4000
    