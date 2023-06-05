from dataclasses import dataclass
import time

@dataclass
class BaseConfig:
    algorithm: str = "vpg"
    env_name: str = "CartPole-v1"
    policy_lr: float = 1e-3
    policy_hidden_n_layers: int = 1
    policy_hidden_sizes: tuple[int, ...] | list[int] = (64, 32)
    frame_stack: int = 1 # for atari envs to show momentum
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
    max_steps = 1e6
    update_steps: int = 80
    max_ep_len: int = 1000
    batch_size: int = 100

@dataclass
class VPGConfig(BaseConfig):
    algorithm: str = "vpg"
    gamma: float = 0.99
    steps_per_epoch: int = 4000
    value_hidden_n_layers: int = 1
    value_hidden_sizes: tuple[int, ...] | list[int] = (64, 32) 
    value_lr: float = 1e-3

@dataclass
class TRPOConfig(VPGConfig):
    algorithm: str = "trpo"
    backtrack_coeff: float = 0.8
    backtrack_iters: int = 10
    cg_iters: int = 10
    max_kl_divergence: float = 0.01
    damping_coeff: float = 0.1
    steps_per_epoch: int = 4000
    
@dataclass
class PPOConfig(VPGConfig):
    algorithm: str = "ppo"
    clip_ratio: float = 0.2
    max_kl_divergence: float = 0.01
    lambda_entropy: float = 0.97
    steps_per_epoch: int = 4000
    
@dataclass
class DQNConfig(BaseConfig):
    algorithm: str = "dqn"
    gamma: float = 0.99
    tau: float = 0.005
    batch_size: int = 32
    memory_size: int = 100000
    warmup_steps: int = 1000
    eps_start: float = 0.9
    eps_end: float = 0.05
    eps_decay: float = 1000

@dataclass
class DDPGConfig(DQNConfig):
    algorithm: str = "ddpg"
    q_lr: float = 0.001
    update_every: int = 50
    update_after: int = 1000
    steps_per_epoch: int = 4000
    q_hidden_n_layers: int = 1
    q_hidden_sizes: tuple[int, ...] | list[int] = (64, 32) 
    gaussian_noise_std: float = 0.1