from config.templates import DQNConfig
import datetime
from os.path import join

# variable must be called 'config'
config = DQNConfig(
    wandb_log = False,
    wandb_project = "dqn_cartpole",
    wandb_run_name = "run_" + str(datetime.datetime.now()),
    out_dir = join("outputs", "out_dqn_cartpole"),
    env_name = "CartPole-v1",
    policy_hidden_n_layers=1,
    policy_hidden_sizes=(128,128),
    policy_lr=1e-4,
    bias=True,
    compile=False,
    eval_interval=15,
    n_eval_epochs = 3,
    device='cuda', # cpu faster for small models
    batch_size=128,
    warmup_steps=1000,
    log_interval=5,
    memory_size=10000,
)