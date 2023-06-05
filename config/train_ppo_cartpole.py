from config.templates import PPOConfig
import datetime
from os.path import join

# variable must be called 'config'
config = PPOConfig(
    wandb_log = False,
    wandb_project = "ppo_cartpole",
    wandb_run_name = "run_" + str(datetime.datetime.now()),
    out_dir = join("outputs", "out_ppo_cartpole"),
    env_name = "CartPole-v1",
    policy_hidden_n_layers=1,
    policy_hidden_sizes=(128,64),
    value_hidden_n_layers=1,
    value_hidden_sizes=(128,64),
    value_lr=0.001,
    policy_lr=0.0003,
    bias=True,
    compile=False,
    eval_interval=10,
    n_eval_epochs = 3,
    device='cpu', # cpu faster for small models
    clip_ratio = 0.2
)