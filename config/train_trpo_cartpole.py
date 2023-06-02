from config.templates import TRPOConfig
import datetime
from os.path import join

# variable must be called 'config'
config = TRPOConfig(
    wandb_log = False,
    wandb_project = "trpo_cartpole",
    wandb_run_name = "run_" + str(datetime.datetime.now()),
    out_dir = join("outputs", "out_trpo_cartpole"),
    env_name = "CartPole-v1",
    policy_hidden_n_layers=1,
    policy_hidden_sizes=(64,32),
    value_hidden_n_layers=1,
    value_hidden_sizes=(64,32),
    value_lr=0.01,
    bias=True,
    compile=False,
    eval_interval=10,
    n_eval_epochs = 3,
    device='cpu', # cpu faster for small models
    batch_size=4000,
    update_steps=80,
    backtrack_coeff=0.8,
    max_kl_divergence=0.3, # increasing will learn much faster but can be unstable
)