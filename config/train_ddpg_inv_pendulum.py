from config.templates import DDPGConfig
import datetime
from os.path import join

# variable must be called 'config'
config = DDPGConfig(
    wandb_log = False,
    wandb_project = "ddpg_inv_pendulum",
    wandb_run_name = "run_" + str(datetime.datetime.now()),
    out_dir = join("outputs", "out_ddpg_inv_pendulum"),
    env_name = "InvertedPendulum-v4",
    policy_hidden_n_layers=1,
    policy_hidden_sizes=(256,256),
    q_hidden_n_layers=1,
    q_hidden_sizes=(400,400),
    policy_lr=0.001,
    q_lr=0.001,
    bias=True,
    compile=False,
    eval_interval=7,
    n_eval_epochs = 1,
    device='cuda',
    batch_size=100,
    warmup_steps=10000,
    log_interval=1,
    memory_size=1000000,
    update_steps=40,
    update_every=50, 
    update_after=1000, 
    steps_per_epoch=4000,
    gaussian_noise_std=0.1,
)