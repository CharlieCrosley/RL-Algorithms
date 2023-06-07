from config.templates import TD3Config
import datetime
from os.path import join

# variable must be called 'config'
config = TD3Config(
    wandb_log = False,
    wandb_project = "td3_halfcheetah",
    wandb_run_name = "run_" + str(datetime.datetime.now()),
    out_dir = join("outputs", "out_td3_halfcheetah"),
    env_name = "HalfCheetah-v4",
    policy_hidden_n_layers=1,
    policy_hidden_sizes=(256,256),
    q_hidden_n_layers=1,
    q_hidden_sizes=(256,256),
    policy_lr=3e-4,
    q_lr=3e-4,
    bias=True,
    compile=False,
    eval_interval=7,
    n_eval_epochs = 10,
    device='cuda',
    batch_size=100,
    warmup_steps=10000,
    log_interval=1,
    memory_size=1000000,
    update_steps=1,
    update_every=1, 
    update_after=0, 
    steps_per_epoch=4000,
)