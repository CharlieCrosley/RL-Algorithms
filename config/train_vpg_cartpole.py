from config.templates import VPGConfig
import datetime
from os.path import join

# variable must be called 'config'
config = VPGConfig(
    wandb_log = False,
    wandb_project = "vpg_cartpole",
    wandb_run_name = "run_" + str(datetime.datetime.now()),
    out_dir = join("outputs", "out_vpg_cartpole"),
    env_name = "CartPole-v1",
    batch_size=4000,
    policy_lr=4e-3,
    value_lr=4e-3,
    compile=False,
    n_epochs=100,
    device='cpu'
)