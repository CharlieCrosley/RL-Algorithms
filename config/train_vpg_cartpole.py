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
    batch_size=256,
    policy_lr=3e-3,
    compile=False,
    num_epochs=7
)