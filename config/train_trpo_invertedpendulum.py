from config.templates import TRPOConfig
import datetime
from os.path import join

# variable must be called 'config'
config = TRPOConfig(
    wandb_log = False,
    wandb_project = "trpo_inverted_pendulum",
    wandb_run_name = "run_" + str(datetime.datetime.now()),
    out_dir = join("outputs", "out_trpo_inverted_pendulum"),
    env_name = "InvertedPendulum-v4",
    policy_hidden_n_layers=1,
    policy_hidden_sizes=(64,64),
    value_hidden_n_layers=0,
    value_hidden_sizes=(64,),
    value_lr=0.01,
    gamma=0.99,
    bias=True,
    compile=False,
    eval_interval=30,
    n_eval_epochs = 3,
    batch_size=4000,
    device='cpu' # cpu faster for small models
)