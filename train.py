import torch
import gymnasium as gym
from gymnasium.wrappers import FrameStack
from utils import get_model_obj
import os
from configurator import get_config


config = get_config()

env = gym.make(config.env_name, render_mode="human")
#env = FrameStack(env, num_stack=n_frame_stack)

seed = 0
os.makedirs(config.out_dir, exist_ok=True)
torch.manual_seed(seed)
env.action_space.seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn

# Set the chosen model
model_type = get_model_obj(config.algorithm)
model = model_type(config, env)
model.to(config.device)

if config.init_from == 'scratch':
    print("Initialized a new model from scratch")
    
elif config.init_from == 'resume':
    print(f"Resuming training from {config.out_dir}")
    # resume training from a checkpoint.
    path = os.path.join(config.out_dir, 'model.pt')
    checkpoint = torch.load(path, map_location=config.device)
    model.load(checkpoint)
    checkpoint = None # free up memory

# compile the model
if config.compile:
    print("compiling the model... (takes a ~minute)")
    model = torch.compile(model) # requires PyTorch 2.0

# logging
if config.wandb_log:
    import wandb
    wandb.init(project=config.wandb_project, name=config.wandb_run_name, config=config)

model.train_model()

print("Finished Training!")