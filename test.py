import torch
import gymnasium as gym
from configurator import get_config
from utils import get_model_obj
import sys 
import os

out_dir = os.path.join(sys.argv[1], 'model.pt')
assert os.path.isfile(out_dir) # ensure the checkpoint exists

checkpoint = torch.load(out_dir)

config = checkpoint['config']
env = gym.make(config.env_name, render_mode="human")
model_type = get_model_obj(config.algorithm)
model = model_type(checkpoint['config'], env)

model.to(config.device)
model.load(checkpoint)

model.eval()

# Print the config
for k, v in config.__dict__.items():
    print(f"{k} = {v}")

# Dont save whilst evaluating
model.eval_model(save=False)