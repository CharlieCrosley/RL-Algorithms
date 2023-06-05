import torch
import gymnasium as gym
from gymnasium.wrappers import NormalizeObservation
from models.shared.utils import get_model_obj
import sys 
import os
import numpy as np

save_ext = ['pt', 'tar']
file_exists = [os.path.isfile(os.path.join(sys.argv[1], f'model.{ext}')) for ext in save_ext]
assert np.any(file_exists) # ensure the checkpoint exists

ext_idx = np.argwhere(np.array(file_exists) == True)[0][0]
out_dir = os.path.join(sys.argv[1], f'model.{save_ext[ext_idx]}')
checkpoint = torch.load(out_dir)

config = checkpoint['config']
config.n_eval_epochs = 1
env = gym.make(config.env_name, render_mode="human")
env = NormalizeObservation(env)
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

print("Finished Testing!")