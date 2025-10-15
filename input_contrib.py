from stable_baselines3 import PPO
from captum.attr import IntegratedGradients
import torch
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
obs_names = [
    # qpos[1:]
    "rootz_pos", "rooty_pos",
    "bthigh_pos", "bshin_pos", "bfoot_pos",
    "fthigh_pos", "fshin_pos", "ffoot_pos",

    # qvel
    "rootx_vel", "rootz_vel", "rooty_vel",
    "bthigh_vel", "bshin_vel", "bfoot_vel",
    "fthigh_vel", "fshin_vel", "ffoot_vel"]#, 'time']

env = gym.make('my_HalfCheetah-v5', render_mode='rgb_array', target_velocity=3)

filename = "/Users/nchaix/Documents/MIT/code/curriculum_pruning/pilot_code/" \
           "log_folder/prun_gradual_except_root_0/seed_0/models/model_6000000.zip"

#filename = '/Users/nchaix/Documents/MIT/code/curriculum_pruning/pilot_code/log_folder/periodic_time_5HZ_1/' \
 #          'seed_0/models/model_1000000.zip'

model = PPO.load(filename)


# Input prep (single state or batch) - must be torch tensor
obs, info = env.reset() # Replace with your own observation
# Make it (1, 17) for batching
#obs = obs.reshape(1, -1)

n_samples = 100  # Adjust for as many as needed
obs_list = []

while len(obs_list) < n_samples:
    obs_list.append(np.array(obs).reshape(1, -1))          # Store observation
    action, _ = model.predict(obs, deterministic=True)     # Policy-suggested action
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()                            # Restart episode if done

# Stack into (n_samples, obs_dim) array
obs_batch = np.concatenate(obs_list, axis=0)

inputs = torch.tensor(obs_batch, dtype=torch.float32, requires_grad=True)
# For Captum, you may want to attribute to a particular action component:
chosen_action_index = 0  # E.g., first dimension of action vector
# Or attribute across the whole action vector
# Define a forward function for the policy
def forward_func(x):
    logits, _, _ = model.policy.forward(x)
    return logits

# Attribution
ig = IntegratedGradients(forward_func)
all_attributions = []
for i in range(len(action)):
    attributions = ig.attribute(inputs, target=i)
    all_attributions.append(attributions.cpu().detach())
    attr_np = attributions.squeeze().cpu().detach().numpy()
    # Convert to numpy for analysis
    #attr_np = attributions.cpu().detach().numpy()  # Shape: [n_samples, obs_dim]
    # Compute per-feature mean and std across the batch
    #attr_mean = attr_np.mean(axis=0)
    #attr_std = attr_np.std(axis=0)

attr_tensor = torch.stack(all_attributions, dim=1)  # shape: [n_samples, n_targets, obs_dim]
attr_np = attr_tensor.numpy()  # Convert to numpy for analysis
attr_mean = attr_np.mean(axis=(0, 1))  # Shape: [obs_dim]
attr_std = attr_np.std(axis=(0, 1))  # Shape: [obs_dim]
plt.figure(figsize=(10, 9))
plt.barh(obs_names, attr_mean, xerr=attr_std, capsize=5, alpha=0.7, color='skyblue')
plt.yticks(rotation=0)
plt.xlabel('Attribution')
plt.title('Integrated Gradients Attributions (Mean ± Std) over Policy Rollouts - Horizontal Orientation')
plt.tight_layout()
plt.grid(axis='x', linestyle='--', alpha=0.4)
plt.show()

print(attr_mean)