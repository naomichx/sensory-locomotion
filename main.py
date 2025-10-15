"""
Reinforcement Learning Training Script for Ant and HalfCheetah Agents
======================================================================

This script trains PPO agents for either the Ant or HalfCheetah environment using
optimized hyperparameters from RL Baselines3 Zoo.

IMPORTANT - SETUP REQUIRED:
-------------------------------
This project uses CUSTOM gymnasium environments with modified reward functions.
Before running this script, you MUST:

1. Create and activate a virtual environment
2. Install required packages (gymnasium, stable-baselines3, mujoco, etc.)
3. Copy custom environment files (my_ant_v5.py, my_half_cheetah_v5.py) to:
   venv/lib/python3.9/site-packages/gymnasium/envs/mujoco/
4. Register the custom environments in gymnasium

SEE README.md FOR COMPLETE SETUP INSTRUCTIONS

QUICK START:
------------
1. Follow setup instructions in README.md
2. Change 'agent_name' variable below (line 51) to 'ant' or 'half_cheetah'
3. Run: python main.py

The script will automatically:
    - Load the correct custom environment (my_Ant-v5 or my_HalfCheetah-v5)
    - Apply agent-specific PPO hyperparameters
    - Save outputs to {agent_name}_folder/test/seed_{n}/
"""

import torch.nn
import os
import numpy as np
import gymnasium as gym
import torch
import imageio
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.logger import configure
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
import mujoco
import json
from gymnasium.spaces import Discrete

# ============================================================================
# CONFIGURATION - Switch between agents here
# ============================================================================
# Choose agent: 'ant' or 'half_cheetah'
agent_name = 'ant'

# Key Differences Between Agents:
# ┌─────────────────┬──────────────────────┬──────────────────────────┐
# │   Parameter     │      Ant             │  HalfCheetah             │
# ├─────────────────┼──────────────────────┼──────────────────────────┤
# │ Environment     │ my_Ant-v5 (CUSTOM)   │ my_HalfCheetah-v5 (CUSTOM)│
# │ Learning Rate   │ 1.90609e-05          │ 2.0633e-05               │
# │ Gamma           │ 0.98                 │ 0.98                     │
# │ Clip Range      │ 0.1                  │ 0.1                      │
# │ Entropy Coef    │ 4.9646e-07           │ 0.000401762              │
# │ GAE Lambda      │ 0.8                  │ 0.92                     │
# │ Batch Size      │ 32                   │ (default)                │
# │ Max Grad Norm   │ 0.6                  │ 0.8                      │
# │ N Epochs        │ 10                   │ 20                       │
# │ N Steps         │ 512                  │ 512                      │
# │ VF Coef         │ 0.677239             │ 0.58096                  │
# │ Device          │ auto                 │ cpu                      │
# └─────────────────┴──────────────────────┴──────────────────────────┘
#
# NOTE: Both environments use CUSTOM reward functions with target velocity tracking
# See my_ant_v5.py and my_half_cheetah_v5.py for implementation details

path_output = os.path.join(os.getcwd(), f'{agent_name}_folder')
os.makedirs(path_output, exist_ok=True)
nb_train = 5e5  # Total number of training timesteps
n_seed = 1  # Number of different random seeds to run
target_velocity = 2  # Target velocity for the agent (used for Ant)
parallelize = False  # Whether to use parallel environments for training
existing_model = False  # Whether to load from an existing model
eval_freq = 5e5  # Frequency of evaluation (in timesteps)


class PeriodicEvalCallback(BaseCallback):
    """
    Callback for periodic evaluation during training.
    Saves videos, models, feature extractor outputs, and simulation data at regular intervals.
    """
    def __init__(self, eval_env, eval_freq, data_path=None, model_path=None, video_path=None, fe_path=None,
                 n_eval_episodes=1, verbose=0):
        """
        Args:
            eval_env: Environment to use for evaluation
            eval_freq: Frequency (in timesteps) at which to evaluate
            data_path: Path to save simulation data
            model_path: Path to save model checkpoints
            video_path: Path to save evaluation videos
            fe_path: Path to save feature extractor outputs
            n_eval_episodes: Number of episodes to run during evaluation
            verbose: Verbosity level
        """
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = int(eval_freq)
        self.n_eval_episodes = n_eval_episodes
        self.video_path = video_path
        self.model_path = model_path
        self.data_path = data_path
        self.fe_path = fe_path
        
        # Create directories for saving outputs
        os.makedirs(video_path, exist_ok=True)
        os.makedirs(model_path, exist_ok=True)
        os.makedirs(data_path, exist_ok=True)
        os.makedirs(fe_path, exist_ok=True)


    def _on_step(self):
        """
        Called at each training step. Performs evaluation and saves outputs when eval_freq is reached.
        """
        if self.n_calls % self.eval_freq == 0:
            # Initialize storage for evaluation data
            frames = []  # Video frames
            fe_outputs = []  # Feature extractor outputs
            all_data = []  # MuJoCo simulation data
            
            # Reset evaluation environment
            obs, _ = self.eval_env.reset()

            # Run evaluation episode (1000 steps)
            for _ in range(1000):
                # Get action from model (deterministic policy)
                action, state = model.predict(obs, deterministic=True)
                obs, _, terminated, truncated, _ = env.step(action)

                # Render current frame
                frame = self.eval_env.render()

                # Extract output of MLP feature extractor
                obs_tensor = torch.as_tensor(obs).float().unsqueeze(0)
                features = model.policy.extract_features(obs_tensor)
                fe_outputs.append(features.detach().numpy())

                # Create a copy of the current simulation data (MuJoCo state)
                data_copy = mujoco.MjData(env.unwrapped.model)
                data_copy.qpos[:] = env.unwrapped.data.qpos  # Joint positions
                data_copy.qvel[:] = env.unwrapped.data.qvel  # Joint velocities
                data_copy.act[:] = env.unwrapped.data.act  # Actuator activations
                data_copy.ctrl[:] = env.unwrapped.data.ctrl  # Control inputs
                mujoco.mj_forward(env.unwrapped.model, data_copy)  # Update derived fields
                all_data.append(data_copy)

                # Append frame to video
                if frame is not None:
                    frames.append(frame)
                else:
                    print(f"Warning: Invalid frame received: {type(frame)}")

            # Save feature extractor outputs
            np.save(os.path.join(self.fe_path, f'all_fe_output_{self.n_calls}.npy'), np.array(fe_outputs))
            
            # Save simulation data
            np.save(os.path.join(self.data_path, f'all_data_{self.n_calls}.npy'),  np.array(all_data))

            # Save video
            video_file = os.path.join(self.video_path, f"eval_{self.n_calls}.mp4")
            imageio.mimsave(video_file, frames, fps=30)

            # Save model checkpoint
            model_file = os.path.join(self.model_path, f"model_{self.n_calls}")
            self.model.save(model_file)

        return True


def create_env(parallelize):
    """
    Create custom environment based on agent_name (ant or half_cheetah).
    
    Uses CUSTOM environments with modified reward functions:
    - my_Ant-v5: Custom Ant with target velocity tracking
    - my_HalfCheetah-v5: Custom HalfCheetah with target velocity and time input
    
    Args:
        parallelize: If True, creates vectorized parallel environments for faster training
        
    Returns:
        Gymnasium environment for the selected agent
    """
    # Select custom environment name based on agent
    if agent_name == 'ant':
        env_name = 'my_Ant-v5'
    elif agent_name == 'half_cheetah':
        env_name = 'my_HalfCheetah-v5'  # Custom HalfCheetah environment
    else:
        raise ValueError(f"Unknown agent_name: {agent_name}. Choose 'ant' or 'half_cheetah'")
    
    if parallelize:
        # Use parallel environments for faster training
        import multiprocessing
        multiprocessing.set_start_method("spawn")  # Optional but safer on macOS
        env = make_vec_env(env_name, n_envs=32, vec_env_cls=SubprocVecEnv,
                           env_kwargs={'target_velocity': target_velocity, 'render_mode': 'rgb_array'})
    else:
        # Single environment with target velocity
        if agent_name == 'ant':
            env = gym.make(env_name, render_mode='rgb_array', target_velocity=target_velocity)
            # Optional: wrap with synergy wrapper for dimensionality reduction
            # env = AntSynergyWrapper(env, n_synergies=4)
        else:  # half_cheetah
            env = gym.make(env_name, render_mode='rgb_array', target_velocity=target_velocity)
    
    return env


def create_model(filename, existing_model=False, model_filename=None):
    """
    Create PPO model with agent-specific hyperparameters.
    
    Args:
        filename: Name of the folder to save model outputs
        existing_model: If True, load policy from an existing model
        model_filename: Path to existing model (required if existing_model=True)
        
    Returns:
        PPO model with appropriate hyperparameters for the selected agent
    """
    # Load policy from existing model if specified
    if not existing_model:
        policy_kwargs = None
    else:
        assert model_filename is not None, "You must specify the source filename of the model."
        original_model = RecurrentPPO.load(model_filename)
        policy_kwargs = original_model.policy_kwargs.copy()

    # ========================================================================
    # Create PPO model with agent-specific hyperparameters
    # Hyperparameters from: https://github.com/DLR-RM/rl-baselines3-zoo
    # ========================================================================
    
    if agent_name == 'ant':
        # Optimized hyperparameters for Ant
        model = PPO(
            policy='MlpPolicy',
            env=env,
            verbose=1,
            device='auto',
            learning_rate=1.90609e-05,
            gamma=0.98,
            clip_range=0.1,
            ent_coef=4.9646e-07,
            gae_lambda=0.8,
            batch_size=32,
            max_grad_norm=0.6,
            n_epochs=10,
            n_steps=512,
            vf_coef=0.677239,
            policy_kwargs=policy_kwargs
        )
    elif agent_name == 'half_cheetah':
        # Optimized hyperparameters for HalfCheetah
        model = PPO(
            policy='MlpPolicy',
            env=env,
            verbose=1,
            device='cpu',
            learning_rate=2.0633e-05,
            gamma=0.98,
            clip_range=0.1,
            ent_coef=0.000401762,
            gae_lambda=0.92,
            max_grad_norm=0.8,
            n_epochs=20,
            n_steps=512,
            vf_coef=0.58096,
            policy_kwargs=policy_kwargs
        )
    else:
        raise ValueError(f"Unknown agent_name: {agent_name}. Choose 'ant' or 'half_cheetah'")
    
    # Optional: Use custom policy like DiagonalGaitPolicy for Ant
    # model = PPO(policy=DiagonalGaitPolicy, env=env, verbose=1, device='auto', 
    #             learning_rate=1.90609e-05, gamma=0.98, clip_range=0.1, ent_coef=4.9646e-07,
    #             gae_lambda=0.8, batch_size=32, max_grad_norm=0.6, n_epochs=10, 
    #             n_steps=512, vf_coef=0.677239, policy_kwargs=policy_kwargs)

    # ========================================================================
    # Save hyperparameters to JSON file for reproducibility
    # ========================================================================
    config = {
        "agent_name": agent_name,
        "policy_kwargs": model.policy_kwargs,
        "learning_rate": model.lr_schedule(1),
        "gamma": model.gamma,
        "clip_range": model.clip_range(1) if callable(model.clip_range) else model.clip_range,
        "ent_coef": model.ent_coef,
        "gae_lambda": model.gae_lambda,
        "batch_size": model.batch_size,
        "max_grad_norm": model.max_grad_norm,
        "n_epochs": model.n_epochs,
        "n_steps": model.n_steps,
        "vf_coef": model.vf_coef,
        "normalize_advantage": model.normalize_advantage,
    }

    # Create output directory
    os.makedirs(os.path.join(path_output, filename), exist_ok=True)

    # Save hyperparameters to JSON
    with open(os.path.join(path_output, filename + "/hyperparams.json"), "w") as f:
        json.dump(config, f, indent=4)

    # Configure logger to save training logs
    model.set_logger(configure(os.path.join(path_output, filename)))

    # Load existing model weights if specified
    if existing_model:
        model.policy.load_state_dict(original_model.policy.state_dict())
    
    return model


def run_test(model, env):
    """
    Run a final test episode after training and save outputs.
    
    Args:
        model: Trained PPO model
        env: Environment to test on
    """
    # Initialize video frames storage
    frames_video = []
    
    # Reset environment (handle both vectorized and single environments)
    if parallelize:
        obs = env.reset()
    else:
        obs, info = env.reset()
    frames_video.append(env.render())

    # Storage for feature extractor outputs and simulation data
    fe_output = []
    all_data = []

    # Run test episode for 1000 steps
    for _ in range(1000):
        # Get deterministic action from trained model
        action, state = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = env.step(action)
        frames_video.append(env.render())

        # Extract output of MLP feature extractor
        obs_tensor = torch.as_tensor(obs).float().unsqueeze(0)
        features = model.policy.extract_features(obs_tensor)
        fe_output.append(features.detach().numpy())

        # Create a copy of the current simulation data
        data_copy = mujoco.MjData(env.unwrapped.model)
        data_copy.qpos[:] = env.unwrapped.data.qpos  # Joint positions
        data_copy.qvel[:] = env.unwrapped.data.qvel  # Joint velocities
        data_copy.act[:] = env.unwrapped.data.act  # Actuator activations
        data_copy.ctrl[:] = env.unwrapped.data.ctrl  # Control inputs
        mujoco.mj_forward(env.unwrapped.model, data_copy)  # Update derived fields
        all_data.append(data_copy)

    # Save all outputs
    np.save(os.path.join(path_output, filename, 'all_data.npy'), np.array(all_data))
    np.save(os.path.join(path_output, filename, 'fe_output.npy'), np.array(fe_output))
    
    # Save video
    writer = imageio.get_writer(os.path.join(path_output, filename, 'video.mp4'), fps=30)
    for frame in frames_video:
        writer.append_data(frame)
    writer.close()
    
    # Close environment
    env.close()



if __name__ == '__main__':
    # ========================================================================
    # Main training loop
    # ========================================================================
    
    # Find available folder name (avoid overwriting existing runs)
    i = 0
    while True:
        folder = os.path.join(path_output, f'test')
        if not os.path.exists(folder):
            break
        i += 1

    # Run training for multiple seeds
    for seed in range(n_seed):
        print(f"\n{'='*70}")
        print(f"Training {agent_name.upper()} - Seed {seed}")
        print(f"{'='*70}\n")
        
        # Create directory for this seed
        filename = os.path.join(folder, f'seed_{seed}')
        
        # Create environment
        env = create_env(parallelize=parallelize)
        
        # Create model with agent-specific hyperparameters
        model = create_model(existing_model=existing_model, filename=filename, model_filename=None)

        # Set up periodic evaluation callback
        eval_callback = PeriodicEvalCallback(
            eval_env=env, 
            eval_freq=eval_freq,
            video_path=os.path.join(filename, 'videos'),
            model_path=os.path.join(filename, 'models'),
            fe_path=os.path.join(filename, 'fe_outputs'),
            data_path=os.path.join(filename, 'data')
        )

        # Train the model
        print(f"\nStarting training for {nb_train:.0e} timesteps...")
        model.learn(total_timesteps=nb_train, callback=eval_callback)
        
        # Run final test episode
        print(f"\nTraining complete! Running final test episode...")
        run_test(model, env)
        
        print(f"\nResults saved to: {filename}\n")














