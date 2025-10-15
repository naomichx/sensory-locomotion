# Reinforcement Learning Training - Ant and HalfCheetah

This project trains PPO (Proximal Policy Optimization) agents for Ant and HalfCheetah. 

## Overview

This training script uses **custom gymnasium environments** with modified reward functions that encourage agents to maintain a specific target velocity rather than simply maximizing speed. The custom environments include:

- **my_Ant-v5**: Modified Ant quadruped with target velocity tracking
- **my_HalfCheetah-v5**: Modified HalfCheetah with target velocity tracking and time-based observations

Both environments use an exponential reward function:
```
reward = exp(-(velocity_error)² / target_velocity) - control_cost
```

This encourages the agent to match a specific target velocity rather than running as fast as possible.

---

## Setup Instructions

### 1. Create and Activate Virtual Environment

```bash
cd /path/to/curriculum_pruning/undergrad_project
python3 -m venv venv
source venv/bin/activate  # On macOS/Linux
# OR on Windows: venv\Scripts\activate
```

### 2. Install Required Packages

```bash
pip install gymnasium
pip install stable-baselines3
pip install sb3-contrib
pip install mujoco
pip install imageio
pip install imageio-ffmpeg
pip install torch
pip install numpy
```

### 3. Install Custom Environments

The custom environments modify the standard MuJoCo environments. You need to add them to your gymnasium installation:

#### Step 3.1: Locate Your Gymnasium Installation

```bash
python -c "import gymnasium; import os; print(os.path.dirname(gymnasium.__file__))"
```

This will show something like: `/path/to/venv/lib/python3.9/site-packages/gymnasium`

#### Step 3.2: Copy Custom Environment Files

Navigate to the gymnasium mujoco environments directory and copy the custom files:

```bash
# Navigate to gymnasium's mujoco directory
cd venv/lib/python3.9/site-packages/gymnasium/envs/mujoco/

# Copy the custom environment files from this project
cp /path/to/curriculum_pruning/undergrad_project/my_ant_v5.py .
cp /path/to/curriculum_pruning/undergrad_project/my_half_cheetah_v5.py .
```

**Important:** Replace `/path/to/curriculum_pruning` with your actual project path.

#### Step 3.3: Register Custom Environments

**a) Edit the mujoco __init__.py file:**

File location: `venv/lib/python3.9/site-packages/gymnasium/envs/mujoco/__init__.py`

Add these imports at the top (after other imports):

```python
from gymnasium.envs.mujoco.my_ant_v5 import AntEnv as MyAntEnv
from gymnasium.envs.mujoco.my_half_cheetah_v5 import HalfCheetahEnv as MyHalfCheetahEnv
```

Add to the `__all__` list:

```python
__all__ = [
    "AntEnv",
    "HalfCheetahEnv",
    # ... other environments ...
    "MyAntEnv",
    "MyHalfCheetahEnv",
]
```

**b) Register in gymnasium's main registry:**

File location: `venv/lib/python3.9/site-packages/gymnasium/envs/__init__.py`

Add these registrations in the MuJoCo section (search for `# ============= mujoco =============`):

```python
# Custom Ant environment with target velocity
register(
    id="my_Ant-v5",
    entry_point="gymnasium.envs.mujoco.my_ant_v5:AntEnv",
    max_episode_steps=1000,
    reward_threshold=6000.0,
)

# Custom HalfCheetah environment with target velocity
register(
    id="my_HalfCheetah-v5",
    entry_point="gymnasium.envs.mujoco.my_half_cheetah_v5:HalfCheetahEnv",
    max_episode_steps=1000,
    reward_threshold=4800.0,
)
```

### 4. Verify Installation

Test that the custom environments are properly installed:

```python
import gymnasium as gym

# Test custom Ant environment
env = gym.make('my_Ant-v5', target_velocity=2, render_mode='rgb_array')
print("✓ Custom Ant environment loaded successfully!")
env.close()

# Test custom HalfCheetah environment
env = gym.make('my_HalfCheetah-v5', target_velocity=3, render_mode='rgb_array')
print("✓ Custom HalfCheetah environment loaded successfully!")
env.close()
```

---

## Usage

### Switching Between Agents

To switch between Ant and HalfCheetah, simply edit `main.py` and change the `agent_name` variable (line 51):

```python
# Choose agent: 'ant' or 'half_cheetah'
agent_name = 'ant'  # Change to 'half_cheetah' to train HalfCheetah
```

The script will automatically:
1. Load the correct environment
2. Apply agent-specific PPO hyperparameters
3. Create appropriate output folders

### Running Training

```bash
python main.py
```

### Configuration Parameters

Edit these parameters in `main.py` (lines 74-79):

```python
nb_train = 5e6          # Total training timesteps
n_seed = 1              # Number of random seeds to run
target_velocity = 2     # Target velocity for the agent
parallelize = False     # Use parallel environments (faster but more memory)
existing_model = False  # Load from existing model
eval_freq = 5e5         # Evaluation frequency (timesteps)
```

---

## Custom Environment Details

### my_Ant-v5 Modifications

**Location:** `undergrad_project/my_ant_v5.py`

**Key Differences from Standard Ant-v5:**
- **Reward Function**: Uses exponential reward based on velocity error instead of linear forward reward
  ```python
  velocity_error = abs(target_velocity - x_velocity)
  forward_reward = exp(-(velocity_error)² / target_velocity)
  ```
- **Target Velocity**: Must be specified when creating environment
- **Observations**: Standard Ant observations (105-dimensional by default)

### my_HalfCheetah-v5 Modifications

**Location:** `undergrad_project/my_half_cheetah_v5.py`

**Key Differences from Standard HalfCheetah-v5:**
- **Reward Function**: Same exponential velocity tracking as Ant
- **Target Velocity**: Must be specified when creating environment
- **Time Input**: Adds normalized time (mod 0.2) to observations for periodic behavior
- **Termination Condition**: Episode terminates if torso height is out of safe range
  ```python
  if y_pos < 0.4 or y_pos > 0.9 or y_front < 0.4 or y_front > 0.8:
      terminated = True
  ```

---

## Output Structure

Training outputs are saved to `{agent_name}_folder/test/seed_{n}/`:

```
ant_folder/test/seed_0/
├── hyperparams.json       # Model hyperparameters
├── progress.csv           # Training progress logs
├── videos/                # Evaluation videos
│   ├── eval_500000.mp4
│   ├── eval_1000000.mp4
│   └── ...
├── models/                # Model checkpoints
│   ├── model_500000.zip
│   ├── model_1000000.zip
│   └── ...
├── fe_outputs/            # Feature extractor outputs
│   ├── all_fe_output_500000.npy
│   └── ...
├── data/                  # MuJoCo simulation data
│   ├── all_data_500000.npy
│   └── ...
├── all_data.npy          # Final episode data
├── fe_output.npy         # Final feature extractor output
└── video.mp4             # Final evaluation video
```

---

## Agent-Specific Hyperparameters

The script automatically applies optimized hyperparameters based on the selected agent:

| Parameter       | Ant            | HalfCheetah    |
|-----------------|----------------|----------------|
| Environment     | my_Ant-v5      | my_HalfCheetah-v5 |
| Learning Rate   | 1.90609e-05    | 2.0633e-05     |
| Gamma           | 0.98           | 0.98           |
| Clip Range      | 0.1            | 0.1            |
| Entropy Coef    | 4.9646e-07     | 0.000401762    |
| GAE Lambda      | 0.8            | 0.92           |
| Batch Size      | 32             | (default)      |
| Max Grad Norm   | 0.6            | 0.8            |
| N Epochs        | 10             | 20             |
| N Steps         | 512            | 512            |
| VF Coef         | 0.677239       | 0.58096        |
| Device          | auto           | cpu            |

---

## Troubleshooting

### Import Error: "No module named 'gymnasium.envs.mujoco.my_ant_v5'"

- Ensure you copied the custom environment files to the correct directory
- Check that you edited the `__init__.py` files correctly
- Restart your Python interpreter/kernel after making changes

### Environment Registration Error

- Make sure you added the `register()` calls to `gymnasium/envs/__init__.py`
- Check for typos in the entry_point paths

### MuJoCo Rendering Issues

- Install required video codecs: `pip install imageio-ffmpeg`
- On headless servers, you may need to set: `export MUJOCO_GL=osmesa`

### Custom Environments Not Loading

Run this diagnostic:

```python
import gymnasium as gym
print(gym.envs.registry.keys())  # Should show 'my_Ant-v5' and 'my_HalfCheetah-v5'
```

---

## Requirements

- Python 3.9+
- gymnasium
- stable-baselines3
- sb3-contrib
- mujoco (>=2.3.3)
- torch
- numpy
- imageio
- imageio-ffmpeg

---

## Citation

If you use this code, please cite the original RL Baselines3 Zoo:

```bibtex
@misc{rl-baselines3-zoo,
  author = {Raffin, Antonin},
  title = {RL Baselines3 Zoo},
  year = {2020},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/DLR-RM/rl-baselines3-zoo}},
}
```

---

## License

See LICENSE file for details.

