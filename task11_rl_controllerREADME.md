# Task 11: Reinforcement Learning Controller for OT-2 Robot

## Overview

This project implements a Reinforcement Learning (RL) controller for precise positioning of the OT-2 pipette robot using the Proximal Policy Optimization (PPO) algorithm. The RL controller is compared against a PID controller to determine the best approach for the final integration.

## Table of Contents

1. [Implementation Steps](#implementation-steps)
2. [Design Choices](#design-choices)
3. [Libraries Used](#libraries-used)
4. [Tuning Strategies](#tuning-strategies)
5. [Performance Metrics](#performance-metrics)
6. [PID vs RL Comparison](#pid-vs-rl-comparison)
7. [Best Hyperparameters](#best-hyperparameters)
8. [Model Weights](#model-weights)
9. [File Structure](#file-structure)
10. [Usage Instructions](#usage-instructions)

---

## Implementation Steps

### Step 1: Environment Wrapper Development

Created a Gymnasium-compatible wrapper (`ot2_gym_wrapper.py`) for the OT-2 simulation that:

- Defines the observation space (4D normalized direction vector)
- Defines the action space (3D velocity commands)
- Implements reward shaping for efficient learning
- Handles episode termination and truncation

### Step 2: Observation Space Design

After multiple iterations, the final observation space uses a **4-dimensional normalized direction vector**:

```
Observation = [direction_x, direction_y, direction_z, distance_normalized]
```

Where:
- `direction_x, direction_y, direction_z`: Unit vector pointing from pipette to goal (range: [-1, 1])
- `distance_normalized`: Distance to goal normalized by workspace size (range: [0, 1])

This design makes learning intuitive: **optimal action = observation[:3]**

### Step 3: Reward Function Design

The reward function combines multiple components:

```python
# 1. Alignment reward - action should match direction
alignment = np.dot(action, direction)
reward = alignment * 100

# 2. Progress reward - getting closer to goal
progress = previous_distance - current_distance
reward += progress * 500

# 3. Success bonuses
if distance < 0.001:  # < 1mm
    reward += 1000
elif distance < 0.005:  # < 5mm
    reward += 50
elif distance < 0.01:   # < 10mm
    reward += 10

# 4. Time penalty
reward -= 1.0
```

### Step 4: Training with ClearML

Training was performed remotely using ClearML with:
- Docker container: `deanis/2023y2b-rl:latest`
- GPU acceleration on remote workers
- Real-time metrics logging to ClearML dashboard

### Step 5: Evaluation and Comparison

Created comparison scripts to evaluate both PID and RL controllers on identical target positions using standardized metrics.

---

## Design Choices

### Why PPO Algorithm?

PPO was chosen for several reasons:
1. **Stability**: PPO's clipped objective prevents destructive policy updates
2. **Sample Efficiency**: Better than vanilla policy gradient methods
3. **Continuous Actions**: Native support for continuous action spaces
4. **Proven Track Record**: Widely used in robotics applications

### Why 4D Observation Space?

Earlier attempts with larger observation spaces (6D, 10D) failed because:
- Raw position data made it difficult for the network to learn the relationship between state and optimal action
- The model learned fixed patterns (e.g., always output [1,1,1]) instead of goal-directed behavior

The 4D normalized direction observation solved this by:
- Making the optimal policy trivial: `action = observation[:3]`
- Providing clear gradient signals for learning
- Reducing the hypothesis space for the neural network

### Why Alignment-Based Reward?

The alignment reward `np.dot(action, direction)` directly rewards the agent for outputting actions that match the direction to the goal. This creates a clear learning signal that converges faster than distance-only rewards.

---

## Libraries Used

| Library | Version | Purpose |
|---------|---------|---------|
| `stable-baselines3` | 2.x | PPO implementation and training |
| `gymnasium` | 0.29+ | Environment interface standard |
| `numpy` | 1.26.4 | Numerical computations |
| `pybullet` | 3.x | Physics simulation |
| `clearml` | 1.x | Remote training and experiment tracking |
| `matplotlib` | 3.x | Visualization and plotting |

### Installation

```bash
pip install stable-baselines3 gymnasium numpy pybullet clearml matplotlib
```

---

## Tuning Strategies

### Hyperparameter Tuning Process

1. **Initial Configuration**: Started with SB3 defaults
2. **Learning Rate**: Tested range [0.0001, 0.001], settled on 0.0003
3. **Batch Size**: Increased from 64 to 128 for more stable gradients
4. **Network Architecture**: Tested [64,64], [128,128], [256,256] - simpler worked better
5. **Episode Length**: Reduced from 1000 to 100 steps for faster iteration

### Key Tuning Insights

| Parameter | Initial | Final | Reason for Change |
|-----------|---------|-------|-------------------|
| `max_steps` | 1000 | 100 | Faster episode turnover, more learning iterations |
| `learning_rate` | 0.0003 | 0.0003 | Default worked well |
| `batch_size` | 64 | 128 | More stable gradient estimates |
| `n_epochs` | 10 | 10 | Sufficient for PPO updates |
| `ent_coef` | 0.01 | 0.01 | Adequate exploration |
| `net_arch` | [64,64] | [64,64] | Simple task, small network sufficient |

### Reward Shaping Evolution

| Version | Reward Design | Result |
|---------|---------------|--------|
| v1 | `-distance` only | No learning (flat reward) |
| v2 | `+progress * 100` | Slow improvement |
| v3 | `+alignment * 50 + progress * 500` | Good learning |
| v4 | `+alignment * 100 + progress * 500 + bonuses` | Best results |

---

## Performance Metrics

### RL Controller Results

| Metric | Value |
|--------|-------|
| Mean Steady State Error | 4.241 mm |
| Std Steady State Error | 5.979 mm |
| Min Error Achieved | 0.150 mm |
| Max Error | 15.851 mm |
| Success Rate (<1mm) | 70.0% |
| Mean Response Time | 27.2 steps |
| Mean Overshoot | 3.424 mm |

### Error Analysis

The RL controller achieves sub-millimeter accuracy (0.150 mm minimum) demonstrating it CAN reach the target precisely. The 70% success rate indicates room for improvement through:
- Longer training
- Further reward tuning
- Curriculum learning

### Training Progress

```
Episode  1000 | Avg Reward: 150.2 | Avg Length: 85.3
Episode  5000 | Avg Reward: 450.8 | Avg Length: 42.1
Episode 10000 | Avg Reward: 580.5 | Avg Length: 28.4
```

---

## PID vs RL Comparison

### Head-to-Head Results

| Metric | PID | RL | Winner |
|--------|-----|-----|--------|
| Steady State Error (mean) | 24.309 mm | **4.241 mm** | RL |
| Steady State Error (std) | 7.649 mm | 5.979 mm | RL |
| Response Time | 100.0 steps | **27.2 steps** | RL |
| Overshoot | **0.021 mm** | 3.424 mm | PID |
| Success Rate (<1mm) | 0.0% | **70.0%** | RL |
| Min Error | 11.671 mm | **0.150 mm** | RL |
| Max Error | 35.849 mm | **15.851 mm** | RL |

### Analysis

**RL Advantages:**
- 5.7x lower mean positioning error (4.2mm vs 24.3mm)
- 3.7x faster response time (27 vs 100 steps)
- 70% success rate vs 0% for PID
- Can achieve sub-millimeter accuracy (0.15mm)

**PID Advantages:**
- Near-zero overshoot (0.021mm)
- More predictable behavior
- No training required

### Recommendation

**RL is recommended** for the final integration based on:
1. Significantly better accuracy (4.2mm vs 24.3mm)
2. Much faster response time
3. Ability to achieve <1mm accuracy target

---

## Best Hyperparameters

### PPO Configuration

```python
# Training hyperparameters
total_timesteps = 500000
learning_rate = 0.0003
batch_size = 128
n_steps = 2048
n_epochs = 10
gamma = 0.99
gae_lambda = 0.95
clip_range = 0.2
ent_coef = 0.01

# Environment settings
max_steps = 100
target_threshold = 0.001  # 1mm

# Network architecture
policy_kwargs = dict(
    net_arch=dict(pi=[64, 64], vf=[64, 64])
)
```

### Reward Function Parameters

```python
alignment_weight = 100
progress_weight = 500
success_bonus_1mm = 1000
success_bonus_5mm = 50
success_bonus_10mm = 10
time_penalty = 1.0
```

---

## Model Weights

### Saved Model Location

The trained model is saved in multiple locations:

1. **Local**: `model/best_model.zip`
2. **ClearML Artifacts**: Available in the ClearML experiment dashboard

### Loading the Model

```python
from stable_baselines3 import PPO

# Load the trained model
model = PPO.load("model/best_model")

# Use for inference
action, _ = model.predict(observation, deterministic=True)
```

### Model Architecture

```
Policy Network (Actor):
  Input: 4 (observation dimension)
  Hidden: 64 → 64 (ReLU activation)
  Output: 3 (action dimension)

Value Network (Critic):
  Input: 4 (observation dimension)
  Hidden: 64 → 64 (ReLU activation)
  Output: 1 (state value)
```

---

## File Structure

```
task_11_2B/
├── model/
│   └── best_model.zip          # Trained PPO model
├── ot2_gym_wrapper.py          # Gymnasium environment wrapper
├── train_rl.py                 # Training script with ClearML
├── test_wrapper.py             # Model evaluation script
├── test_env.py                 # Environment verification script
├── compare_controllers.py      # PID vs RL comparison script
├── sim_class.py                # OT-2 simulation class
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

---

## Usage Instructions

### 1. Verify Environment

```bash
python test_env.py
```

Expected output:
```
✓ SUCCESS! Environment works correctly.
  action = obs[:3] reaches goal < 1mm
```

### 2. Train the Model

```bash
python train_rl.py
```

This will:
- Initialize ClearML task
- Train PPO for 500,000 timesteps
- Save model to `trained_model.zip`
- Upload artifacts to ClearML

### 3. Evaluate the Model

```bash
python test_wrapper.py --model_path model/best_model --n_episodes 100
```

### 4. Compare PID vs RL

```bash
python compare_controllers.py --model_path model/best_model --n_episodes 20
```

### 5. Use in Final Integration

```python
from stable_baselines3 import PPO
from ot2_gym_wrapper import OT2Env

# Load model and environment
model = PPO.load("model/best_model")
env = OT2Env(render=True)

# Run control loop
obs, _ = env.reset()
while True:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    
    if terminated or truncated:
        break

print(f"Final distance: {info['distance']*1000:.3f} mm")
```

---

## References

- [Stable Baselines3 Documentation](https://stable-baselines3.readthedocs.io/)
- [Gymnasium Documentation](https://gymnasium.farama.org/)
- [PPO Paper (Schulman et al., 2017)](https://arxiv.org/abs/1707.06347)
- [ClearML Documentation](https://clear.ml/docs/)

---

## Author

Task 11 - Reinforcement Learning Controller  
2B AI & Data Science

## License

This project is part of the academic curriculum and is for educational purposes.
