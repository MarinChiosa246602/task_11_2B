"""
PID vs RL Controller Comparison
================================
Compare performance of PID and RL controllers on the same target positions.

Metrics:
- Overshoot: Maximum distance past the target
- Steady State Error: Final positioning error
- Response Time: Time to reach within threshold of target
- Trajectory smoothness
"""

import numpy as np
import matplotlib.pyplot as plt
from ot2_gym_wrapper import OT2Env

# Try to import RL model
try:
    from stable_baselines3 import PPO
    RL_AVAILABLE = True
except ImportError:
    RL_AVAILABLE = False
    print("Warning: stable_baselines3 not available, RL comparison disabled")


# =============================================================================
# PID Controller
# =============================================================================

class PIDController:
    """PID Controller for OT-2 positioning."""
    
    def __init__(self, Kp=10.0, Ki=1.0, Kd=2.0):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.integral = np.zeros(3)
        self.previous_error = np.zeros(3)
        self.dt = 1/240  # Simulation timestep
        
    def reset(self):
        self.integral = np.zeros(3)
        self.previous_error = np.zeros(3)
    
    def compute(self, current_pos, target_pos):
        """Compute PID control action."""
        error = target_pos - current_pos
        
        # Proportional
        P = self.Kp * error
        
        # Integral
        self.integral += error * self.dt
        I = self.Ki * self.integral
        
        # Derivative
        derivative = (error - self.previous_error) / self.dt
        D = self.Kd * derivative
        
        self.previous_error = error.copy()
        
        # Compute action and clip to [-1, 1]
        action = P + I + D
        action = np.clip(action, -1.0, 1.0)
        
        return action.astype(np.float32)


# =============================================================================
# RL Controller Wrapper
# =============================================================================

class RLController:
    """Wrapper for trained RL model."""
    
    def __init__(self, model_path="model/best_model"):
        self.model = PPO.load(model_path)
    
    def reset(self):
        pass  # RL model is stateless
    
    def predict(self, obs):
        action, _ = self.model.predict(obs, deterministic=True)
        return action


# =============================================================================
# Evaluation Functions
# =============================================================================

def evaluate_controller(env, controller, controller_type, n_episodes=20, max_steps=100):
    """
    Evaluate a controller and collect performance metrics.
    
    Returns dict with:
    - steady_state_errors: Final distance to target for each episode
    - response_times: Steps to reach within 5mm of target
    - overshoots: Maximum overshoot past target
    - trajectories: Distance over time for each episode
    """
    
    results = {
        'steady_state_errors': [],
        'response_times': [],
        'overshoots': [],
        'trajectories': [],
        'successes': []
    }
    
    for ep in range(n_episodes):
        obs, _ = env.reset()
        controller.reset()
        
        # Get initial positions based on observation format
        if len(obs) == 4:  # 4D: [dir_x, dir_y, dir_z, dist_norm]
            # Need to track positions differently for 4D obs
            initial_distance = obs[3] * 0.3  # Unnormalize
            goal_pos = env.goal_position
            pipette_pos = env.pipette_position
        elif len(obs) == 6:  # 6D: [pipette_xyz, goal_xyz]
            pipette_pos = obs[:3]
            goal_pos = obs[3:6]
            initial_distance = np.linalg.norm(pipette_pos - goal_pos)
        elif len(obs) == 10:  # 10D: [delta, dist, pipette, goal]
            pipette_pos = obs[4:7]
            goal_pos = obs[7:10]
            initial_distance = np.linalg.norm(pipette_pos - goal_pos)
        else:
            initial_distance = 0.15  # Default
            goal_pos = np.zeros(3)
        
        trajectory = [initial_distance]
        min_distance = initial_distance
        response_time = max_steps  # Default if never reached
        reached_threshold = False
        
        for step in range(max_steps):
            # Get action based on controller type
            if controller_type == 'PID':
                # PID needs current and target positions
                if hasattr(env, 'pipette_position') and hasattr(env, 'goal_position'):
                    current_pos = env.pipette_position
                    target_pos = env.goal_position
                else:
                    # Extract from observation
                    if len(obs) == 6:
                        current_pos = obs[:3]
                        target_pos = obs[3:6]
                    elif len(obs) == 10:
                        current_pos = obs[4:7]
                        target_pos = obs[7:10]
                    else:
                        current_pos = np.zeros(3)
                        target_pos = np.zeros(3)
                action = controller.compute(current_pos, target_pos)
            else:  # RL
                action = controller.predict(obs)
            
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Get current distance
            distance = info.get('distance', 0)
            trajectory.append(distance)
            
            # Track minimum distance (for overshoot calculation)
            if distance < min_distance:
                min_distance = distance
            
            # Check if reached threshold (5mm)
            if distance < 0.005 and not reached_threshold:
                response_time = step + 1
                reached_threshold = True
            
            if terminated:
                break
        
        # Calculate metrics
        final_distance = trajectory[-1]
        
        # Overshoot: if we got closer than final position, that's overshoot
        overshoot = max(0, final_distance - min_distance)
        
        results['steady_state_errors'].append(final_distance)
        results['response_times'].append(response_time)
        results['overshoots'].append(overshoot)
        results['trajectories'].append(trajectory)
        results['successes'].append(final_distance < 0.001)
    
    return results


def print_comparison(pid_results, rl_results):
    """Print comparison table."""
    
    print("\n" + "="*70)
    print("CONTROLLER COMPARISON RESULTS")
    print("="*70)
    
    print("\n{:<30} {:>15} {:>15}".format("Metric", "PID", "RL"))
    print("-"*70)
    
    # Steady State Error
    pid_sse = np.mean(pid_results['steady_state_errors']) * 1000
    rl_sse = np.mean(rl_results['steady_state_errors']) * 1000
    pid_sse_std = np.std(pid_results['steady_state_errors']) * 1000
    rl_sse_std = np.std(rl_results['steady_state_errors']) * 1000
    print("{:<30} {:>12.3f} mm {:>12.3f} mm".format(
        "Steady State Error (mean)", pid_sse, rl_sse))
    print("{:<30} {:>12.3f} mm {:>12.3f} mm".format(
        "Steady State Error (std)", pid_sse_std, rl_sse_std))
    
    # Response Time
    pid_rt = np.mean(pid_results['response_times'])
    rl_rt = np.mean(rl_results['response_times'])
    print("{:<30} {:>11.1f} steps {:>10.1f} steps".format(
        "Response Time (to 5mm)", pid_rt, rl_rt))
    
    # Overshoot
    pid_os = np.mean(pid_results['overshoots']) * 1000
    rl_os = np.mean(rl_results['overshoots']) * 1000
    print("{:<30} {:>12.3f} mm {:>12.3f} mm".format(
        "Overshoot (mean)", pid_os, rl_os))
    
    # Success Rate
    pid_sr = np.mean(pid_results['successes']) * 100
    rl_sr = np.mean(rl_results['successes']) * 100
    print("{:<30} {:>13.1f} % {:>13.1f} %".format(
        "Success Rate (<1mm)", pid_sr, rl_sr))
    
    # Min/Max errors
    pid_min = np.min(pid_results['steady_state_errors']) * 1000
    rl_min = np.min(rl_results['steady_state_errors']) * 1000
    pid_max = np.max(pid_results['steady_state_errors']) * 1000
    rl_max = np.max(rl_results['steady_state_errors']) * 1000
    print("{:<30} {:>12.3f} mm {:>12.3f} mm".format(
        "Min Error", pid_min, rl_min))
    print("{:<30} {:>12.3f} mm {:>12.3f} mm".format(
        "Max Error", pid_max, rl_max))
    
    print("-"*70)
    
    # Recommendation
    print("\nRECOMMENDATION:")
    
    scores = {'PID': 0, 'RL': 0}
    
    if pid_sse < rl_sse:
        scores['PID'] += 1
        print("  ✓ PID has lower steady state error")
    else:
        scores['RL'] += 1
        print("  ✓ RL has lower steady state error")
    
    if pid_rt < rl_rt:
        scores['PID'] += 1
        print("  ✓ PID has faster response time")
    else:
        scores['RL'] += 1
        print("  ✓ RL has faster response time")
    
    if pid_os < rl_os:
        scores['PID'] += 1
        print("  ✓ PID has less overshoot")
    else:
        scores['RL'] += 1
        print("  ✓ RL has less overshoot")
    
    if pid_sr > rl_sr:
        scores['PID'] += 1
        print("  ✓ PID has higher success rate")
    else:
        scores['RL'] += 1
        print("  ✓ RL has higher success rate")
    
    winner = "PID" if scores['PID'] > scores['RL'] else "RL"
    print(f"\n  → RECOMMENDED CONTROLLER: {winner} (Score: PID={scores['PID']}, RL={scores['RL']})")
    print("="*70)
    
    return winner


def plot_comparison(pid_results, rl_results, save_path="comparison_plot.png"):
    """Create comparison plots."""
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Trajectory comparison (first 5 episodes)
    ax1 = axes[0, 0]
    for i in range(min(5, len(pid_results['trajectories']))):
        ax1.plot(pid_results['trajectories'][i], 'b-', alpha=0.5, label='PID' if i==0 else '')
        ax1.plot(rl_results['trajectories'][i], 'r-', alpha=0.5, label='RL' if i==0 else '')
    ax1.axhline(y=0.001, color='g', linestyle='--', label='Target (1mm)')
    ax1.set_xlabel('Steps')
    ax1.set_ylabel('Distance to Goal (m)')
    ax1.set_title('Trajectory Comparison')
    ax1.legend()
    ax1.set_yscale('log')
    
    # Plot 2: Steady State Error Distribution
    ax2 = axes[0, 1]
    pid_errors = np.array(pid_results['steady_state_errors']) * 1000
    rl_errors = np.array(rl_results['steady_state_errors']) * 1000
    ax2.boxplot([pid_errors, rl_errors], labels=['PID', 'RL'])
    ax2.axhline(y=1, color='g', linestyle='--', label='Target (1mm)')
    ax2.set_ylabel('Steady State Error (mm)')
    ax2.set_title('Steady State Error Distribution')
    ax2.legend()
    
    # Plot 3: Response Time Comparison
    ax3 = axes[1, 0]
    ax3.bar(['PID', 'RL'], 
            [np.mean(pid_results['response_times']), np.mean(rl_results['response_times'])],
            yerr=[np.std(pid_results['response_times']), np.std(rl_results['response_times'])],
            capsize=5, color=['blue', 'red'], alpha=0.7)
    ax3.set_ylabel('Response Time (steps)')
    ax3.set_title('Response Time to 5mm')
    
    # Plot 4: Success Rate
    ax4 = axes[1, 1]
    pid_sr = np.mean(pid_results['successes']) * 100
    rl_sr = np.mean(rl_results['successes']) * 100
    ax4.bar(['PID', 'RL'], [pid_sr, rl_sr], color=['blue', 'red'], alpha=0.7)
    ax4.set_ylabel('Success Rate (%)')
    ax4.set_title('Success Rate (<1mm)')
    ax4.set_ylim(0, 100)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"\nPlot saved to: {save_path}")
    plt.show()


# =============================================================================
# Main
# =============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Compare PID vs RL controllers')
    parser.add_argument('--model_path', type=str, default='model/best_model',
                        help='Path to trained RL model')
    parser.add_argument('--n_episodes', type=int, default=20,
                        help='Number of episodes to evaluate')
    parser.add_argument('--max_steps', type=int, default=100,
                        help='Max steps per episode')
    parser.add_argument('--no_plot', action='store_true',
                        help='Skip plotting')
    args = parser.parse_args()
    
    print("="*70)
    print("PID vs RL CONTROLLER COMPARISON")
    print("="*70)
    
    # Create environment
    env = OT2Env(render=False, max_steps=args.max_steps)
    
    # Initialize controllers
    pid_controller = PIDController(Kp=10.0, Ki=1.0, Kd=2.0)
    
    print("\n[1/2] Evaluating PID Controller...")
    pid_results = evaluate_controller(env, pid_controller, 'PID', 
                                       n_episodes=args.n_episodes, 
                                       max_steps=args.max_steps)
    
    if RL_AVAILABLE:
        try:
            rl_controller = RLController(args.model_path)
            print("[2/2] Evaluating RL Controller...")
            rl_results = evaluate_controller(env, rl_controller, 'RL',
                                             n_episodes=args.n_episodes,
                                             max_steps=args.max_steps)
        except Exception as e:
            print(f"Error loading RL model: {e}")
            print("Creating dummy RL results for comparison...")
            rl_results = pid_results.copy()  # Placeholder
    else:
        print("RL not available, using dummy results")
        rl_results = pid_results.copy()
    
    env.close()
    
    # Print comparison
    winner = print_comparison(pid_results, rl_results)
    
    # Plot comparison
    if not args.no_plot:
        try:
            plot_comparison(pid_results, rl_results)
        except Exception as e:
            print(f"Could not create plot: {e}")
    
    return winner


if __name__ == "__main__":
    main()
