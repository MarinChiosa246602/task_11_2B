
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

# Try to import optional dependencies
try:
    from stable_baselines3 import PPO
    RL_AVAILABLE = True
except ImportError:
    RL_AVAILABLE = False
    print("Warning: stable_baselines3 not available")

from ot2_gym_wrapper import OT2Env


# =============================================================================
# Controllers
# =============================================================================

class PIDController:
    """PID Controller for comparison."""
    
    def __init__(self, Kp=10.0, Ki=1.0, Kd=2.0):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.dt = 1/240
        self.integral = np.zeros(3)
        self.previous_error = np.zeros(3)
        
    def reset(self):
        self.integral = np.zeros(3)
        self.previous_error = np.zeros(3)
    
    def compute(self, current_pos, target_pos):
        error = target_pos - current_pos
        P = self.Kp * error
        self.integral += error * self.dt
        self.integral = np.clip(self.integral, -1.0, 1.0)
        I = self.Ki * self.integral
        derivative = (error - self.previous_error) / self.dt
        D = self.Kd * derivative
        self.previous_error = error.copy()
        action = np.clip(P + I + D, -1.0, 1.0)
        return action.astype(np.float32)


class RLController:
    """RL Controller wrapper."""
    
    def __init__(self, model_path):
        self.model = PPO.load(model_path)
    
    def reset(self):
        pass
    
    def predict(self, obs):
        action, _ = self.model.predict(obs, deterministic=True)
        return action


# =============================================================================
# Evaluation
# =============================================================================

def evaluate_controller(env, controller, controller_type, n_episodes=20, max_steps=100):
    """Evaluate a controller and collect detailed metrics."""
    
    results = {
        'steady_state_errors': [],
        'response_times': [],
        'overshoots': [],
        'trajectories': [],
        'successes': [],
        'settling_times': []
    }
    
    for ep in range(n_episodes):
        obs, _ = env.reset()
        controller.reset()
        
        trajectory = []
        min_distance = float('inf')
        response_time = max_steps
        settling_time = max_steps
        reached_5mm = False
        settled = False
        
        for step in range(max_steps):
            # Get action
            if controller_type == 'PID':
                if hasattr(env, 'pipette_position') and hasattr(env, 'goal_position'):
                    action = controller.compute(env.pipette_position, env.goal_position)
                else:
                    action = np.zeros(3, dtype=np.float32)
            else:
                action = controller.predict(obs)
            
            obs, reward, terminated, truncated, info = env.step(action)
            distance = info.get('distance', 0)
            trajectory.append(distance)
            
            if distance < min_distance:
                min_distance = distance
            
            if distance < 0.005 and not reached_5mm:
                response_time = step + 1
                reached_5mm = True
            
            if distance < 0.002 and not settled:
                settling_time = step + 1
                settled = True
            
            if terminated:
                break
        
        final_distance = trajectory[-1] if trajectory else 0
        overshoot = max(0, final_distance - min_distance)
        
        results['steady_state_errors'].append(final_distance)
        results['response_times'].append(response_time)
        results['overshoots'].append(overshoot)
        results['trajectories'].append(trajectory)
        results['successes'].append(final_distance < 0.001)
        results['settling_times'].append(settling_time)
    
    return results


# =============================================================================
# Visualization Functions
# =============================================================================

def create_comparison_plots(pid_results, rl_results, output_dir='.'):
    """Create all comparison visualizations."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Set style
    plt.style.use('seaborn-v0_8-whitegrid')
    colors = {'PID': '#3498db', 'RL': '#e74c3c'}
    
    # ==========================================================================
    # Plot 1: Trajectory Comparison
    # ==========================================================================
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot multiple trajectories
    for i in range(min(5, len(pid_results['trajectories']))):
        alpha = 0.6 if i > 0 else 1.0
        label_pid = 'PID' if i == 0 else ''
        label_rl = 'RL' if i == 0 else ''
        
        ax.plot(np.array(pid_results['trajectories'][i]) * 1000, 
                color=colors['PID'], alpha=alpha, linewidth=1.5, label=label_pid)
        ax.plot(np.array(rl_results['trajectories'][i]) * 1000, 
                color=colors['RL'], alpha=alpha, linewidth=1.5, label=label_rl)
    
    ax.axhline(y=1, color='green', linestyle='--', linewidth=2, label='Target (1mm)')
    ax.axhline(y=5, color='orange', linestyle=':', linewidth=1.5, label='5mm threshold')
    
    ax.set_xlabel('Steps', fontsize=12)
    ax.set_ylabel('Distance to Goal (mm)', fontsize=12)
    ax.set_title('Trajectory Comparison: PID vs RL Controller', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.set_ylim(0, max(200, max(max(t) for t in pid_results['trajectories']) * 1000 * 1.1))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'trajectory_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Saved: trajectory_comparison.png")
    
    # ==========================================================================
    # Plot 2: Steady State Error Box Plot
    # ==========================================================================
    fig, ax = plt.subplots(figsize=(8, 6))
    
    pid_errors = np.array(pid_results['steady_state_errors']) * 1000
    rl_errors = np.array(rl_results['steady_state_errors']) * 1000
    
    bp = ax.boxplot([pid_errors, rl_errors], labels=['PID', 'RL'], 
                     patch_artist=True, widths=0.6)
    
    bp['boxes'][0].set_facecolor(colors['PID'])
    bp['boxes'][1].set_facecolor(colors['RL'])
    for box in bp['boxes']:
        box.set_alpha(0.7)
    
    ax.axhline(y=1, color='green', linestyle='--', linewidth=2, label='Target (1mm)')
    
    ax.set_ylabel('Steady State Error (mm)', fontsize=12)
    ax.set_title('Positioning Accuracy Comparison', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    
    # Add mean values as text
    ax.text(1, np.mean(pid_errors) + 2, f'Mean: {np.mean(pid_errors):.2f}mm', 
            ha='center', fontsize=10, color=colors['PID'])
    ax.text(2, np.mean(rl_errors) + 2, f'Mean: {np.mean(rl_errors):.2f}mm', 
            ha='center', fontsize=10, color=colors['RL'])
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'error_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Saved: error_comparison.png")
    
    # ==========================================================================
    # Plot 3: Multi-Metric Bar Chart
    # ==========================================================================
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Subplot 1: Steady State Error
    ax1 = axes[0, 0]
    means = [np.mean(pid_errors), np.mean(rl_errors)]
    stds = [np.std(pid_errors), np.std(rl_errors)]
    bars = ax1.bar(['PID', 'RL'], means, yerr=stds, capsize=5, 
                   color=[colors['PID'], colors['RL']], alpha=0.7)
    ax1.axhline(y=1, color='green', linestyle='--', label='Target')
    ax1.set_ylabel('Error (mm)')
    ax1.set_title('Steady State Error', fontweight='bold')
    ax1.legend()
    
    # Subplot 2: Response Time
    ax2 = axes[0, 1]
    pid_rt = np.mean(pid_results['response_times'])
    rl_rt = np.mean(rl_results['response_times'])
    ax2.bar(['PID', 'RL'], [pid_rt, rl_rt], 
            color=[colors['PID'], colors['RL']], alpha=0.7)
    ax2.set_ylabel('Steps')
    ax2.set_title('Response Time (to 5mm)', fontweight='bold')
    
    # Subplot 3: Success Rate
    ax3 = axes[1, 0]
    pid_sr = np.mean(pid_results['successes']) * 100
    rl_sr = np.mean(rl_results['successes']) * 100
    ax3.bar(['PID', 'RL'], [pid_sr, rl_sr], 
            color=[colors['PID'], colors['RL']], alpha=0.7)
    ax3.set_ylabel('Success Rate (%)')
    ax3.set_title('Success Rate (<1mm)', fontweight='bold')
    ax3.set_ylim(0, 100)
    
    # Subplot 4: Overshoot
    ax4 = axes[1, 1]
    pid_os = np.mean(pid_results['overshoots']) * 1000
    rl_os = np.mean(rl_results['overshoots']) * 1000
    ax4.bar(['PID', 'RL'], [pid_os, rl_os], 
            color=[colors['PID'], colors['RL']], alpha=0.7)
    ax4.set_ylabel('Overshoot (mm)')
    ax4.set_title('Mean Overshoot', fontweight='bold')
    
    plt.suptitle('PID vs RL Controller: Performance Metrics', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'metrics_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Saved: metrics_comparison.png")
    
    # ==========================================================================
    # Plot 4: Error Distribution Histogram
    # ==========================================================================
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bins = np.linspace(0, max(max(pid_errors), max(rl_errors)) * 1.1, 20)
    
    ax.hist(pid_errors, bins=bins, alpha=0.6, label='PID', color=colors['PID'], edgecolor='black')
    ax.hist(rl_errors, bins=bins, alpha=0.6, label='RL', color=colors['RL'], edgecolor='black')
    
    ax.axvline(x=1, color='green', linestyle='--', linewidth=2, label='Target (1mm)')
    
    ax.set_xlabel('Steady State Error (mm)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Error Distribution: PID vs RL', fontsize=14, fontweight='bold')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'error_distribution.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Saved: error_distribution.png")
    
    # ==========================================================================
    # Plot 5: Summary Table as Figure
    # ==========================================================================
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('off')
    
    # Create table data
    metrics = [
        'Steady State Error (mean)',
        'Steady State Error (std)',
        'Response Time (steps)',
        'Overshoot (mm)',
        'Success Rate (%)',
        'Min Error (mm)',
        'Max Error (mm)'
    ]
    
    pid_vals = [
        f'{np.mean(pid_errors):.3f}',
        f'{np.std(pid_errors):.3f}',
        f'{np.mean(pid_results["response_times"]):.1f}',
        f'{np.mean(pid_results["overshoots"])*1000:.3f}',
        f'{np.mean(pid_results["successes"])*100:.1f}',
        f'{np.min(pid_errors):.3f}',
        f'{np.max(pid_errors):.3f}'
    ]
    
    rl_vals = [
        f'{np.mean(rl_errors):.3f}',
        f'{np.std(rl_errors):.3f}',
        f'{np.mean(rl_results["response_times"]):.1f}',
        f'{np.mean(rl_results["overshoots"])*1000:.3f}',
        f'{np.mean(rl_results["successes"])*100:.1f}',
        f'{np.min(rl_errors):.3f}',
        f'{np.max(rl_errors):.3f}'
    ]
    
    # Determine winners
    winners = []
    for i, m in enumerate(metrics):
        if 'Error' in m or 'Overshoot' in m or 'Time' in m:
            winner = 'RL' if float(rl_vals[i]) < float(pid_vals[i]) else 'PID'
        else:  # Success Rate
            winner = 'RL' if float(rl_vals[i]) > float(pid_vals[i]) else 'PID'
        winners.append(winner)
    
    table_data = [[m, p, r, w] for m, p, r, w in zip(metrics, pid_vals, rl_vals, winners)]
    
    table = ax.table(
        cellText=table_data,
        colLabels=['Metric', 'PID', 'RL', 'Winner'],
        loc='center',
        cellLoc='center',
        colWidths=[0.4, 0.2, 0.2, 0.2]
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)
    
    # Style header
    for i in range(4):
        table[(0, i)].set_facecolor('#2c3e50')
        table[(0, i)].set_text_props(color='white', fontweight='bold')
    
    # Style winner column
    for i in range(1, len(metrics) + 1):
        winner = table_data[i-1][3]
        if winner == 'RL':
            table[(i, 3)].set_facecolor('#e8f8f5')
            table[(i, 2)].set_facecolor('#e8f8f5')
        else:
            table[(i, 3)].set_facecolor('#ebf5fb')
            table[(i, 1)].set_facecolor('#ebf5fb')
    
    ax.set_title('Performance Comparison Summary', fontsize=16, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'comparison_summary.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Saved: comparison_summary.png")
    
    print(f"\n{'='*60}")
    print(f"All visualizations saved to: {output_dir}/")
    print(f"{'='*60}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Generate PID vs RL comparison plots')
    parser.add_argument('--model_path', type=str, default='model/best_model',
                        help='Path to trained RL model')
    parser.add_argument('--n_episodes', type=int, default=20,
                        help='Number of episodes to evaluate')
    parser.add_argument('--output_dir', type=str, default='comparison_plots',
                        help='Output directory for plots')
    args = parser.parse_args()
    
    print("="*60)
    print("Generating PID vs RL Comparison Visualizations")
    print("="*60)
    
    # Create environment
    env = OT2Env(render=False, max_steps=100)
    
    # Evaluate PID
    print("\n[1/2] Evaluating PID Controller...")
    pid_controller = PIDController(Kp=10.0, Ki=1.0, Kd=2.0)
    pid_results = evaluate_controller(env, pid_controller, 'PID', n_episodes=args.n_episodes)
    print(f"      Mean Error: {np.mean(pid_results['steady_state_errors'])*1000:.2f} mm")
    
    # Evaluate RL
    if RL_AVAILABLE:
        print("[2/2] Evaluating RL Controller...")
        try:
            rl_controller = RLController(args.model_path)
            rl_results = evaluate_controller(env, rl_controller, 'RL', n_episodes=args.n_episodes)
            print(f"      Mean Error: {np.mean(rl_results['steady_state_errors'])*1000:.2f} mm")
        except Exception as e:
            print(f"Error loading RL model: {e}")
            rl_results = pid_results
    else:
        print("[2/2] RL not available, using dummy data...")
        rl_results = pid_results
    
    env.close()
    
    # Generate plots
    print("\nGenerating visualizations...")
    create_comparison_plots(pid_results, rl_results, args.output_dir)


if __name__ == "__main__":
    main()
