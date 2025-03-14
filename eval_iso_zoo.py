import os
import argparse
import gymnasium as gym
import numpy as np
import pandas as pd
import yaml
import json
import matplotlib.pyplot as plt
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from rl_zoo3.utils import ALGOS

from energy_net.market.iso.demand_patterns import DemandPattern
from energy_net.market.iso.cost_types import CostType
from energy_net.env import PricingPolicy
from gymnasium.wrappers import RescaleAction

def parse_args():
    """Parse command line arguments for ISO evaluation"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", type=str, default="ppo", help="RL algorithm to evaluate")
    parser.add_argument("--env", type=str, default="ISOEnv-v0", help="Environment ID")
    parser.add_argument("--model-path", type=str, required=True, help="Path to the trained model")
    parser.add_argument("--normalizer-path", type=str, help="Path to the normalizer file")
    parser.add_argument("--n-eval-episodes", type=int, default=5, help="Number of evaluation episodes")
    parser.add_argument("--deterministic", action="store_true", default=True, help="Use deterministic actions")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--demand-pattern", type=str, default="SINUSOIDAL",
                       choices=["SINUSOIDAL", "RANDOM", "PERIODIC", "SPIKES"],
                       help="Demand pattern to use")
    parser.add_argument("--cost-type", type=str, default="CONSTANT",
                       choices=["CONSTANT", "VARIABLE", "TIME_OF_USE"],
                       help="Cost type to use")
    parser.add_argument("--pricing-policy", type=str, default="ONLINE",
                       choices=["QUADRATIC", "ONLINE", "CONSTANT"],
                       help="Pricing policy to use")
    parser.add_argument("--num-pcs-agents", type=int, default=1,
                       help="Number of PCS agents to simulate in the environment")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory to save evaluation results")
    return parser.parse_args()

def create_env(args):
    """Create ISO evaluation environment"""
    # Convert pricing_policy string to enum
    pricing_policy_enum = PricingPolicy[args.pricing_policy]
    
    def _init():
        env = gym.make(
            args.env,
            render_mode="rgb_array",
            pricing_policy=pricing_policy_enum,
            demand_pattern=DemandPattern[args.demand_pattern],
            cost_type=CostType[args.cost_type],
            num_pcs_agents=args.num_pcs_agents
        )
        
        # Add action rescaling to match PCS pipeline
        env = RescaleAction(env, min_action=-1.0, max_action=1.0)
        
        # Set up monitoring for the environment
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
            monitor_path = os.path.join(args.output_dir, "eval_monitor")
            env = Monitor(env, monitor_path)
        
        return env
    return _init

def plot_episode_results(episode_data, episode_idx, output_dir):
    """Plot ISO performance metrics for a specific episode with formatting matching PCS plots"""
    # Create directory for plots if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract data
    time_steps = np.arange(len(episode_data['actions']))
    actions = episode_data['actions']
    
    # ===== Figure 1: Energy flows + Prices =====
    fig = plt.figure(figsize=(15, 12))
    
    # Upper plot: Energy flows
    ax1 = plt.subplot(2, 1, 1)
    
    # Plot demand and dispatch
    if 'dispatch_amounts' in episode_data:
        ax1.bar(time_steps, episode_data['dispatch_amounts'], width=0.8, color='lightblue', label='dispatch')
    
    if 'demand_predictions' in episode_data:
        ax1.plot(time_steps, episode_data['demand_predictions'], 'k--', linewidth=2, label='Predicted Demand')
    
    if 'actual_demand' in episode_data:
        ax1.plot(time_steps, episode_data['actual_demand'], 'b-', linewidth=2, label='Non Strategic Demand')
    
    # Add net demand if available, otherwise use actual demand
    if 'net_demand' in episode_data:
        ax1.plot(time_steps, episode_data['net_demand'], 'r-', linewidth=2, label='Total Demand')
    elif 'actual_demand' in episode_data:
        ax1.plot(time_steps, episode_data['actual_demand'], 'r-', linewidth=2, label='Total Demand')
    
    ax1.set_ylabel('Energy (MWh)', fontsize=12)
    ax1.set_title(f'ISO Energy Flows - Episode {episode_idx}', fontsize=14)
    ax1.grid(True, alpha=0.3)
    
    # Only add legend if there are labeled elements
    handles, labels = ax1.get_legend_handles_labels()
    if handles:
        ax1.legend(loc='upper right', fontsize=10)

    # Lower plot: Battery Levels and Prices
    ax2 = plt.subplot(2, 1, 2)
    
    # Add battery levels if available
    if 'battery_levels' in episode_data and episode_data['battery_levels']:
        for agent_idx, battery_level in enumerate(episode_data['battery_levels']):
            ax2.plot(time_steps, battery_level, '-', linewidth=2, label=f'PCS {agent_idx + 1} Battery')
    
    # Prices on secondary y-axis
    ax3 = ax2.twinx()
    ax3.plot(time_steps, actions[:, 1], 'r--', linewidth=2, label='ISO Sell Price')
    ax3.plot(time_steps, actions[:, 0], 'g--', linewidth=2, label='ISO Buy Price')
    ax3.set_ylabel('Price ($/MWh)', color='black', fontsize=12)
    ax3.tick_params(axis='y', labelcolor='black')
    
    # Set labels and grid for battery axis
    ax2.set_ylabel('Battery Level (MWh)', fontsize=12)
    ax2.set_xlabel('Time Step', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    # Combine legends
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax3.get_legend_handles_labels()
    if lines1 or lines2:
        ax3.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=10)

    plt.tight_layout()
    fig_path_1 = os.path.join(output_dir, f'episode_{episode_idx}_flows_prices.png')
    plt.savefig(fig_path_1, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved flows and prices plot to {fig_path_1}")
    
    # ===== Figure 2: Cost components =====
    if 'cost_components' in episode_data:
        cost_components = episode_data['cost_components']
        
        if cost_components:
            fig2 = plt.figure(figsize=(10, 6))
            ax4 = fig2.add_subplot(1, 1, 1)
            
            # Get cost components with consistent colors
            dispatch_costs = np.array(cost_components.get('dispatch_cost', np.zeros(len(time_steps))))
            pcs_costs = np.array(cost_components.get('pcs_costs', np.zeros(len(time_steps))))
            reserve_costs = np.array(cost_components.get('reserve_cost', np.zeros(len(time_steps))))
            
            # Create stacked bar chart
            ax4.bar(time_steps, dispatch_costs, label='Dispatch Cost', color='lightblue')
            ax4.bar(time_steps, pcs_costs, bottom=dispatch_costs, label='PCS Exchange Cost', color='lightgreen')
            ax4.bar(time_steps, reserve_costs, bottom=dispatch_costs + pcs_costs, label='Reserve Cost', color='salmon')
            
            ax4.set_xlabel('Time Step', fontsize=12)
            ax4.set_ylabel('Cost ($)', fontsize=12)
            ax4.set_title('Cost Components Over Time', fontsize=14)
            ax4.grid(True, alpha=0.3)
            ax4.legend(loc='upper right', fontsize=10)
            
            fig2.tight_layout()
            fig_path_2 = os.path.join(output_dir, f'episode_{episode_idx}_cost_components.png')
            plt.savefig(fig_path_2, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Saved cost components plot to {fig_path_2}")
    
    # ===== Figure 3: Final cost distribution =====
    if 'cost_components' in episode_data and episode_data['cost_components']:
        fig3 = plt.figure(figsize=(6, 6))
        ax5 = fig3.add_subplot(1, 1, 1)
        
        # Calculate total costs per component
        cost_components = episode_data['cost_components']
        total_dispatch = sum(cost_components.get('dispatch_cost', [0]))
        total_pcs = sum(cost_components.get('pcs_costs', [0]))
        total_reserve = sum(cost_components.get('reserve_cost', [0]))
        
        # Create stacked bar chart for total costs
        ax5.bar([0], [total_dispatch], color='lightblue', label='Dispatch Cost')
        ax5.bar([0], [total_pcs], bottom=[total_dispatch], color='lightgreen', label='PCS Exchange Cost')
        ax5.bar([0], [total_reserve], bottom=[total_dispatch + total_pcs], color='salmon', label='Reserve Cost')
        
        ax5.set_ylabel('Total Cost ($)', fontsize=12)
        ax5.set_title('Episode Final Cost Distribution', fontsize=14)
        ax5.set_xticks([])
        
        # Place legend on the right side
        ax5.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=10)
        
        fig3.tight_layout()
        final_cost_path = os.path.join(output_dir, f'episode_{episode_idx}_final_cost_distribution.png')
        plt.savefig(final_cost_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved final cost distribution plot to {final_cost_path}")

def collect_episode_data(env, model, deterministic=True):
    """Collect data from a single episode for an ISO agent"""
    # Reset the environment and get initial state
    observation, _ = env.reset()
    done = False
    
    # Initialize data containers
    actions = []
    infos = []
    demand_predictions = []
    actual_demand = []
    net_demand_values = []
    dispatch_amounts = []
    battery_levels = []
    cost_components = {}
    
    # Run the episode
    while not done:
        # Get action from the model
        action, _ = model.predict(observation, deterministic=deterministic)
        actions.append(action)
        
        # Step the environment
        observation, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        infos.append(info)
        
        # Debug: Print the first info dict to see what's available
        if len(infos) == 1:
            print("\nInfo dictionary keys available from ISO environment:")
            for key in info.keys():
                print(f"- {key}: {type(info[key])}")
            print(f"Info sample: {info}")
        
        # Extract specific ISO metrics from info if available
        # Map ISO environment keys to expected plotting keys
        if 'predicted_demand' in info:
            demand_predictions.append(info['predicted_demand'])
        if 'realized_demand' in info:
            actual_demand.append(info['realized_demand'])
        if 'net_demand' in info:
            net_demand_values.append(info['net_demand'])
        if 'dispatch' in info:
            dispatch_amounts.append(info['dispatch'])
        if 'battery_level' in info:
            battery_levels.append(info['battery_level'])
        
        # Create cost components dictionary from available cost information
        if 'dispatch_cost' in info or 'reserve_cost' in info or 'pcs_costs' in info:
            # Initialize cost components if not already done
            if not cost_components:
                cost_components = {
                    'dispatch_cost': [],
                    'reserve_cost': [],
                    'pcs_costs': []
                }
            
            # Add cost values for this step
            cost_components['dispatch_cost'].append(float(info.get('dispatch_cost', 0)))
            cost_components['reserve_cost'].append(float(info.get('reserve_cost', 0)))
            cost_components['pcs_costs'].append(float(info.get('pcs_costs', 0)))
    
    # Compile data
    episode_data = {
        'actions': np.array(actions),
        'infos': infos
    }
    
    if demand_predictions:
        episode_data['demand_predictions'] = np.array(demand_predictions)
    if actual_demand:
        episode_data['actual_demand'] = np.array(actual_demand)
    if net_demand_values:
        episode_data['net_demand'] = np.array(net_demand_values)
    if dispatch_amounts:
        episode_data['dispatch_amounts'] = np.array(dispatch_amounts)
    if battery_levels:
        # Transpose the battery levels to get one array per PCS agent
        if battery_levels and isinstance(battery_levels[0], list) and len(battery_levels[0]) > 0:
            transposed_levels = list(map(list, zip(*battery_levels)))
            episode_data['battery_levels'] = transposed_levels
    if cost_components and any(len(v) > 0 for v in cost_components.values()):
        episode_data['cost_components'] = cost_components
    
    # Debug: Print what data was collected
    print("\nData collected from ISO environment:")
    for key in episode_data.keys():
        if key != 'infos':  # Skip printing the full infos list
            print(f"- {key}: {type(episode_data[key])}")
    print(f"Total steps collected: {len(actions)}")
    
    return episode_data

def main():
    """Main function for evaluating ISO agent"""
    args = parse_args()
    
    # Set random seed
    set_random_seed(args.seed)
    
    # Create output directory if specified
    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)
    
    # Print evaluation parameters
    print(f"Evaluating ISO agent with:")
    print(f"  Algorithm: {args.algo}")
    print(f"  Environment: {args.env}")
    print(f"  Demand Pattern: {args.demand_pattern}")
    print(f"  Cost Type: {args.cost_type}")
    print(f"  Pricing Policy: {args.pricing_policy}")
    print(f"  Number of PCS Agents: {args.num_pcs_agents}")
    print(f"  Model Path: {args.model_path}")
    if args.normalizer_path:
        print(f"  Normalizer Path: {args.normalizer_path}")
    print(f"  Episodes: {args.n_eval_episodes}")
    print(f"  Deterministic: {args.deterministic}")
    if args.output_dir:
        print(f"  Output Directory: {args.output_dir}")
    
    # Create environment
    env_fn = create_env(args)
    env = DummyVecEnv([env_fn])
    
    # Load normalizer if provided
    if args.normalizer_path:
        try:
            env = VecNormalize.load(args.normalizer_path, env)
            # Important: disable updates and reward normalization during evaluation
            env.training = False  # Don't update normalization statistics during evaluation
            env.norm_reward = False  # Don't normalize rewards during evaluation
            print(f"Loaded normalizer from {args.normalizer_path}")
            print(f"Observation scaling: mean={env.obs_rms.mean}, var={env.obs_rms.var}")
        except Exception as e:
            print(f"Error loading normalizer: {e}")
            print("Continuing without normalization - THIS WILL LIKELY CAUSE POOR PERFORMANCE")
    
    # Load the model
    model = ALGOS[args.algo.lower()].load(args.model_path)
    print("Loaded model from", args.model_path)
    
    # Run evaluation
    mean_reward, std_reward = evaluate_policy(
        model,
        env,
        n_eval_episodes=args.n_eval_episodes,
        deterministic=args.deterministic,
        return_episode_rewards=False,
        warn=True
    )
    
    print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
    
    # If output directory is specified, save results and generate plots
    if args.output_dir:
        # Prepare to collect detailed data
        env_fn = create_env(args)
        single_env = env_fn()
        
        if args.normalizer_path:
            wrapped_env = VecNormalize.load(args.normalizer_path, DummyVecEnv([lambda: single_env]))
            wrapped_env.norm_reward = False
            obs_rms = wrapped_env.obs_rms
            single_env = wrapped_env.envs[0]
        
        all_episode_data = []
        for episode_idx in range(args.n_eval_episodes):
            # Collect detailed data for the episode
            episode_data = collect_episode_data(single_env, model, deterministic=args.deterministic)
            all_episode_data.append(episode_data)
            
            # Plot the episode results
            plot_episode_results(episode_data, episode_idx, args.output_dir)
        
        # Save the evaluation results to CSV
        results = {
            'mean_reward': [mean_reward],
            'std_reward': [std_reward],
            'n_episodes': [args.n_eval_episodes],
            'algorithm': [args.algo],
            'environment': [args.env],
            'demand_pattern': [args.demand_pattern],
            'cost_type': [args.cost_type],
            'pricing_policy': [args.pricing_policy],
            'num_pcs_agents': [args.num_pcs_agents]
        }
        
        results_df = pd.DataFrame(results)
        results_path = os.path.join(args.output_dir, 'evaluation_results.csv')
        results_df.to_csv(results_path, index=False)
        print(f"Saved evaluation results to {results_path}")
        
        # Save plots
        print(f"Saved plots to {args.output_dir}")
    
    # Close the environment
    env.close()

if __name__ == "__main__":
    main()