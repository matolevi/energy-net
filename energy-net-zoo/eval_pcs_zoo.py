import os
import argparse
import gymnasium as gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from rl_zoo3.utils import ALGOS
from gymnasium.wrappers import RescaleAction  # Add this import

from energy_net.market.iso.demand_patterns import DemandPattern
from energy_net.market.iso.cost_types import CostType

def parse_args():
    """Parse command line arguments for PCS evaluation"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", type=str, required=True, help="RL Algorithm", choices=list(ALGOS.keys()))
    parser.add_argument("--env", type=str, default="PCSUnitEnv-v0", help="environment ID")
    parser.add_argument("--model-path", type=str, required=True, help="Path to the trained model")
    parser.add_argument("--normalizer-path", type=str, help="Path to the saved normalizer")
    parser.add_argument("--n-eval-episodes", type=int, default=10, help="Number of evaluation episodes")
    parser.add_argument("--demand-pattern", type=str, default="SINUSOIDAL", choices=[p.name for p in DemandPattern])
    parser.add_argument("--cost-type", type=str, default="CONSTANT", choices=[c.name for c in CostType])
    parser.add_argument("--output-dir", type=str, default="eval_results", help="Directory to save evaluation results")
    parser.add_argument("--deterministic", action="store_true", help="Use deterministic actions")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()

def create_env(args):
    """Create PCS evaluation environment"""
    def _init():
        env = gym.make(
            args.env,
            demand_pattern=DemandPattern[args.demand_pattern],
            cost_type=CostType[args.cost_type],
            env_config_path='configs/environment_config.yaml',
            iso_config_path='configs/iso_config.yaml',
            pcs_unit_config_path='configs/pcs_unit_config.yaml'
        )
        
        # Add explicit action rescaling - CRITICAL to match training environment
        env = RescaleAction(env, min_action=-1.0, max_action=1.0)
        
        # Set up monitoring for the environment
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
            monitor_path = os.path.join(args.output_dir, "eval_monitor")
            env = Monitor(env, monitor_path)
        
        return env
    return _init

def collect_episode_data(env, model, deterministic=True):
    """Collect data from a single episode for a PCS agent"""
    # Handle different reset() return formats correctly (observation or observation+info)
    try:
        # Try the newer Gymnasium API format
        reset_result = env.reset()
        if isinstance(reset_result, tuple) and len(reset_result) == 2:
            observation, _ = reset_result
        else:
            # Handle case where only observation is returned (vectorized environments often do this)
            observation = reset_result
    except Exception as e:
        print(f"Error during environment reset: {e}")
        print(f"Reset returned: {env.reset()}")
        raise

    done = False
    
    # Initialize data containers
    raw_actions = []  # Actions before validation
    validated_actions = []  # Actions after validation
    steps = []
    rewards = []
    infos = []  # Store all info dictionaries
    production = []
    consumption = []
    net_exchange = []
    battery_level = []
    predicted_demand = []
    realized_demand = []
    iso_sell_prices = []
    iso_buy_prices = []
    dispatch = []
    
    # Initialize cost components with empty lists
    cost_components = {
        'dispatch_cost': [],
        'pcs_exchange_cost': [],
        'reserve_cost': []
    }
    
    step_count = 0
    episode_reward = 0.0
    
    # Run the episode
    while not done:
        # Get action from the model
        action, _ = model.predict(observation, deterministic=deterministic)
        
        # Store raw action
        raw_action = float(action[0]) if isinstance(action, np.ndarray) and len(action) > 0 else float(action)
        raw_actions.append(raw_action)
        
        # Step the environment - handle vectorized environments correctly
        step_result = env.step(action)
        
        # Check if we have a vectorized environment (returns 4 values) or standard env (returns 5 values)
        if len(step_result) == 4:
            # Vectorized environment returns (obs, rewards, dones, infos)
            observation, reward, dones, info = step_result
            done = dones[0] if isinstance(dones, (list, np.ndarray)) else dones
            # No separate terminated/truncated in this case
        else:
            # Standard environment returns (obs, reward, terminated, truncated, info)
            observation, reward, terminated, truncated, info = step_result
            done = terminated or truncated
        
        # Extract scalar values from numpy arrays if needed
        reward = float(reward[0]) if isinstance(reward, (np.ndarray, list, tuple)) else float(reward)
        info = info[0] if isinstance(info, (list, tuple)) else info
        infos.append(info)
        episode_reward += reward
        
        # Debug: Print the first info dict to see what's available
        if len(infos) == 1:
            print("\nInfo dictionary keys available from PCS environment:")
            for key in info.keys():
                print(f"- {key}: {type(info[key])}")
            print(f"Info sample: {info}")
        
        # Store all data
        validated_actions.append(float(info.get('battery_action', raw_action)))  # Use validated action from info
        steps.append(step_count)
        rewards.append(reward)  # Already converted to float above
        production.append(float(info.get('production', 0.0)))
        consumption.append(float(info.get('consumption', 0.0)))
        net_exchange.append(float(info.get('net_exchange', 0.0)))
        battery_level.append(float(info.get('battery_level', 0.0)))
        predicted_demand.append(float(info.get('predicted_demand', 0.0)))
        realized_demand.append(float(info.get('realized_demand', 0.0)))
        iso_sell_prices.append(float(info.get('iso_sell_price', 0.0)))
        iso_buy_prices.append(float(info.get('iso_buy_price', 0.0)))
        dispatch.append(float(info.get('dispatch', 0.0)))
        
        # Get cost components directly from environment info
        cost_components['dispatch_cost'].append(float(info.get('dispatch_cost', 0.0)))
        cost_components['pcs_exchange_cost'].append(float(info.get('pcs_exchange_cost', 0.0)))
        cost_components['reserve_cost'].append(float(info.get('reserve_cost', 0.0)))
        
        step_count += 1
    
    # Print episode summary
    print(f"\nEpisode Summary:")
    print(f"Total Steps: {step_count}")
    print(f"Total Reward: {episode_reward:.4f}")
    print(f"Mean Step Reward: {np.mean(rewards):.4f}")
    print(f"Final Battery Level: {battery_level[-1]:.4f}")
    print(f"Total Net Exchange: {sum(net_exchange):.4f}")
    
    # Convert lists to numpy arrays for easier manipulation
    episode_data = {
        'steps': np.array(steps),
        'raw_actions': np.array(raw_actions),  # Original actions from model
        'actions': np.array(validated_actions),  # Validated actions that were actually executed
        'rewards': np.array(rewards),
        'total_reward': float(episode_reward),  # Ensure this is a scalar
        'production': np.array(production),
        'consumption': np.array(consumption),
        'net_exchange': np.array(net_exchange),
        'battery_level': np.array(battery_level),
        'predicted_demand': np.array(predicted_demand),
        'realized_demand': np.array(realized_demand),
        'iso_sell_prices': np.array(iso_sell_prices),
        'iso_buy_prices': np.array(iso_buy_prices),
        'dispatch': np.array(dispatch),
        'cost_components': {
            'dispatch_cost': np.array(cost_components['dispatch_cost']),
            'pcs_exchange_cost': np.array(cost_components['pcs_exchange_cost']),
            'reserve_cost': np.array(cost_components['reserve_cost'])
        },
        'infos': infos  # Store all info dictionaries for reference
    }
    
    # Debug: Print what data was collected
    print("\nData collected from PCS environment:")
    for key in episode_data.keys():
        if key != 'infos':  # Skip printing the full infos list
            print(f"- {key}: {type(episode_data[key])}")
    print(f"Total steps collected: {len(validated_actions)}")
    
    return episode_data

def plot_episode_results(episode_data: dict, episode_num: int, save_path: str, agent_name: str = "PCS"):
    """Generate visualization similar to ActionTrackingCallback, matching ISO style exactly"""
    if not episode_data:
        print(f"No data for episode {episode_num}")
        return
    
    # Extract data from the episode_data dictionary
    steps = episode_data['steps']
    production = episode_data['production']
    consumption = episode_data['consumption']
    net_exchange = episode_data['net_exchange']
    battery_level = episode_data['battery_level']
    predicted_demand = episode_data['predicted_demand']
    realized_demand = episode_data['realized_demand']
    iso_sell_prices = episode_data['iso_sell_prices']
    iso_buy_prices = episode_data['iso_buy_prices']
    dispatch = episode_data['dispatch']
    
    # Compute net demand
    net_demand = realized_demand + net_exchange
    
    # Get cost components
    dispatch_costs = episode_data['cost_components']['dispatch_cost']
    pcs_costs = episode_data['cost_components']['pcs_exchange_cost']
    reserve_costs = episode_data['cost_components']['reserve_cost']
    
    # Define ISO-consistent colors
    COLORS = {
        'dispatch': '#ADD8E6',  # lightblue
        'pcs_exchange': '#90EE90',  # lightgreen
        'reserve': '#FA8072',  # salmon
        'predicted_demand': 'black',
        'realized_demand': '#1f77b4',  # blue
        'total_demand': '#d62728',  # red
        'battery_level': '#1f77b4',  # blue
        'iso_sell': '#d62728',  # red
        'iso_buy': '#2ca02c'  # green
    }
    
    # ===== Figure 1: Energy flows + Battery levels and Prices =====
    fig = plt.figure(figsize=(15, 12))
    
    # Upper plot: Energy flows
    ax1 = plt.subplot(2, 1, 1)
    
    # Plot dispatch bars with consistent color
    ax1.bar(steps, dispatch, width=0.8, color=COLORS['dispatch'], label='Dispatch')
    
    # Plot demand lines with consistent colors
    ax1.plot(steps, predicted_demand, '--', color=COLORS['predicted_demand'], linewidth=2, label='Predicted Demand')
    ax1.plot(steps, realized_demand, '-', color=COLORS['realized_demand'], linewidth=2, label='Non Strategic Demand')
    ax1.plot(steps, net_demand, '-', color=COLORS['total_demand'], linewidth=2, label='Total Demand')
    
    ax1.set_ylabel('Energy (MWh)', fontsize=12)
    ax1.set_title(f'{agent_name} Energy Flows - Episode {episode_num}', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right', fontsize=10)

    # Lower plot: Battery Levels and Prices
    ax2 = plt.subplot(2, 1, 2)
    
    # Battery levels with consistent color
    ax2.plot(steps, battery_level, '-', linewidth=2, label='Battery Level', color=COLORS['battery_level'])
    
    # Prices on secondary y-axis with consistent colors
    ax3 = ax2.twinx()
    ax3.plot(steps, iso_sell_prices, '--', linewidth=2, label='ISO Sell Price', color=COLORS['iso_sell'])
    ax3.plot(steps, iso_buy_prices, '--', linewidth=2, label='ISO Buy Price', color=COLORS['iso_buy'])
    ax3.set_ylabel('Price ($/MWh)', color='black', fontsize=12)
    ax3.tick_params(axis='y', labelcolor='black')
    
    ax2.set_ylabel('Battery Level (MWh)', fontsize=12)
    ax2.set_xlabel('Time Step', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    # Combine legends
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax3.get_legend_handles_labels()
    ax3.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=10)

    plt.tight_layout()
    fig_path_1 = os.path.join(save_path, f'episode_{episode_num}_flows_prices.png')
    plt.savefig(fig_path_1, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved flows and prices plot to {fig_path_1}")
    
    # ===== Figure 2: Cost components with data for every time step =====
    fig2 = plt.figure(figsize=(10, 6))
    ax4 = fig2.add_subplot(1, 1, 1)
    
    # Create stacked bar chart with consistent colors and data for every time step
    width = 0.8  # Width of the bars
    
    # Plot each component separately
    ax4.bar(steps, dispatch_costs, width=width, label='Dispatch Cost', color=COLORS['dispatch'])
    ax4.bar(steps, pcs_costs, width=width, bottom=dispatch_costs, label='PCS Exchange Cost', color=COLORS['pcs_exchange'])
    ax4.bar(steps, reserve_costs, width=width, bottom=dispatch_costs + pcs_costs, label='Reserve Cost', color=COLORS['reserve'])
    
    ax4.set_xlabel('Time Step', fontsize=12)
    ax4.set_ylabel('Cost ($)', fontsize=12)
    ax4.set_title('Cost Components Over Time', fontsize=14)
    ax4.grid(True, alpha=0.3)
    ax4.legend(loc='upper right', fontsize=10)
    
    # Ensure x-axis shows all time steps
    ax4.set_xticks(steps)
    ax4.set_xticklabels(steps, rotation=45)
    
    # Add minor gridlines for better visibility
    ax4.grid(True, which='major', alpha=0.3)
    ax4.grid(True, which='minor', alpha=0.15)
    
    fig2.tight_layout()
    fig_path_2 = os.path.join(save_path, f'episode_{episode_num}_cost_components.png')
    plt.savefig(fig_path_2, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved cost components plot to {fig_path_2}")
    
    # ===== Figure 3: Final cost distribution with consistent colors =====
    fig3 = plt.figure(figsize=(6, 6))
    ax5 = fig3.add_subplot(1, 1, 1)
    
    # Calculate total costs
    total_dispatch = np.sum(dispatch_costs)
    total_pcs = np.sum(pcs_costs)
    total_reserve = np.sum(reserve_costs)
    
    # Add small values if all costs are zero
    if total_dispatch == 0 and total_pcs == 0 and total_reserve == 0:
        total_dispatch = 0.1
        total_pcs = 0.1 
        total_reserve = 0.1
    
    # Create stacked bar with consistent colors
    ax5.bar([0], [total_dispatch], color=COLORS['dispatch'], label='Dispatch Cost')
    ax5.bar([0], [total_pcs], bottom=[total_dispatch], color=COLORS['pcs_exchange'], label='PCS Exchange Cost')
    ax5.bar([0], [total_reserve], bottom=[total_dispatch + total_pcs], color=COLORS['reserve'], label='Reserve Cost')
    
    ax5.set_ylabel('Total Cost ($)', fontsize=12)
    ax5.set_title('Episode Final Cost Distribution', fontsize=14)
    ax5.set_xticks([])
    ax5.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=10)
    
    fig3.tight_layout()
    final_cost_path = os.path.join(save_path, f'episode_{episode_num}_final_cost_distribution.png')
    plt.savefig(final_cost_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved final cost distribution plot to {final_cost_path}")
    
    # ===== Figure 4: PCS Actions with consistent style =====
    raw_actions = episode_data['raw_actions']
    validated_actions = episode_data['actions']
    plt.figure(figsize=(10, 6))
    plt.plot(steps, raw_actions, '--', linewidth=2, label='Raw Action', color='gray', alpha=0.5)
    plt.plot(steps, validated_actions, '-', linewidth=2, label='Validated Action', color=COLORS['battery_level'])
    plt.axhline(y=0, color=COLORS['total_demand'], linestyle='--', alpha=0.3)
    plt.xlabel('Time Step', fontsize=12)
    plt.ylabel('Battery Action', fontsize=12)
    plt.title(f'PCS Agent Actions - Episode {episode_num}', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    actions_path = os.path.join(save_path, f'episode_{episode_num}_pcs_actions.png')
    plt.savefig(actions_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved PCS actions plot to {actions_path}")

def main():
    """Main function for evaluating PCS agent"""
    args = parse_args()
    
    # Set random seed
    set_random_seed(args.seed)
    
    # Create output directory if specified
    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)
    
    # Print evaluation parameters
    print("\n" + "="*50)
    print("PCS Agent Evaluation Parameters")
    print("="*50)
    print(f"  Algorithm: {args.algo}")
    print(f"  Environment: {args.env}")
    print(f"  Demand Pattern: {args.demand_pattern}")
    print(f"  Cost Type: {args.cost_type}")
    print(f"  Model Path: {args.model_path}")
    if args.normalizer_path:
        print(f"  Normalizer Path: {args.normalizer_path}")
    print(f"  Episodes: {args.n_eval_episodes}")
    print(f"  Deterministic: {args.deterministic}")
    if args.output_dir:
        print(f"  Output Directory: {args.output_dir}")
    print("="*50 + "\n")
    
    # Create environment
    env_fn = create_env(args)
    
    # Create and wrap evaluation environment
    single_env = env_fn()
    vec_env = DummyVecEnv([lambda: single_env])
    
    # Load normalizer if provided
    if args.normalizer_path:
        try:
            vec_env = VecNormalize.load(args.normalizer_path, vec_env)
            # Important: disable updates and reward normalization during evaluation
            vec_env.training = False  # Don't update normalization statistics during evaluation
            vec_env.norm_reward = False  # Don't normalize rewards during evaluation
            print(f"Loaded normalizer from {args.normalizer_path}")
            print(f"Observation scaling: mean={vec_env.obs_rms.mean}, var={vec_env.obs_rms.var}")
        except Exception as e:
            print(f"Error loading normalizer: {e}")
            print("Continuing without normalization - THIS WILL LIKELY CAUSE POOR PERFORMANCE")
    
    # Load the model
    try:
        # Make sure to pass the correct environment to the model
        model = ALGOS[args.algo.lower()].load(args.model_path, env=vec_env)
        print(f"Loaded model from {args.model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Run evaluation using evaluate_policy for overall metrics
    print("\nRunning evaluation with normalize_reward=False...")
    mean_reward, std_reward = evaluate_policy(
        model,
        vec_env,
        n_eval_episodes=args.n_eval_episodes,
        deterministic=args.deterministic,
        return_episode_rewards=False,
        warn=True
    )
    
    print("\n" + "="*50)
    print("Overall Evaluation Results")
    print("="*50)
    print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
    print("="*50 + "\n")
    
    # If output directory is specified, save results and generate plots
    if args.output_dir:
        print("\nCollecting detailed episode data...")
        
        all_episode_data = []
        episode_rewards = []
        for episode_idx in range(args.n_eval_episodes):
            print(f"\nRunning episode {episode_idx + 1}/{args.n_eval_episodes}")
            
            # Reset VecNormalize before collecting detailed data
            if isinstance(vec_env, VecNormalize):
                print("Resetting normalizer for detailed data collection")
                vec_env.reset()
            
            # Collect detailed data for the episode
            episode_data = collect_episode_data(vec_env, model, deterministic=args.deterministic)
            all_episode_data.append(episode_data)
            episode_rewards.append(episode_data['total_reward'])
            
            # Plot the episode results
            plot_episode_results(episode_data, episode_idx, args.output_dir)
        
        # Print final statistics
        print("\n" + "="*50)
        print("Final Statistics")
        print("="*50)
        print(f"Episodes run: {args.n_eval_episodes}")
        print(f"Mean episode reward: {np.mean(episode_rewards):.4f}")
        print(f"Std episode reward: {np.std(episode_rewards):.4f}")
        print(f"Min episode reward: {np.min(episode_rewards):.4f}")
        print(f"Max episode reward: {np.max(episode_rewards):.4f}")
        print("="*50 + "\n")
        
        # Save the evaluation results to CSV
        results = {
            'mean_reward': [mean_reward],
            'std_reward': [std_reward],
            'n_episodes': [args.n_eval_episodes],
            'algorithm': [args.algo],
            'environment': [args.env],
            'demand_pattern': [args.demand_pattern],
            'cost_type': [args.cost_type],
            'min_episode_reward': [np.min(episode_rewards)],
            'max_episode_reward': [np.max(episode_rewards)]
        }
        
        results_df = pd.DataFrame(results)
        results_path = os.path.join(args.output_dir, 'evaluation_results.csv')
        results_df.to_csv(results_path, index=False)
        print(f"Saved evaluation results to {results_path}")
    
    # Close the environment
    if hasattr(vec_env, 'close'):
        vec_env.close()

if __name__ == "__main__":
    main()