"""
Script to load and plot data from Multiple_Trials folder.
Creates comparison plots and numerical metrics for LC-SAC vs Baseline SAC:
1. Average reward comparison
2. Trajectory XZ comparison (LC-SAC vs Baseline vs Reference)
3. Lyapunov loss decay (LC-SAC only)
4. Comprehensive numerical metrics comparison (episode rewards, evaluation rewards, convergence metrics)
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
import glob


def get_file_prefix(algorithm_name):
    """
    Convert algorithm name to file prefix format.

    Args:
        algorithm_name: Name of algorithm ('baseline_SAC' or 'LC_SAC')

    Returns:
        File prefix string (e.g., 'baseline_sac' or 'lc-sac')
    """
    # Map algorithm names to their file prefix format
    prefix_map = {
        'baseline_SAC': 'baseline_sac',
        'LC_SAC': 'lc-sac'
    }
    return prefix_map.get(algorithm_name, algorithm_name.lower().replace('_', '-'))


def load_aggregated_data(base_dir, algorithm_name):
    """
    Load aggregated data from Multiple_Trials folder.

    Args:
        base_dir: Base directory containing algorithm folders
        algorithm_name: Name of algorithm ('baseline_SAC' or 'LC_SAC')

    Returns:
        Dictionary with loaded data or None if files don't exist
    """
    algo_dir = base_dir / algorithm_name
    if not algo_dir.exists():
        print(f"Warning: Directory {algo_dir} does not exist")
        return None

    data = {}
    file_prefix = get_file_prefix(algorithm_name)

    # Load episode rewards
    episode_rewards_file = algo_dir / f"{file_prefix}_episode_rewards_all_trials.npy"
    if episode_rewards_file.exists():
        data['episode_rewards_all'] = np.load(episode_rewards_file)
        data['episode_rewards_mean'] = np.load(algo_dir / f"{file_prefix}_episode_rewards_mean.npy")
        data['episode_rewards_std'] = np.load(algo_dir / f"{file_prefix}_episode_rewards_std.npy")
        print(f"Loaded episode rewards from {algorithm_name}")

    # Load evaluation rewards
    eval_rewards_file = algo_dir / f"{file_prefix}_eval_rewards_all_trials.npy"
    if eval_rewards_file.exists():
        data['eval_rewards_all'] = np.load(eval_rewards_file)
        data['eval_rewards_steps'] = np.load(algo_dir / f"{file_prefix}_eval_rewards_steps.npy")
        print(f"Loaded evaluation rewards from {algorithm_name}")

    # Load summary
    summary_file = algo_dir / f"{file_prefix}_summary.json"
    if summary_file.exists():
        with open(summary_file, 'r', encoding='utf-8') as f:
            data['summary'] = json.load(f)
        print(f"Loaded summary from {algorithm_name}")

    return data if data else None


def load_trajectory_data_from_trials(base_dir, algorithm_name, trajectory_type='circle'):
    """
    Load trajectory data from individual trial folders.

    Args:
        base_dir: Base directory containing algorithm folders
        algorithm_name: Name of algorithm ('baseline_SAC' or 'LC_SAC')
        trajectory_type: Trajectory type suffix

    Returns:
        List of trajectory data dictionaries, one per trial
    """
    algo_dir = base_dir / algorithm_name
    if not algo_dir.exists():
        return []

    trajectory_data_list = []
    trial_dirs = sorted([d for d in algo_dir.iterdir() if d.is_dir() and 'trial' in d.name])

    for trial_dir in trial_dirs:
        # Look for trajectory_data files
        traj_files = list(trial_dir.glob(f'trajectory_data*{trajectory_type}.json'))
        if traj_files:
            with open(traj_files[0], 'r', encoding='utf-8') as f:
                traj_data = json.load(f)
                trajectory_data_list.append(traj_data)
            print(f"Loaded trajectory data from {trial_dir.name}")

    return trajectory_data_list


def plot_reward_comparison(baseline_data, lcsac_data, output_dir):
    """
    Plot average reward comparison between LC-SAC and Baseline SAC.

    Args:
        baseline_data: Dictionary with baseline SAC data
        lcsac_data: Dictionary with LC-SAC data
        output_dir: Output directory for plots
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))

    # Plot 1: Episode Rewards Comparison
    if baseline_data and 'episode_rewards_mean' in baseline_data:
        baseline_mean = baseline_data['episode_rewards_mean']
        baseline_std = baseline_data['episode_rewards_std']
        episodes_baseline = np.arange(len(baseline_mean))

        axes[0].plot(episodes_baseline, baseline_mean, label='Baseline SAC', linewidth=2, color='C0')
        axes[0].fill_between(episodes_baseline, baseline_mean - baseline_std,
                            baseline_mean + baseline_std, alpha=0.3, color='C0')

    if lcsac_data and 'episode_rewards_mean' in lcsac_data:
        lcsac_mean = lcsac_data['episode_rewards_mean']
        lcsac_std = lcsac_data['episode_rewards_std']
        episodes_lcsac = np.arange(len(lcsac_mean))

        axes[0].plot(episodes_lcsac, lcsac_mean, label='LC-SAC', linewidth=2, color='C1')
        axes[0].fill_between(episodes_lcsac, lcsac_mean - lcsac_std,
                            lcsac_mean + lcsac_std, alpha=0.3, color='C1')

    axes[0].set_xlabel('Episode', fontsize=12)
    axes[0].set_ylabel('Average Reward', fontsize=12)
    axes[0].set_title('Episode Rewards Comparison (Mean ± Std)', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Evaluation Rewards Comparison
    if baseline_data and 'eval_rewards_all' in baseline_data:
        baseline_eval = baseline_data['eval_rewards_all']
        baseline_steps = baseline_data['eval_rewards_steps']
        baseline_eval_mean = np.nanmean(baseline_eval, axis=0)
        baseline_eval_std = np.nanstd(baseline_eval, axis=0)

        axes[1].plot(baseline_steps, baseline_eval_mean, 'o-', label='Baseline SAC',
                    linewidth=2, markersize=4, color='C0')
        axes[1].fill_between(baseline_steps, baseline_eval_mean - baseline_eval_std,
                            baseline_eval_mean + baseline_eval_std, alpha=0.3, color='C0')

    if lcsac_data and 'eval_rewards_all' in lcsac_data:
        lcsac_eval = lcsac_data['eval_rewards_all']
        lcsac_steps = lcsac_data['eval_rewards_steps']
        lcsac_eval_mean = np.nanmean(lcsac_eval, axis=0)
        lcsac_eval_std = np.nanstd(lcsac_eval, axis=0)

        axes[1].plot(lcsac_steps, lcsac_eval_mean, 'o-', label='LC-SAC',
                    linewidth=2, markersize=4, color='C1')
        axes[1].fill_between(lcsac_steps, lcsac_eval_mean - lcsac_eval_std,
                            lcsac_eval_mean + lcsac_eval_std, alpha=0.3, color='C1')

    axes[1].set_xlabel('Training Steps', fontsize=12)
    axes[1].set_ylabel('Average Evaluation Reward', fontsize=12)
    axes[1].set_title('Evaluation Rewards Comparison (Mean ± Std)', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    # Save plot
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_path = output_dir / "reward_comparison.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved reward comparison plot to {plot_path}")
    plt.show()


def load_lyapunov_loss_data(base_dir, algorithm_name, trajectory_type='circle'):
    """
    Load Lyapunov loss data from individual trial folders.

    Args:
        base_dir: Base directory containing algorithm folders
        algorithm_name: Name of algorithm ('LC_SAC' only, as baseline doesn't have Lyapunov loss)
        trajectory_type: Trajectory type suffix

    Returns:
        Dictionary with 'steps' (list of step arrays) and 'lyap_losses' (list of loss arrays)
    """
    algo_dir = base_dir / algorithm_name
    if not algo_dir.exists():
        return None

    # Only LC_SAC has Lyapunov loss
    if algorithm_name != 'LC_SAC':
        return None

    steps_list = []
    lyap_losses_list = []
    trial_dirs = sorted([d for d in algo_dir.iterdir() if d.is_dir() and 'trial' in d.name])

    for trial_dir in trial_dirs:
        # Look for training_losses JSON files
        loss_files = list(trial_dir.glob(f'training_losses*{trajectory_type}.json'))
        if loss_files:
            try:
                with open(loss_files[0], 'r', encoding='utf-8') as f:
                    training_losses = json.load(f)

                # Extract steps and Lyapunov losses
                steps = [l['step'] for l in training_losses if 'step' in l]
                lyap_losses = [l.get('lyap_loss', 0.0) for l in training_losses if 'step' in l]

                if steps and lyap_losses:
                    steps_list.append(np.array(steps))
                    lyap_losses_list.append(np.array(lyap_losses))
                    print(f"Loaded Lyapunov loss data from {trial_dir.name} ({len(steps)} points)")
            except Exception as e:
                print(f"Warning: Could not load Lyapunov loss from {trial_dir}: {e}")

    if not steps_list:
        return None

    return {
        'steps': steps_list,
        'lyap_losses': lyap_losses_list
    }


def plot_lyapunov_loss_decay(lyap_data, output_dir):
    """
    Plot Lyapunov loss decay across all trials (mean ± std).

    Args:
        lyap_data: Dictionary with 'steps' and 'lyap_losses' lists
        output_dir: Output directory for plots
    """
    if not lyap_data or not lyap_data['steps']:
        print("Warning: No Lyapunov loss data found.")
        return

    fig, axes = plt.subplots(2, 1, figsize=(12, 10))

    steps_list = lyap_data['steps']
    lyap_losses_list = lyap_data['lyap_losses']

    # Plot 1: All trials overlay
    for i, (steps, losses) in enumerate(zip(steps_list, lyap_losses_list)):
        # Preserve all initial 30,000 steps, filter spikes only after that
        initial_threshold = 30000
        mask_initial = steps <= initial_threshold
        mask_after = (steps > initial_threshold) & (losses < 1e2)
        mask = mask_initial | mask_after

        steps_filtered = steps[mask]
        losses_filtered = losses[mask]

        axes[0].plot(steps_filtered, losses_filtered, '--', linewidth=1,
                    alpha=0.4, color='C1', zorder=1)

    # Calculate mean and std across trials
    # Interpolate all trials to a common step grid
    min_step = min(s.min() for s in steps_list)
    max_step = max(s.max() for s in steps_list)
    common_steps = np.linspace(min_step, max_step, 1000)

    # Interpolate each trial to common steps
    interpolated_losses = []
    for steps, losses in zip(steps_list, lyap_losses_list):
        # Preserve all initial 30,000 steps, filter spikes only after that
        initial_threshold = 30000
        mask_initial = steps <= initial_threshold
        mask_after = (steps > initial_threshold) & (losses < 1e2)
        mask = mask_initial | mask_after

        steps_clean = steps[mask]
        losses_clean = losses[mask]

        if len(steps_clean) > 1:
            # Interpolate to common grid
            interp_losses = np.interp(common_steps, steps_clean, losses_clean,
                                     left=losses_clean[0], right=losses_clean[-1])
            interpolated_losses.append(interp_losses)

    if interpolated_losses:
        mean_losses = np.mean(interpolated_losses, axis=0)
        std_losses = np.std(interpolated_losses, axis=0)

        axes[0].plot(common_steps, mean_losses, '-', linewidth=2.5,
                    color='C1', label='LC-SAC (Mean)', zorder=2)
        axes[0].fill_between(common_steps, mean_losses - std_losses,
                            mean_losses + std_losses, alpha=0.3, color='C1', zorder=1)

    axes[0].set_xlabel('Training Steps', fontsize=12)
    axes[0].set_ylabel('Lyapunov Loss', fontsize=12)
    axes[0].set_title('Lyapunov Loss Decay (All Trials)', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_yscale('log')  # Log scale for better visualization

    # Plot 2: Moving average
    if interpolated_losses:
        window_size = 50
        if len(mean_losses) >= window_size:
            # Calculate moving average
            ma_losses = np.convolve(mean_losses, np.ones(window_size)/window_size, mode='valid')
            ma_steps = common_steps[:len(ma_losses)]

            # Calculate std for moving average (approximate)
            ma_std = np.convolve(std_losses, np.ones(window_size)/window_size, mode='valid')

            axes[1].plot(ma_steps, ma_losses, '-', linewidth=2.5,
                        color='C2', label=f'LC-SAC (MA, window={window_size})')
            axes[1].fill_between(ma_steps, ma_losses - ma_std,
                                ma_losses + ma_std, alpha=0.3, color='C2')

            axes[1].set_xlabel('Training Steps', fontsize=12)
            axes[1].set_ylabel('Lyapunov Loss (Moving Average)', fontsize=12)
            axes[1].set_title(f'Lyapunov Loss Moving Average (window={window_size})',
                            fontsize=14, fontweight='bold')
            axes[1].legend(fontsize=11)
            axes[1].grid(True, alpha=0.3)
            axes[1].set_yscale('log')

    plt.tight_layout()

    # Save plot
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_path = output_dir / "lyapunov_loss_decay.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Saved Lyapunov loss decay plot to {plot_path}")
    plt.show()


def plot_trajectory_comparison(baseline_trajectories, lcsac_trajectories, output_dir):
    """
    Plot XZ trajectory comparison between LC-SAC, Baseline SAC, and Reference.

    Args:
        baseline_trajectories: List of trajectory data dictionaries from baseline SAC
        lcsac_trajectories: List of trajectory data dictionaries from LC-SAC
        output_dir: Output directory for plots
    """
    if not baseline_trajectories and not lcsac_trajectories:
        print("Warning: No trajectory data found. Trajectory data must be generated first.")
        print("Trajectory data is saved when evaluate_trajectory() is called during training.")
        return

    fig, axes = plt.subplots(2, 1, figsize=(12, 10))

    # Plot 1: XZ Trajectory (all trials overlay)
    ref_positions = None

    # Plot baseline trajectories
    if baseline_trajectories:
        for i, traj_data in enumerate(baseline_trajectories):
            ref_positions = np.array(traj_data['reference_positions'])
            agent_positions = np.array(traj_data['agent_positions'])

            if i == 0:
                # Plot reference only once
                axes[0].plot(ref_positions[:, 0], ref_positions[:, 1],
                           'b-', linewidth=3, label='Reference', alpha=0.8, zorder=3)
            axes[0].plot(agent_positions[:, 0], agent_positions[:, 1],
                        '--', linewidth=1.5, color='C0', alpha=0.5, zorder=1)

        # Plot mean baseline trajectory if multiple trials
        if len(baseline_trajectories) > 1:
            # Interpolate to common length for averaging
            min_len = min(len(traj['agent_positions']) for traj in baseline_trajectories)
            all_baseline_trajs = []
            for traj in baseline_trajectories:
                agent_pos = np.array(traj['agent_positions'])[:min_len]
                all_baseline_trajs.append(agent_pos)
            baseline_mean = np.mean(all_baseline_trajs, axis=0)
            axes[0].plot(baseline_mean[:, 0], baseline_mean[:, 1],
                        '-', linewidth=2.5, color='C0', label='Baseline SAC (Mean)', zorder=2)

    # Plot LC-SAC trajectories
    if lcsac_trajectories:
        for i, traj_data in enumerate(lcsac_trajectories):
            if ref_positions is None:
                ref_positions = np.array(traj_data['reference_positions'])
            agent_positions = np.array(traj_data['agent_positions'])

            if i == 0 and not baseline_trajectories:
                # Plot reference only once if no baseline
                axes[0].plot(ref_positions[:, 0], ref_positions[:, 1],
                           'b-', linewidth=3, label='Reference', alpha=0.8, zorder=3)
            axes[0].plot(agent_positions[:, 0], agent_positions[:, 1],
                        '--', linewidth=1.5, color='C1', alpha=0.5, zorder=1)

        # Plot mean LC-SAC trajectory if multiple trials
        if len(lcsac_trajectories) > 1:
            # Interpolate to common length for averaging
            min_len = min(len(traj['agent_positions']) for traj in lcsac_trajectories)
            all_lcsac_trajs = []
            for traj in lcsac_trajectories:
                agent_pos = np.array(traj['agent_positions'])[:min_len]
                all_lcsac_trajs.append(agent_pos)
            lcsac_mean = np.mean(all_lcsac_trajs, axis=0)
            axes[0].plot(lcsac_mean[:, 0], lcsac_mean[:, 1],
                        '-', linewidth=2.5, color='C1', label='LC-SAC (Mean)', zorder=2)

    axes[0].set_xlabel('X Position (m)', fontsize=12)
    axes[0].set_ylabel('Z Position (m)', fontsize=12)
    axes[0].set_title('XZ Trajectory Comparison (All Trials)', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=11, loc='best')
    axes[0].grid(True, alpha=0.3)
    axes[0].axis('equal')
    axes[0].set_aspect('equal', adjustable='box')

    # Plot 2: Mean trajectories comparison
    if baseline_trajectories and lcsac_trajectories:
        # Find common minimum length
        min_len_baseline = min(len(traj['agent_positions']) for traj in baseline_trajectories)
        min_len_lcsac = min(len(traj['agent_positions']) for traj in lcsac_trajectories)
        min_len = min(min_len_baseline, min_len_lcsac)

        # Baseline mean
        all_baseline = [np.array(traj['agent_positions'])[:min_len] for traj in baseline_trajectories]
        baseline_mean = np.mean(all_baseline, axis=0)
        baseline_std = np.std(all_baseline, axis=0)

        # LC-SAC mean
        all_lcsac = [np.array(traj['agent_positions'])[:min_len] for traj in lcsac_trajectories]
        lcsac_mean = np.mean(all_lcsac, axis=0)
        lcsac_std = np.std(all_lcsac, axis=0)

        # Reference (use first available)
        ref_pos = np.array(baseline_trajectories[0]['reference_positions'])[:min_len]

        axes[1].plot(ref_pos[:, 0], ref_pos[:, 1], 'b-', linewidth=3, label='Reference', alpha=0.8)
        axes[1].plot(baseline_mean[:, 0], baseline_mean[:, 1], '-', linewidth=2, color='C0', label='Baseline SAC (Mean)')
        axes[1].fill_between(baseline_mean[:, 0], baseline_mean[:, 1] - baseline_std[:, 1],
                            baseline_mean[:, 1] + baseline_std[:, 1], alpha=0.2, color='C0')
        axes[1].plot(lcsac_mean[:, 0], lcsac_mean[:, 1], '-', linewidth=2, color='C1', label='LC-SAC (Mean)')
        axes[1].fill_between(lcsac_mean[:, 0], lcsac_mean[:, 1] - lcsac_std[:, 1],
                            lcsac_mean[:, 1] + lcsac_std[:, 1], alpha=0.2, color='C1')

    elif baseline_trajectories:
        min_len = min(len(traj['agent_positions']) for traj in baseline_trajectories)
        all_baseline = [np.array(traj['agent_positions'])[:min_len] for traj in baseline_trajectories]
        baseline_mean = np.mean(all_baseline, axis=0)
        ref_pos = np.array(baseline_trajectories[0]['reference_positions'])[:min_len]
        axes[1].plot(ref_pos[:, 0], ref_pos[:, 1], 'b-', linewidth=3, label='Reference', alpha=0.8)
        axes[1].plot(baseline_mean[:, 0], baseline_mean[:, 1], '-', linewidth=2, color='C0', label='Baseline SAC (Mean)')

    elif lcsac_trajectories:
        min_len = min(len(traj['agent_positions']) for traj in lcsac_trajectories)
        all_lcsac = [np.array(traj['agent_positions'])[:min_len] for traj in lcsac_trajectories]
        lcsac_mean = np.mean(all_lcsac, axis=0)
        ref_pos = np.array(lcsac_trajectories[0]['reference_positions'])[:min_len]
        axes[1].plot(ref_pos[:, 0], ref_pos[:, 1], 'b-', linewidth=3, label='Reference', alpha=0.8)
        axes[1].plot(lcsac_mean[:, 0], lcsac_mean[:, 1], '-', linewidth=2, color='C1', label='LC-SAC (Mean)')

    axes[1].set_xlabel('X Position (m)', fontsize=12)
    axes[1].set_ylabel('Z Position (m)', fontsize=12)
    axes[1].set_title('XZ Trajectory Comparison (Mean ± Std)', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=11, loc='best')
    axes[1].grid(True, alpha=0.3)
    axes[1].axis('equal')
    axes[1].set_aspect('equal', adjustable='box')

    plt.tight_layout()

    # Save plot
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_path = output_dir / "trajectory_comparison.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Saved trajectory comparison plot to {plot_path}")
    plt.show()


def compute_numerical_metrics(baseline_data, lcsac_data, output_dir):
    """
    Compute and display comprehensive numerical metrics for comparison.

    Args:
        baseline_data: Dictionary with baseline SAC data
        lcsac_data: Dictionary with LC-SAC data
        output_dir: Output directory for saving metrics
    """
    print("\n" + "="*80)
    print("NUMERICAL METRICS COMPARISON")
    print("="*80)

    metrics = {}

    # Process Baseline SAC
    if baseline_data:
        baseline_metrics = {}

        # Episode rewards metrics
        if 'episode_rewards_all' in baseline_data:
            episode_rewards = baseline_data['episode_rewards_all']

            # Final performance (last 10 episodes)
            final_10_rewards = []
            for trial in episode_rewards:
                if len(trial) >= 10:
                    final_10_rewards.append(np.nanmean(trial[-10:]))

            baseline_metrics['episode_final_10_mean'] = np.nanmean(final_10_rewards) if final_10_rewards else None
            baseline_metrics['episode_final_10_std'] = np.nanstd(final_10_rewards) if final_10_rewards else None

            # Maximum episode reward per trial
            max_rewards = [np.nanmax(trial) for trial in episode_rewards if len(trial) > 0]
            baseline_metrics['episode_max_mean'] = np.nanmean(max_rewards) if max_rewards else None
            baseline_metrics['episode_max_std'] = np.nanstd(max_rewards) if max_rewards else None

            # Overall mean and std
            all_rewards = np.concatenate([trial for trial in episode_rewards if len(trial) > 0])
            baseline_metrics['episode_overall_mean'] = np.nanmean(all_rewards) if len(all_rewards) > 0 else None
            baseline_metrics['episode_overall_std'] = np.nanstd(all_rewards) if len(all_rewards) > 0 else None

            # Convergence metric: episodes to reach 90% of max performance
            convergence_episodes = []
            for trial in episode_rewards:
                if len(trial) > 0:
                    max_val = np.nanmax(trial)
                    target = 0.9 * max_val
                    converged_idx = np.where(trial >= target)[0]
                    if len(converged_idx) > 0:
                        convergence_episodes.append(converged_idx[0])
            baseline_metrics['convergence_episodes_mean'] = np.nanmean(convergence_episodes) if convergence_episodes else None
            baseline_metrics['convergence_episodes_std'] = np.nanstd(convergence_episodes) if convergence_episodes else None

        # Evaluation rewards metrics
        if 'eval_rewards_all' in baseline_data:
            eval_rewards = baseline_data['eval_rewards_all']

            # Final evaluation reward
            final_eval_rewards = []
            for trial in eval_rewards:
                if len(trial) > 0:
                    final_eval_rewards.append(trial[-1] if not np.isnan(trial[-1]) else None)
            final_eval_rewards = [r for r in final_eval_rewards if r is not None]
            baseline_metrics['eval_final_mean'] = np.nanmean(final_eval_rewards) if final_eval_rewards else None
            baseline_metrics['eval_final_std'] = np.nanstd(final_eval_rewards) if final_eval_rewards else None

            # Best evaluation reward
            best_eval_rewards = [np.nanmax(trial) for trial in eval_rewards if len(trial) > 0]
            baseline_metrics['eval_best_mean'] = np.nanmean(best_eval_rewards) if best_eval_rewards else None
            baseline_metrics['eval_best_std'] = np.nanstd(best_eval_rewards) if best_eval_rewards else None

        # Summary statistics
        if 'summary' in baseline_data and 'statistics' in baseline_data['summary']:
            stats = baseline_data['summary']['statistics']
            if 'episode_rewards' in stats:
                baseline_metrics['summary_final_10_mean'] = stats['episode_rewards'].get('mean_final_10')
                baseline_metrics['summary_final_10_std'] = stats['episode_rewards'].get('std_final_10')
                baseline_metrics['summary_max_mean'] = stats['episode_rewards'].get('mean_max')
                baseline_metrics['summary_max_std'] = stats['episode_rewards'].get('std_max')
            if 'best_eval_reward' in stats:
                baseline_metrics['summary_eval_best_mean'] = stats['best_eval_reward'].get('mean')
                baseline_metrics['summary_eval_best_std'] = stats['best_eval_reward'].get('std')
                baseline_metrics['summary_eval_best_min'] = stats['best_eval_reward'].get('min')
                baseline_metrics['summary_eval_best_max'] = stats['best_eval_reward'].get('max')

        metrics['Baseline SAC'] = baseline_metrics

    # Process LC-SAC
    if lcsac_data:
        lcsac_metrics = {}

        # Episode rewards metrics
        if 'episode_rewards_all' in lcsac_data:
            episode_rewards = lcsac_data['episode_rewards_all']

            # Final performance (last 10 episodes)
            final_10_rewards = []
            for trial in episode_rewards:
                if len(trial) >= 10:
                    final_10_rewards.append(np.nanmean(trial[-10:]))

            lcsac_metrics['episode_final_10_mean'] = np.nanmean(final_10_rewards) if final_10_rewards else None
            lcsac_metrics['episode_final_10_std'] = np.nanstd(final_10_rewards) if final_10_rewards else None

            # Maximum episode reward per trial
            max_rewards = [np.nanmax(trial) for trial in episode_rewards if len(trial) > 0]
            lcsac_metrics['episode_max_mean'] = np.nanmean(max_rewards) if max_rewards else None
            lcsac_metrics['episode_max_std'] = np.nanstd(max_rewards) if max_rewards else None

            # Overall mean and std
            all_rewards = np.concatenate([trial for trial in episode_rewards if len(trial) > 0])
            lcsac_metrics['episode_overall_mean'] = np.nanmean(all_rewards) if len(all_rewards) > 0 else None
            lcsac_metrics['episode_overall_std'] = np.nanstd(all_rewards) if len(all_rewards) > 0 else None

            # Convergence metric: episodes to reach 90% of max performance
            convergence_episodes = []
            for trial in episode_rewards:
                if len(trial) > 0:
                    max_val = np.nanmax(trial)
                    target = 0.9 * max_val
                    converged_idx = np.where(trial >= target)[0]
                    if len(converged_idx) > 0:
                        convergence_episodes.append(converged_idx[0])
            lcsac_metrics['convergence_episodes_mean'] = np.nanmean(convergence_episodes) if convergence_episodes else None
            lcsac_metrics['convergence_episodes_std'] = np.nanstd(convergence_episodes) if convergence_episodes else None

        # Evaluation rewards metrics
        if 'eval_rewards_all' in lcsac_data:
            eval_rewards = lcsac_data['eval_rewards_all']

            # Final evaluation reward
            final_eval_rewards = []
            for trial in eval_rewards:
                if len(trial) > 0:
                    final_eval_rewards.append(trial[-1] if not np.isnan(trial[-1]) else None)
            final_eval_rewards = [r for r in final_eval_rewards if r is not None]
            lcsac_metrics['eval_final_mean'] = np.nanmean(final_eval_rewards) if final_eval_rewards else None
            lcsac_metrics['eval_final_std'] = np.nanstd(final_eval_rewards) if final_eval_rewards else None

            # Best evaluation reward
            best_eval_rewards = [np.nanmax(trial) for trial in eval_rewards if len(trial) > 0]
            lcsac_metrics['eval_best_mean'] = np.nanmean(best_eval_rewards) if best_eval_rewards else None
            lcsac_metrics['eval_best_std'] = np.nanstd(best_eval_rewards) if best_eval_rewards else None

        # Summary statistics
        if 'summary' in lcsac_data and 'statistics' in lcsac_data['summary']:
            stats = lcsac_data['summary']['statistics']
            if 'episode_rewards' in stats:
                lcsac_metrics['summary_final_10_mean'] = stats['episode_rewards'].get('mean_final_10')
                lcsac_metrics['summary_final_10_std'] = stats['episode_rewards'].get('std_final_10')
                lcsac_metrics['summary_max_mean'] = stats['episode_rewards'].get('mean_max')
                lcsac_metrics['summary_max_std'] = stats['episode_rewards'].get('std_max')
            if 'best_eval_reward' in stats:
                lcsac_metrics['summary_eval_best_mean'] = stats['best_eval_reward'].get('mean')
                lcsac_metrics['summary_eval_best_std'] = stats['best_eval_reward'].get('std')
                lcsac_metrics['summary_eval_best_min'] = stats['best_eval_reward'].get('min')
                lcsac_metrics['summary_eval_best_max'] = stats['best_eval_reward'].get('max')

        metrics['LC-SAC'] = lcsac_metrics

    # Display metrics in a formatted table
    print("\n" + "-"*80)
    print(f"{'Metric':<45} {'Baseline SAC':<20} {'LC-SAC':<20}")
    print("-"*80)

    # Episode Rewards Section
    print("\n📊 EPISODE REWARDS METRICS")
    print("-"*80)

    metric_names = [
        ('Final 10 Episodes (Mean)', 'episode_final_10_mean', 'episode_final_10_std'),
        ('Final 10 Episodes (Std)', None, None),
        ('Max Episode Reward (Mean)', 'episode_max_mean', 'episode_max_std'),
        ('Max Episode Reward (Std)', None, None),
        ('Overall Mean', 'episode_overall_mean', 'episode_overall_std'),
        ('Overall Std', None, None),
        ('Convergence Episodes (Mean)', 'convergence_episodes_mean', 'convergence_episodes_std'),
        ('Convergence Episodes (Std)', None, None),
    ]

    for metric_name, mean_key, std_key in metric_names:
        if mean_key:
            baseline_val = metrics.get('Baseline SAC', {}).get(mean_key)
            lcsac_val = metrics.get('LC-SAC', {}).get(mean_key)

            baseline_str = f"{baseline_val:.2f}" if baseline_val is not None else "N/A"
            lcsac_str = f"{lcsac_val:.2f}" if lcsac_val is not None else "N/A"

            # Highlight better value
            if baseline_val is not None and lcsac_val is not None:
                if metric_name.startswith('Convergence'):
                    # Lower is better for convergence
                    if lcsac_val < baseline_val:
                        lcsac_str = f"✓ {lcsac_str}"
                    elif baseline_val < lcsac_val:
                        baseline_str = f"✓ {baseline_str}"
                else:
                    # Higher is better for rewards
                    if lcsac_val > baseline_val:
                        lcsac_str = f"✓ {lcsac_str}"
                    elif baseline_val > lcsac_val:
                        baseline_str = f"✓ {baseline_str}"

            print(f"{metric_name:<45} {baseline_str:<20} {lcsac_str:<20}")
        elif std_key:
            baseline_val = metrics.get('Baseline SAC', {}).get(std_key)
            lcsac_val = metrics.get('LC-SAC', {}).get(std_key)

            baseline_str = f"  ±{baseline_val:.2f}" if baseline_val is not None else "N/A"
            lcsac_str = f"  ±{lcsac_val:.2f}" if lcsac_val is not None else "N/A"
            print(f"{'  (Std Dev)':<45} {baseline_str:<20} {lcsac_str:<20}")

    # Evaluation Rewards Section
    print("\n🎯 EVALUATION REWARDS METRICS")
    print("-"*80)

    eval_metric_names = [
        ('Best Eval Reward (Mean)', 'eval_best_mean', 'eval_best_std'),
        ('Best Eval Reward (Std)', None, None),
        ('Final Eval Reward (Mean)', 'eval_final_mean', 'eval_final_std'),
        ('Final Eval Reward (Std)', None, None),
    ]

    for metric_name, mean_key, std_key in eval_metric_names:
        if mean_key:
            baseline_val = metrics.get('Baseline SAC', {}).get(mean_key)
            lcsac_val = metrics.get('LC-SAC', {}).get(mean_key)

            baseline_str = f"{baseline_val:.2f}" if baseline_val is not None else "N/A"
            lcsac_str = f"{lcsac_val:.2f}" if lcsac_val is not None else "N/A"

            # Highlight better value
            if baseline_val is not None and lcsac_val is not None:
                if lcsac_val > baseline_val:
                    lcsac_str = f"✓ {lcsac_str}"
                elif baseline_val > lcsac_val:
                    baseline_str = f"✓ {baseline_str}"

            print(f"{metric_name:<45} {baseline_str:<20} {lcsac_str:<20}")
        elif std_key:
            baseline_val = metrics.get('Baseline SAC', {}).get(std_key)
            lcsac_val = metrics.get('LC-SAC', {}).get(std_key)

            baseline_str = f"  ±{baseline_val:.2f}" if baseline_val is not None else "N/A"
            lcsac_str = f"  ±{lcsac_val:.2f}" if lcsac_val is not None else "N/A"
            print(f"{'  (Std Dev)':<45} {baseline_str:<20} {lcsac_str:<20}")

    # Summary from JSON files
    print("\n📋 SUMMARY STATISTICS (from JSON)")
    print("-"*80)

    summary_metrics = [
        ('Summary: Final 10 Mean', 'summary_final_10_mean', 'summary_final_10_std'),
        ('Summary: Final 10 Std', None, None),
        ('Summary: Max Mean', 'summary_max_mean', 'summary_max_std'),
        ('Summary: Max Std', None, None),
        ('Summary: Best Eval Mean', 'summary_eval_best_mean', 'summary_eval_best_std'),
        ('Summary: Best Eval Std', None, None),
    ]

    for metric_name, mean_key, std_key in summary_metrics:
        if mean_key:
            baseline_val = metrics.get('Baseline SAC', {}).get(mean_key)
            lcsac_val = metrics.get('LC-SAC', {}).get(mean_key)

            baseline_str = f"{baseline_val:.2f}" if baseline_val is not None else "N/A"
            lcsac_str = f"{lcsac_val:.2f}" if lcsac_val is not None else "N/A"

            if baseline_val is not None and lcsac_val is not None:
                if lcsac_val > baseline_val:
                    lcsac_str = f"✓ {lcsac_str}"
                elif baseline_val > lcsac_val:
                    baseline_str = f"✓ {baseline_str}"

            print(f"{metric_name:<45} {baseline_str:<20} {lcsac_str:<20}")
        elif std_key:
            baseline_val = metrics.get('Baseline SAC', {}).get(std_key)
            lcsac_val = metrics.get('LC-SAC', {}).get(std_key)

            baseline_str = f"  ±{baseline_val:.2f}" if baseline_val is not None else "N/A"
            lcsac_str = f"  ±{lcsac_val:.2f}" if lcsac_val is not None else "N/A"
            print(f"{'  (Std Dev)':<45} {baseline_str:<20} {lcsac_str:<20}")

    print("\n" + "="*80)
    print("Note: ✓ indicates better performance for that metric")
    print("="*80)

    # Save metrics to JSON file
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "numerical_metrics_comparison.json"
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, default=str)
    print(f"\nSaved numerical metrics to {metrics_path}")


def main():
    """Main execution function."""
    # Configuration
    BASE_DIR = Path("RL_Model/Multiple_Trials")
    OUTPUT_DIR = BASE_DIR / "Comparison_Plots"
    TRAJECTORY_TYPE = "circle"  # Change if using different trajectory type

    print("="*60)
    print("Loading and Plotting Multiple Trials Results")
    print("="*60)

    # Load aggregated data
    print("\nLoading aggregated data...")
    baseline_data = load_aggregated_data(BASE_DIR, "baseline_SAC")
    lcsac_data = load_aggregated_data(BASE_DIR, "LC_SAC")

    # Load trajectory data
    print("\nLoading trajectory data from trials...")
    baseline_trajectories = load_trajectory_data_from_trials(BASE_DIR, "baseline_SAC", TRAJECTORY_TYPE)
    lcsac_trajectories = load_trajectory_data_from_trials(BASE_DIR, "LC_SAC", TRAJECTORY_TYPE)

    # Load Lyapunov loss data (only for LC_SAC)
    print("\nLoading Lyapunov loss data from LC_SAC trials...")
    lyap_data = load_lyapunov_loss_data(BASE_DIR, "LC_SAC", TRAJECTORY_TYPE)

    # Create plots
    print("\n" + "="*60)
    print("Generating Comparison Plots")
    print("="*60)

    # Plot 1: Reward Comparison
    if baseline_data or lcsac_data:
        print("\n1. Plotting reward comparison...")
        plot_reward_comparison(baseline_data, lcsac_data, OUTPUT_DIR)
    else:
        print("\nWarning: No reward data found. Skipping reward comparison plot.")

    # Plot 2: Trajectory Comparison
    if baseline_trajectories or lcsac_trajectories:
        print("\n2. Plotting trajectory comparison...")
        plot_trajectory_comparison(baseline_trajectories, lcsac_trajectories, OUTPUT_DIR)
    else:
        print("\nWarning: No trajectory data found.")
        print("Trajectory data is generated when evaluate_trajectory() is called.")
        print("This typically happens during individual training runs, not in run_multiple_trials.py")
        print("You may need to run trajectory evaluation separately.")

    # Plot 3: Lyapunov Loss Decay
    if lyap_data:
        print("\n3. Plotting Lyapunov loss decay...")
        plot_lyapunov_loss_decay(lyap_data, OUTPUT_DIR)
    else:
        print("\nWarning: No Lyapunov loss data found.")
        print("Lyapunov loss data is saved during LC-SAC training.")

    # Compute and display numerical metrics
    print("\n4. Computing numerical metrics...")
    compute_numerical_metrics(baseline_data, lcsac_data, OUTPUT_DIR)

    print("\n" + "="*60)
    print("Plotting Complete!")
    print(f"Results saved to: {OUTPUT_DIR.absolute()}")
    print("="*60)


if __name__ == "__main__":
    main()
