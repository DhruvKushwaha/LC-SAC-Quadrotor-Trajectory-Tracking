"""
Script to run multiple trials with different seeds for baseline_SAC_2D.py and LC_SAC_2D_discrete.py.
Generates mean-variance plots and saves aggregated training data.
"""

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")  # Use non-interactive backend so plots never block
import matplotlib.pyplot as plt
import copy
import json
import pickle
from pathlib import Path
import yaml
from datetime import datetime
import traceback

# Import necessary modules from both scripts
from baseline_SAC_2D import (
    make_quadrotor_2d_env as make_baseline_env,
    train_sac,
    plot_training_results,
    save_training_data,
    evaluate_trajectory,
    get_traj_suffix
)
from safe_control_gym.controllers.sac.sac_utils import SACAgent, SACBuffer

# Import from LC_SAC script
from LC_SAC_2D_discrete import (
    make_quadrotor_2d_env as make_lcsac_env,
    train_lcsac,
    plot_training_results as plot_lcsac_training_results,
    save_training_data as save_lcsac_training_data,
    evaluate_trajectory as evaluate_lcsac_trajectory,
    plot_lyapunov_loss,
    get_traj_suffix as get_lcsac_traj_suffix
)
from LC_SAC import LCSAC
from Modified_SAC_Buffer import SACBuffer as LCSACBuffer

# Clear cuda cache if using GPU
if torch.cuda.is_available():
    torch.cuda.empty_cache()


def run_baseline_sac_trial(seed, env_config, sac_config, output_dir, trial_num, num_trials):
    """
    Run a single baseline SAC trial with a given seed.

    Args:
        seed: Random seed for this trial
        env_config: Environment configuration
        sac_config: SAC algorithm configuration
        output_dir: Output directory for this trial
        trial_num: Trial number (for logging)
        num_trials: Total number of trials (for logging)

    Returns:
        Dictionary with training results
    """
    print(f"\n{'='*60}")
    print(f"BASELINE SAC - Trial {trial_num}/{num_trials} - Seed: {seed}")
    print(f"{'='*60}")

    # Set seeds
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Override seed in env_config
    trial_env_config = copy.deepcopy(env_config)
    # trial_env_config['seed'] = seed

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Extract config values
    HIDDEN_DIM = sac_config.get("hidden_dim", 128)
    ACTIVATION = sac_config.get("activation", "relu")
    GAMMA = sac_config.get("gamma", 0.99)
    TAU = sac_config.get("tau", 0.005)
    USE_ENTROPY_TUNING = sac_config.get("use_entropy_tuning", True)
    TRAIN_INTERVAL = sac_config.get("train_interval", 1)
    TRAIN_BATCH_SIZE = int(sac_config.get("train_batch_size", 128))
    ACTOR_LR = sac_config.get("actor_lr", 0.0003)
    CRITIC_LR = sac_config.get("critic_lr", 0.0003)
    ENTROPY_LR = sac_config.get("entropy_lr", 0.0003)
    MAX_ENV_STEPS = sac_config.get("max_env_steps", 400000)
    WARM_UP_STEPS = sac_config.get("warm_up_steps", 5000)
    MAX_BUFFER_SIZE = int(sac_config.get("max_buffer_size", 1000000))
    EVAL_BATCH_SIZE = sac_config.get("eval_batch_size", 10)
    LOG_INTERVAL = sac_config.get("log_interval", 100)
    EVAL_INTERVAL = sac_config.get("eval_interval", 4000)
    TRAJECTORY_TYPE = sac_config.get("trajectory_type", "circle")
    INIT_TEMPERATURE = sac_config.get("init_temperature", 0.2)

    try:
        # Initialize environment
        env = make_baseline_env(gui=False, trajectory_type=TRAJECTORY_TYPE, env_config=trial_env_config)

        # Initialize SAC agent
        agent = SACAgent(
            obs_space=env.observation_space,
            act_space=env.action_space,
            hidden_dim=HIDDEN_DIM,
            gamma=GAMMA,
            tau=TAU,
            use_entropy_tuning=USE_ENTROPY_TUNING,
            init_temperature=INIT_TEMPERATURE,
            actor_lr=ACTOR_LR,
            critic_lr=CRITIC_LR,
            entropy_lr=ENTROPY_LR,
            activation=ACTIVATION
        )
        agent.to(device)

        # Fix log_alpha
        log_alpha_value = agent.log_alpha.item() if hasattr(agent.log_alpha, 'item') else float(agent.log_alpha)
        agent.log_alpha = torch.tensor(log_alpha_value, device=device, requires_grad=USE_ENTROPY_TUNING)
        if USE_ENTROPY_TUNING:
            agent.alpha_opt = torch.optim.Adam([agent.log_alpha], lr=ENTROPY_LR)

        # Initialize replay buffer
        replay_buffer = SACBuffer(
            obs_space=env.observation_space,
            act_space=env.action_space,
            max_size=MAX_BUFFER_SIZE,
            batch_size=TRAIN_BATCH_SIZE
        )

        # Train agent
        training_results = train_sac(
            agent=agent,
            env=env,
            replay_buffer=replay_buffer,
            max_steps=MAX_ENV_STEPS,
            warm_up_steps=WARM_UP_STEPS,
            train_interval=TRAIN_INTERVAL,
            train_batch_size=TRAIN_BATCH_SIZE,
            eval_interval=EVAL_INTERVAL if EVAL_INTERVAL > 0 else 0,
            eval_batch_size=EVAL_BATCH_SIZE,
            log_interval=LOG_INTERVAL if LOG_INTERVAL > 0 else 50,
            output_dir=output_dir,
            norm_obs=False,
            device=device,
            trajectory_type=TRAJECTORY_TYPE,
            env_config=trial_env_config,
            sac_config=sac_config
        )

        # Save training data for this trial
        print(f"\nSaving trial {trial_num} training data...")
        plot_training_results(training_results, output_dir, TRAJECTORY_TYPE, trial_env_config)
        save_training_data(
            training_results, output_dir, TRAJECTORY_TYPE, trial_env_config,
            MAX_ENV_STEPS, WARM_UP_STEPS, TRAIN_INTERVAL, TRAIN_BATCH_SIZE,
            HIDDEN_DIM, GAMMA, TAU, ACTOR_LR, CRITIC_LR
        )

        # Save final model
        traj_suffix = get_traj_suffix(trajectory_type=TRAJECTORY_TYPE, env_config=trial_env_config)
        final_model_path = output_dir / f"sac_model_final{traj_suffix}.pth"
        torch.save(agent.state_dict(), str(final_model_path))
        print(f"Saved final model to {final_model_path}")

        # Evaluate trajectory
        print(f"Evaluating trajectory for trial {trial_num}...")
        eval_env = make_baseline_env(
            gui=False,
            override_config={'randomized_init': False},
            trajectory_type=TRAJECTORY_TYPE,
            env_config=trial_env_config
        )
        evaluate_trajectory(
            agent=agent,
            env=eval_env,
            output_dir=output_dir,
            trajectory_type=TRAJECTORY_TYPE,
            env_config=trial_env_config,
            hidden_dim=HIDDEN_DIM,
            gamma=GAMMA,
            tau=TAU,
            actor_lr=ACTOR_LR,
            critic_lr=CRITIC_LR,
            entropy_lr=ENTROPY_LR,
            use_entropy_tuning=USE_ENTROPY_TUNING,
            activation=ACTIVATION,
            device=device
        )
        eval_env.close()

        env.close()

        print(f"✓ Baseline SAC Trial {trial_num} completed successfully")
        return training_results

    except Exception as e:
        print(f"✗ Error in Baseline SAC Trial {trial_num}: {e}")
        traceback.print_exc()
        return None


def run_lcsac_trial(seed, env_config, sac_config, output_dir, trial_num, num_trials,
                    edmd_model, P_lifted, A, B):
    """
    Run a single LC-SAC trial with a given seed.

    Args:
        seed: Random seed for this trial
        env_config: Environment configuration
        sac_config: SAC algorithm configuration
        output_dir: Output directory for this trial
        trial_num: Trial number (for logging)
        num_trials: Total number of trials (for logging)
        edmd_model: EDMD model for Lyapunov constraint
        P_lifted: Lifted P matrix
        A: Lifted A matrix
        B: Lifted B matrix

    Returns:
        Dictionary with training results
    """
    print(f"\n{'='*60}")
    print(f"LC-SAC - Trial {trial_num}/{num_trials} - Seed: {seed}")
    print(f"{'='*60}")

    # Set seeds
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Override seed in env_config
    trial_env_config = copy.deepcopy(env_config)
    #trial_env_config['seed'] = seed

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Extract config values
    HIDDEN_DIM = sac_config.get("hidden_dim", 128)
    GAMMA = sac_config.get("gamma", 0.99)
    TAU = sac_config.get("tau", 0.005)
    INIT_TEMPERATURE = sac_config.get("init_temperature", 0.2)
    USE_ENTROPY_TUNING = sac_config.get("use_entropy_tuning", False)
    TRAIN_INTERVAL = sac_config.get("train_interval", 1)
    TRAIN_BATCH_SIZE = int(sac_config.get("train_batch_size", 128))
    ACTOR_LR = sac_config.get("actor_lr", 0.0001)
    CRITIC_LR = sac_config.get("critic_lr", 0.0003)
    ENTROPY_LR = sac_config.get("entropy_lr", 0.0003)
    MAX_ENV_STEPS = sac_config.get("max_env_steps", 400000)
    WARM_UP_STEPS = sac_config.get("warm_up_steps", 5000)
    MAX_BUFFER_SIZE = int(sac_config.get("max_buffer_size", 1000000))
    EVAL_BATCH_SIZE = sac_config.get("eval_batch_size", 10)
    LOG_INTERVAL = sac_config.get("log_interval", 100)
    EVAL_INTERVAL = sac_config.get("eval_interval", 4000)
    TRAJECTORY_TYPE = sac_config.get("trajectory_type", "circle")
    QUADTYPE = "quadrotor_2D"

    try:
        # Initialize environment
        env = make_lcsac_env(gui=False, trajectory_type=TRAJECTORY_TYPE, env_config=trial_env_config)

        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        action_range = [env.action_space.low.copy(), env.action_space.high.copy()]

        # Initialize LC-SAC agent
        agent = LCSAC(
            state_dim=state_dim,
            action_dim=action_dim,
            action_range=action_range,
            hidden_dim=HIDDEN_DIM,
            device=device,
            edmd_model=edmd_model,
            P_lifted=P_lifted,
            A=A,
            B=B,
            gamma=GAMMA,
            init_temperature=INIT_TEMPERATURE,
            tau=TAU,
            actor_lr=ACTOR_LR,
            critic_lr=CRITIC_LR,
            entropy_lr=ENTROPY_LR,
            use_entropy_tuning=USE_ENTROPY_TUNING,
            quadtype=QUADTYPE
        )

        # Initialize replay buffer
        replay_buffer = LCSACBuffer(
            obs_space=env.observation_space,
            act_space=env.action_space,
            max_size=MAX_BUFFER_SIZE,
            batch_size=TRAIN_BATCH_SIZE
        )

        # Train agent
        training_results = train_lcsac(
            agent=agent,
            env=env,
            edmd_model=edmd_model,
            replay_buffer=replay_buffer,
            max_steps=MAX_ENV_STEPS,
            warm_up_steps=WARM_UP_STEPS,
            train_interval=TRAIN_INTERVAL,
            train_batch_size=TRAIN_BATCH_SIZE,
            eval_interval=EVAL_INTERVAL if EVAL_INTERVAL > 0 else 0,
            eval_batch_size=EVAL_BATCH_SIZE,
            log_interval=LOG_INTERVAL if LOG_INTERVAL > 0 else 1,
            output_dir=output_dir,
            obs_normalizer=None,
            norm_obs=False,
            trajectory_type=TRAJECTORY_TYPE,
            env_config=trial_env_config
        )

        # Save training data for this trial
        print(f"\nSaving trial {trial_num} training data...")
        plot_lcsac_training_results(training_results, output_dir, TRAJECTORY_TYPE, trial_env_config)
        save_lcsac_training_data(
            training_results, output_dir, TRAJECTORY_TYPE, trial_env_config,
            MAX_ENV_STEPS, WARM_UP_STEPS, TRAIN_INTERVAL, TRAIN_BATCH_SIZE,
            HIDDEN_DIM, GAMMA, TAU, INIT_TEMPERATURE, ACTOR_LR, CRITIC_LR
        )

        # Plot Lyapunov loss
        plot_lyapunov_loss(training_results, output_dir, TRAJECTORY_TYPE, trial_env_config)

        # Save final model
        traj_suffix = get_lcsac_traj_suffix(trajectory_type=TRAJECTORY_TYPE, env_config=trial_env_config)
        final_model_path = output_dir / f"lcsac_model_final{traj_suffix}.pth"
        agent.save(str(final_model_path))
        print(f"Saved final model to {final_model_path}")

        # Evaluate trajectory
        print(f"Evaluating trajectory for trial {trial_num}...")
        eval_env = make_lcsac_env(
            gui=False,
            override_config={'randomized_init': False},
            trajectory_type=TRAJECTORY_TYPE,
            env_config=trial_env_config
        )
        evaluate_lcsac_trajectory(
            agent=agent,
            env=eval_env,
            output_dir=output_dir,
            trajectory_type=TRAJECTORY_TYPE,
            env_config=trial_env_config,
            hidden_dim=HIDDEN_DIM,
            gamma=GAMMA,
            tau=TAU,
            init_temperature=INIT_TEMPERATURE,
            actor_lr=ACTOR_LR,
            critic_lr=CRITIC_LR,
            entropy_lr=ENTROPY_LR,
            use_entropy_tuning=USE_ENTROPY_TUNING,
            edmd_model=edmd_model,
            P_lifted=P_lifted,
            A=A,
            B=B
        )
        eval_env.close()

        env.close()

        print(f"✓ LC-SAC Trial {trial_num} completed successfully")
        return training_results

    except Exception as e:
        print(f"✗ Error in LC-SAC Trial {trial_num}: {e}")
        traceback.print_exc()
        return None


def align_episode_data(all_episode_data):
    """
    Align episode data from multiple trials to the same episode indices.
    Uses the maximum episode count and pads shorter trials with NaN.

    Args:
        all_episode_data: List of arrays, each containing episode data from one trial

    Returns:
        Aligned array of shape (num_trials, max_episodes)
    """
    if not all_episode_data:
        return np.array([])

    max_len = max(len(data) for data in all_episode_data)
    aligned = []

    for data in all_episode_data:
        if len(data) < max_len:
            padded = np.full(max_len, np.nan)
            padded[:len(data)] = data
            aligned.append(padded)
        else:
            aligned.append(data)

    return np.array(aligned)


def align_step_data(all_step_data):
    """
    Align step-based data (like losses, eval rewards) from multiple trials.
    Interpolates to common step values.

    Args:
        all_step_data: List of lists of dicts, each dict has 'step' and value keys

    Returns:
        Tuple of (steps, aligned_values) where aligned_values is (num_trials, num_steps)
    """
    if not all_step_data or not all_step_data[0]:
        return np.array([]), np.array([])

    # Get all unique step values
    all_steps = set()
    for trial_data in all_step_data:
        if trial_data:
            all_steps.update([d['step'] for d in trial_data])

    if not all_steps:
        return np.array([]), np.array([])

    steps = sorted(all_steps)
    aligned_values = []

    for trial_data in all_step_data:
        if not trial_data:
            aligned_values.append(np.full(len(steps), np.nan))
            continue

        # Create mapping from step to value
        step_to_value = {d['step']: d for d in trial_data}

        # Interpolate values for each step
        values = []
        for step in steps:
            if step in step_to_value:
                # Get the value (try common keys)
                value_dict = step_to_value[step]
                if 'reward' in value_dict:
                    values.append(value_dict['reward'])
                elif 'critic_loss' in value_dict:
                    values.append(value_dict['critic_loss'])
                else:
                    values.append(np.nan)
            else:
                values.append(np.nan)

        aligned_values.append(values)

    return np.array(steps), np.array(aligned_values)


def plot_mean_variance(results_dict, output_dir, algorithm_name):
    """
    Create mean-variance plots for training results.

    Args:
        results_dict: Dictionary with keys 'episode_rewards', 'episode_lengths',
                     'eval_rewards', 'training_losses' containing lists of trial results
        output_dir: Output directory for plots
        algorithm_name: Name of algorithm (for titles and filenames)
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(16, 12))

    # 1. Episode Rewards
    if results_dict.get('episode_rewards'):
        ax1 = plt.subplot(3, 2, 1)
        episode_rewards_aligned = align_episode_data(results_dict['episode_rewards'])

        if episode_rewards_aligned.size > 0:
            mean_rewards = np.nanmean(episode_rewards_aligned, axis=0)
            std_rewards = np.nanstd(episode_rewards_aligned, axis=0)
            episodes = np.arange(len(mean_rewards))

            ax1.plot(episodes, mean_rewards, label='Mean', linewidth=2)
            ax1.fill_between(episodes, mean_rewards - std_rewards,
                           mean_rewards + std_rewards, alpha=0.3, label='±1 Std')
            ax1.set_xlabel('Episode')
            ax1.set_ylabel('Reward')
            ax1.set_title(f'{algorithm_name} - Episode Rewards (Mean ± Std)')
            ax1.legend()
            ax1.grid(True)

    # 2. Moving Average of Episode Rewards
    if results_dict.get('episode_rewards'):
        ax2 = plt.subplot(3, 2, 2)
        episode_rewards_aligned = align_episode_data(results_dict['episode_rewards'])

        if episode_rewards_aligned.size > 0:
            window = min(100, episode_rewards_aligned.shape[1] // 10)
            if window > 1:
                # Compute moving average for each trial, then average
                ma_trials = []
                for trial in episode_rewards_aligned:
                    valid = ~np.isnan(trial)
                    if np.sum(valid) >= window:
                        ma = np.convolve(trial[valid], np.ones(window)/window, mode='valid')
                        ma_padded = np.full(len(trial), np.nan)
                        ma_padded[window-1:window-1+len(ma)] = ma
                        ma_trials.append(ma_padded)

                if ma_trials:
                    ma_aligned = np.array(ma_trials)
                    mean_ma = np.nanmean(ma_aligned, axis=0)
                    std_ma = np.nanstd(ma_aligned, axis=0)
                    episodes = np.arange(len(mean_ma))

                    ax2.plot(episodes, mean_ma, label='Mean', linewidth=2)
                    ax2.fill_between(episodes, mean_ma - std_ma, mean_ma + std_ma,
                                   alpha=0.3, label='±1 Std')
                    ax2.set_xlabel('Episode')
                    ax2.set_ylabel('Moving Average Reward')
                    ax2.set_title(f'{algorithm_name} - Moving Average Rewards (window={window})')
                    ax2.legend()
                    ax2.grid(True)

    # 3. Episode Lengths
    if results_dict.get('episode_lengths'):
        ax3 = plt.subplot(3, 2, 3)
        episode_lengths_aligned = align_episode_data(results_dict['episode_lengths'])

        if episode_lengths_aligned.size > 0:
            mean_lengths = np.nanmean(episode_lengths_aligned, axis=0)
            std_lengths = np.nanstd(episode_lengths_aligned, axis=0)
            episodes = np.arange(len(mean_lengths))

            ax3.plot(episodes, mean_lengths, label='Mean', linewidth=2)
            ax3.fill_between(episodes, mean_lengths - std_lengths,
                           mean_lengths + std_lengths, alpha=0.3, label='±1 Std')
            ax3.set_xlabel('Episode')
            ax3.set_ylabel('Length')
            ax3.set_title(f'{algorithm_name} - Episode Lengths (Mean ± Std)')
            ax3.legend()
            ax3.grid(True)

    # 4. Evaluation Rewards
    if results_dict.get('eval_rewards'):
        ax4 = plt.subplot(3, 2, 4)
        steps, eval_rewards_aligned = align_step_data(results_dict['eval_rewards'])

        if eval_rewards_aligned.size > 0:
            mean_eval = np.nanmean(eval_rewards_aligned, axis=0)
            std_eval = np.nanstd(eval_rewards_aligned, axis=0)

            ax4.plot(steps, mean_eval, 'o-', label='Mean', linewidth=2, markersize=4)
            ax4.fill_between(steps, mean_eval - std_eval, mean_eval + std_eval,
                            alpha=0.3, label='±1 Std')
            ax4.set_xlabel('Training Steps')
            ax4.set_ylabel('Average Reward')
            ax4.set_title(f'{algorithm_name} - Evaluation Rewards (Mean ± Std)')
            ax4.legend()
            ax4.grid(True)

    # 5. Training Losses - Critic Loss
    if results_dict.get('training_losses'):
        ax5 = plt.subplot(3, 2, 5)

        # Extract critic losses
        critic_losses = []
        for trial_losses in results_dict['training_losses']:
            if trial_losses:
                trial_critic = [{'step': l.get('step', 0),
                                'critic_loss': l.get('critic_loss', l.get('critic_1_loss', np.nan))}
                               for l in trial_losses]
                critic_losses.append(trial_critic)

        if critic_losses:
            steps, critic_aligned = align_step_data(critic_losses)

            if critic_aligned.size > 0:
                mean_critic = np.nanmean(critic_aligned, axis=0)
                std_critic = np.nanstd(critic_aligned, axis=0)

                ax5.plot(steps, mean_critic, label='Mean Critic Loss', linewidth=2)
                ax5.fill_between(steps, mean_critic - std_critic, mean_critic + std_critic,
                                alpha=0.3, label='±1 Std')
                ax5.set_xlabel('Training Steps')
                ax5.set_ylabel('Loss')
                ax5.set_title(f'{algorithm_name} - Critic Loss (Mean ± Std)')
                ax5.legend()
                ax5.grid(True)

    # 6. Training Losses - Actor/Policy Loss
    if results_dict.get('training_losses'):
        ax6 = plt.subplot(3, 2, 6)

        # Extract actor/policy losses
        actor_losses = []
        for trial_losses in results_dict['training_losses']:
            if trial_losses:
                trial_actor = [{'step': l.get('step', 0),
                               'actor_loss': l.get('actor_loss', l.get('policy_loss', np.nan))}
                              for l in trial_losses]
                actor_losses.append(trial_actor)

        if actor_losses:
            steps, actor_aligned = align_step_data(actor_losses)

            if actor_aligned.size > 0:
                mean_actor = np.nanmean(actor_aligned, axis=0)
                std_actor = np.nanstd(actor_aligned, axis=0)

                ax6.plot(steps, mean_actor, label='Mean Actor Loss', linewidth=2, color='C2')
                ax6.fill_between(steps, mean_actor - std_actor, mean_actor + std_actor,
                                alpha=0.3, label='±1 Std', color='C2')
                ax6.set_xlabel('Training Steps')
                ax6.set_ylabel('Loss')
                ax6.set_title(f'{algorithm_name} - Actor/Policy Loss (Mean ± Std)')
                ax6.legend()
                ax6.grid(True)

    plt.tight_layout()

    # Save plot
    plot_path = output_dir / f"{algorithm_name.lower().replace(' ', '_')}_mean_variance_plots.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved mean-variance plots to {plot_path}")

    # Close figure to avoid blocking or accumulating open figures
    plt.close()


def save_aggregated_data(results_dict, output_dir, algorithm_name, seeds):
    """
    Save aggregated training data from all trials.

    Args:
        results_dict: Dictionary with aggregated results
        output_dir: Output directory
        algorithm_name: Name of algorithm
        seeds: List of seeds used
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save episode rewards
    if results_dict.get('episode_rewards'):
        episode_rewards_aligned = align_episode_data(results_dict['episode_rewards'])
        np.save(output_dir / f"{algorithm_name.lower().replace(' ', '_')}_episode_rewards_all_trials.npy",
                episode_rewards_aligned)

        mean_rewards = np.nanmean(episode_rewards_aligned, axis=0)
        std_rewards = np.nanstd(episode_rewards_aligned, axis=0)
        np.save(output_dir / f"{algorithm_name.lower().replace(' ', '_')}_episode_rewards_mean.npy",
                mean_rewards)
        np.save(output_dir / f"{algorithm_name.lower().replace(' ', '_')}_episode_rewards_std.npy",
                std_rewards)

    # Save episode lengths
    if results_dict.get('episode_lengths'):
        episode_lengths_aligned = align_episode_data(results_dict['episode_lengths'])
        np.save(output_dir / f"{algorithm_name.lower().replace(' ', '_')}_episode_lengths_all_trials.npy",
                episode_lengths_aligned)

    # Save evaluation rewards
    if results_dict.get('eval_rewards'):
        steps, eval_rewards_aligned = align_step_data(results_dict['eval_rewards'])
        if eval_rewards_aligned.size > 0:
            np.save(output_dir / f"{algorithm_name.lower().replace(' ', '_')}_eval_rewards_all_trials.npy",
                    eval_rewards_aligned)
            np.save(output_dir / f"{algorithm_name.lower().replace(' ', '_')}_eval_rewards_steps.npy",
                    steps)

    # Save summary statistics
    summary = {
        'algorithm': algorithm_name,
        'num_trials': len(seeds),
        'seeds': seeds,
        'timestamp': datetime.now().isoformat(),
        'statistics': {}
    }

    if results_dict.get('episode_rewards'):
        episode_rewards_aligned = align_episode_data(results_dict['episode_rewards'])
        if episode_rewards_aligned.size > 0:
            summary['statistics']['episode_rewards'] = {
                'mean_final_10': float(np.nanmean([np.nanmean(trial[-10:])
                                                   for trial in episode_rewards_aligned
                                                   if len(trial) >= 10])),
                'std_final_10': float(np.nanstd([np.nanmean(trial[-10:])
                                                for trial in episode_rewards_aligned
                                                if len(trial) >= 10])),
                'mean_max': float(np.nanmean([np.nanmax(trial) for trial in episode_rewards_aligned])),
                'std_max': float(np.nanstd([np.nanmax(trial) for trial in episode_rewards_aligned])),
            }

    if results_dict.get('best_eval_rewards'):
        summary['statistics']['best_eval_reward'] = {
            'mean': float(np.mean(results_dict['best_eval_rewards'])),
            'std': float(np.std(results_dict['best_eval_rewards'])),
            'min': float(np.min(results_dict['best_eval_rewards'])),
            'max': float(np.max(results_dict['best_eval_rewards']))
        }

    summary_path = output_dir / f"{algorithm_name.lower().replace(' ', '_')}_summary.json"
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

    print(f"Saved aggregated data summary to {summary_path}")


def main():
    """Main execution function."""
    # Configuration
    NUM_TRIALS = 5  # Number of trials to run
    SEEDS = list(range(1, 1 + NUM_TRIALS))  # Seeds for each trial

    # Output directories
    BASE_OUTPUT_DIR = Path("RL_Model/Multiple_Trials")
    BASELINE_OUTPUT_DIR = BASE_OUTPUT_DIR / "baseline_SAC"
    LCSAC_OUTPUT_DIR = BASE_OUTPUT_DIR / "LC_SAC"

    BASE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    BASELINE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LCSAC_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Output directory: {BASE_OUTPUT_DIR.absolute()}")
    print(f"Running {NUM_TRIALS} trials with seeds: {SEEDS}")

    # Load environment configuration
    ENV_CONFIG_PATH = Path("./Params/Quadrotor_2D/quadrotor_2D_track.yaml")
    assert ENV_CONFIG_PATH.exists(), f"Missing {ENV_CONFIG_PATH}"

    with open(ENV_CONFIG_PATH, "r", encoding='utf-8') as f:
        env_config = yaml.safe_load(f)["task_config"]

    print("Loaded environment config from quadrotor_2D_track.yaml")

    # Load SAC algorithm configuration
    SAC_CONFIG_PATH = Path("./Params/Quadrotor_2D/sac_quadrotor_2D.yaml")
    assert SAC_CONFIG_PATH.exists(), f"Missing {SAC_CONFIG_PATH}"

    with open(SAC_CONFIG_PATH, "r", encoding='utf-8') as f:
        sac_config_base = yaml.safe_load(f)

    with open(SAC_CONFIG_PATH, "r", encoding='utf-8') as f:
        sac_config_override = yaml.safe_load(f).get("algo_config", {})

    # Merge configs
    sac_config = copy.deepcopy(sac_config_base)
    sac_config.update(sac_config_override)

    # Override config settings
    sac_config['use_entropy_tuning'] = True
    sac_config['norm_obs'] = False
    sac_config['trajectory_type'] = 'circle'  # Can be changed

    print("Loaded SAC algorithm config")

    # Load EDMD model and LQR matrices for LC-SAC
    SAVE_DATA_DIR = Path("Saved_data")
    edmd_model = None
    P_lifted = None
    A = None
    B = None

    if SAVE_DATA_DIR.exists():
        EDMD_MODEL_PATH = SAVE_DATA_DIR / 'edmd_model_2D.pkl'
        RICCATI_PATH = SAVE_DATA_DIR / 'lqr_matrices_2D.npz'

        if EDMD_MODEL_PATH.exists():
            with open(EDMD_MODEL_PATH, 'rb') as f:
                edmd_model = pickle.load(f)
            print(f"Loaded EDMD model from {EDMD_MODEL_PATH}")

        if RICCATI_PATH.exists():
            mats = np.load(RICCATI_PATH)
            P_lifted = mats['P']
            A = mats['A_lifted']
            B = mats['B_lifted']
            print(f"Loaded LQR matrices from {RICCATI_PATH}")

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ========== Run Baseline SAC Trials ==========
    print(f"\n{'#'*60}")
    print("RUNNING BASELINE SAC TRIALS")
    print(f"{'#'*60}")

    baseline_results = {
        'episode_rewards': [],
        'episode_lengths': [],
        'eval_rewards': [],
        'training_losses': [],
        'best_eval_rewards': []
    }

    for i, seed in enumerate(SEEDS, 1):
        trial_output_dir = BASELINE_OUTPUT_DIR / f"trial_{i}_seed_{seed}"
        trial_output_dir.mkdir(parents=True, exist_ok=True)

        result = run_baseline_sac_trial(
            seed=seed,
            env_config=env_config,
            sac_config=sac_config,
            output_dir=trial_output_dir,
            trial_num=i,
            num_trials=NUM_TRIALS
        )

        if result:
            baseline_results['episode_rewards'].append(np.array(result['episode_rewards']))
            baseline_results['episode_lengths'].append(np.array(result['episode_lengths']))
            baseline_results['eval_rewards'].append(result['eval_rewards'])
            baseline_results['training_losses'].append(result['training_losses'])
            baseline_results['best_eval_rewards'].append(result['best_eval_reward'])

    # Clear CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ========== Run LC-SAC Trials ==========
    print(f"\n{'#'*60}")
    print("RUNNING LC-SAC TRIALS")
    print(f"{'#'*60}")

    lcsac_results = {
        'episode_rewards': [],
        'episode_lengths': [],
        'eval_rewards': [],
        'training_losses': [],
        'best_eval_rewards': []
    }

    for i, seed in enumerate(SEEDS, 1):
        trial_output_dir = LCSAC_OUTPUT_DIR / f"trial_{i}_seed_{seed}"
        trial_output_dir.mkdir(parents=True, exist_ok=True)

        result = run_lcsac_trial(
            seed=seed,
            env_config=env_config,
            sac_config=sac_config,
            output_dir=trial_output_dir,
            trial_num=i,
            num_trials=NUM_TRIALS,
            edmd_model=edmd_model,
            P_lifted=P_lifted,
            A=A,
            B=B
        )

        if result:
            lcsac_results['episode_rewards'].append(np.array(result['episode_rewards']))
            lcsac_results['episode_lengths'].append(np.array(result['episode_lengths']))
            lcsac_results['eval_rewards'].append(result['eval_rewards'])
            lcsac_results['training_losses'].append(result['training_losses'])
            lcsac_results['best_eval_rewards'].append(result['best_eval_reward'])

    # ========== Generate Plots and Save Data ==========
    print(f"\n{'#'*60}")
    print("GENERATING PLOTS AND SAVING DATA")
    print(f"{'#'*60}")

    # Filter out None results
    baseline_successful_seeds = [SEEDS[i] for i, r in enumerate(baseline_results['episode_rewards']) if r is not None and len(r) > 0]
    lcsac_successful_seeds = [SEEDS[i] for i, r in enumerate(lcsac_results['episode_rewards']) if r is not None and len(r) > 0]

    # Plot and save Baseline SAC results
    if baseline_results['episode_rewards']:
        # Filter out None results
        baseline_results_clean = {
            'episode_rewards': [r for r in baseline_results['episode_rewards'] if r is not None and len(r) > 0],
            'episode_lengths': [r for r in baseline_results['episode_lengths'] if r is not None and len(r) > 0],
            'eval_rewards': [r for r in baseline_results['eval_rewards'] if r is not None],
            'training_losses': [r for r in baseline_results['training_losses'] if r is not None],
            'best_eval_rewards': [r for r in baseline_results['best_eval_rewards'] if r is not None and not np.isinf(r)]
        }

        if baseline_results_clean['episode_rewards']:
            print(f"\nBaseline SAC: {len(baseline_results_clean['episode_rewards'])} successful trials")
            plot_mean_variance(baseline_results_clean, BASELINE_OUTPUT_DIR, "Baseline SAC")
            save_aggregated_data(baseline_results_clean, BASELINE_OUTPUT_DIR, "Baseline SAC", baseline_successful_seeds)
        else:
            print("\nWarning: No successful Baseline SAC trials to plot")

    # Plot and save LC-SAC results
    if lcsac_results['episode_rewards']:
        # Filter out None results
        lcsac_results_clean = {
            'episode_rewards': [r for r in lcsac_results['episode_rewards'] if r is not None and len(r) > 0],
            'episode_lengths': [r for r in lcsac_results['episode_lengths'] if r is not None and len(r) > 0],
            'eval_rewards': [r for r in lcsac_results['eval_rewards'] if r is not None],
            'training_losses': [r for r in lcsac_results['training_losses'] if r is not None],
            'best_eval_rewards': [r for r in lcsac_results['best_eval_rewards'] if r is not None and not np.isinf(r)]
        }

        if lcsac_results_clean['episode_rewards']:
            print(f"\nLC-SAC: {len(lcsac_results_clean['episode_rewards'])} successful trials")
            plot_mean_variance(lcsac_results_clean, LCSAC_OUTPUT_DIR, "LC-SAC")
            save_aggregated_data(lcsac_results_clean, LCSAC_OUTPUT_DIR, "LC-SAC", lcsac_successful_seeds)
        else:
            print("\nWarning: No successful LC-SAC trials to plot")

    # ========== Load Trajectory Data for Comparison ==========
    print("\nLoading trajectory data for comparison...")
    TRAJECTORY_TYPE_COMPARE = sac_config.get("trajectory_type", "circle")
    baseline_trajectories = []
    lcsac_trajectories = []

    # Load baseline trajectories
    for i, seed in enumerate(SEEDS, 1):
        trial_dir = BASELINE_OUTPUT_DIR / f"trial_{i}_seed_{seed}"
        traj_files = list(trial_dir.glob(f'trajectory_data*{TRAJECTORY_TYPE_COMPARE}.json'))
        if traj_files:
            try:
                with open(traj_files[0], 'r', encoding='utf-8') as f:
                    baseline_trajectories.append(json.load(f))
            except Exception as e:
                print(f"Warning: Could not load trajectory from {trial_dir}: {e}")

    # Load LC-SAC trajectories
    for i, seed in enumerate(SEEDS, 1):
        trial_dir = LCSAC_OUTPUT_DIR / f"trial_{i}_seed_{seed}"
        traj_files = list(trial_dir.glob(f'trajectory_data*{TRAJECTORY_TYPE_COMPARE}.json'))
        if traj_files:
            try:
                with open(traj_files[0], 'r', encoding='utf-8') as f:
                    lcsac_trajectories.append(json.load(f))
            except Exception as e:
                print(f"Warning: Could not load trajectory from {trial_dir}: {e}")

    print(f"Loaded {len(baseline_trajectories)} baseline trajectories and {len(lcsac_trajectories)} LC-SAC trajectories")

    # ========== Comparison Plot ==========
    baseline_has_data = baseline_results.get('episode_rewards') and any(r is not None and len(r) > 0 for r in baseline_results['episode_rewards'])
    lcsac_has_data = lcsac_results.get('episode_rewards') and any(r is not None and len(r) > 0 for r in lcsac_results['episode_rewards'])

    # Plot 1: Average Rewards Comparison
    if baseline_has_data and lcsac_has_data:
        print("\nGenerating comparison plot...")

        _, axes = plt.subplots(2, 2, figsize=(16, 12))

        # Episode Rewards Comparison
        baseline_ep_rewards = align_episode_data([r for r in baseline_results['episode_rewards'] if r is not None and len(r) > 0])
        lcsac_ep_rewards = align_episode_data([r for r in lcsac_results['episode_rewards'] if r is not None and len(r) > 0])

        if baseline_ep_rewards.size > 0:
            baseline_mean = np.nanmean(baseline_ep_rewards, axis=0)
            baseline_std = np.nanstd(baseline_ep_rewards, axis=0)
            episodes = np.arange(len(baseline_mean))

            axes[0, 0].plot(episodes, baseline_mean, label='Baseline SAC', linewidth=2, color='C0')
            axes[0, 0].fill_between(episodes, baseline_mean - baseline_std,
                                  baseline_mean + baseline_std, alpha=0.3, color='C0')

        if lcsac_ep_rewards.size > 0:
            lcsac_mean = np.nanmean(lcsac_ep_rewards, axis=0)
            lcsac_std = np.nanstd(lcsac_ep_rewards, axis=0)
            episodes_lcsac = np.arange(len(lcsac_mean))

            axes[0, 0].plot(episodes_lcsac, lcsac_mean, label='LC-SAC', linewidth=2, color='C1')
            axes[0, 0].fill_between(episodes_lcsac, lcsac_mean - lcsac_std,
                                  lcsac_mean + lcsac_std, alpha=0.3, color='C1')

        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Reward')
        axes[0, 0].set_title('Episode Rewards Comparison (Mean ± Std)')
        axes[0, 0].legend()
        axes[0, 0].grid(True)

        # Evaluation Rewards Comparison
        baseline_eval_clean = [r for r in baseline_results.get('eval_rewards', []) if r is not None]
        lcsac_eval_clean = [r for r in lcsac_results.get('eval_rewards', []) if r is not None]

        if baseline_eval_clean and lcsac_eval_clean:
            baseline_steps, baseline_eval = align_step_data(baseline_eval_clean)
            lcsac_steps, lcsac_eval = align_step_data(lcsac_eval_clean)

            if baseline_eval.size > 0:
                baseline_eval_mean = np.nanmean(baseline_eval, axis=0)
                baseline_eval_std = np.nanstd(baseline_eval, axis=0)
                axes[0, 1].plot(baseline_steps, baseline_eval_mean, 'o-',
                              label='Baseline SAC', linewidth=2, markersize=4, color='C0')
                axes[0, 1].fill_between(baseline_steps, baseline_eval_mean - baseline_eval_std,
                                      baseline_eval_mean + baseline_eval_std, alpha=0.3, color='C0')

            if lcsac_eval.size > 0:
                lcsac_eval_mean = np.nanmean(lcsac_eval, axis=0)
                lcsac_eval_std = np.nanstd(lcsac_eval, axis=0)
                axes[0, 1].plot(lcsac_steps, lcsac_eval_mean, 'o-',
                              label='LC-SAC', linewidth=2, markersize=4, color='C1')
                axes[0, 1].fill_between(lcsac_steps, lcsac_eval_mean - lcsac_eval_std,
                                      lcsac_eval_mean + lcsac_eval_std, alpha=0.3, color='C1')

            axes[0, 1].set_xlabel('Training Steps')
            axes[0, 1].set_ylabel('Average Reward')
            axes[0, 1].set_title('Evaluation Rewards Comparison (Mean ± Std)')
            axes[0, 1].legend()
            axes[0, 1].grid(True)

        # Best Eval Rewards Comparison
        baseline_best_clean = [r for r in baseline_results.get('best_eval_rewards', []) if r is not None and not np.isinf(r)]
        lcsac_best_clean = [r for r in lcsac_results.get('best_eval_rewards', []) if r is not None and not np.isinf(r)]

        if baseline_best_clean and lcsac_best_clean:
            axes[1, 0].bar(['Baseline SAC', 'LC-SAC'],
                          [np.mean(baseline_best_clean),
                           np.mean(lcsac_best_clean)],
                          yerr=[np.std(baseline_best_clean),
                                np.std(lcsac_best_clean)],
                          capsize=10, alpha=0.7, color=['C0', 'C1'])
            axes[1, 0].set_ylabel('Best Evaluation Reward')
            axes[1, 0].set_title('Best Evaluation Reward Comparison (Mean ± Std)')
            axes[1, 0].grid(True, axis='y')

        # Final 10 Episode Average Comparison
        if baseline_ep_rewards.size > 0 and lcsac_ep_rewards.size > 0:
            # Calculate final 10 episodes average for each trial
            # Use only valid (non-NaN) values from the last 10 episodes
            baseline_final_10 = []
            for trial in baseline_ep_rewards:
                # Get valid (non-NaN) episodes
                valid_trial = trial[~np.isnan(trial)]
                if len(valid_trial) >= 10:
                    # Take last 10 valid episodes
                    final_10_mean = np.mean(valid_trial[-10:])
                    if not np.isnan(final_10_mean):
                        baseline_final_10.append(final_10_mean)

            lcsac_final_10 = []
            for trial in lcsac_ep_rewards:
                # Get valid (non-NaN) episodes
                valid_trial = trial[~np.isnan(trial)]
                if len(valid_trial) >= 10:
                    # Take last 10 valid episodes
                    final_10_mean = np.mean(valid_trial[-10:])
                    if not np.isnan(final_10_mean):
                        lcsac_final_10.append(final_10_mean)

            # Only plot if we have valid data for both algorithms
            if baseline_final_10 and lcsac_final_10:
                baseline_mean_val = np.mean(baseline_final_10)
                baseline_std_val = np.std(baseline_final_10)
                lcsac_mean_val = np.mean(lcsac_final_10)
                lcsac_std_val = np.std(lcsac_final_10)

                axes[1, 1].bar(['Baseline SAC', 'LC-SAC'],
                          [baseline_mean_val, lcsac_mean_val],
                          yerr=[baseline_std_val, lcsac_std_val],
                          capsize=10, alpha=0.7, color=['C0', 'C1'])
                axes[1, 1].set_ylabel('Average Reward')
                axes[1, 1].set_title('Final 10 Episodes Average Reward (Mean ± Std)')
                axes[1, 1].grid(True, axis='y')
            elif baseline_final_10:
                # Only baseline has data
                baseline_mean_val = np.mean(baseline_final_10)
                baseline_std_val = np.std(baseline_final_10)
                axes[1, 1].bar(['Baseline SAC'],
                          [baseline_mean_val],
                          yerr=[baseline_std_val],
                          capsize=10, alpha=0.7, color=['C0'])
                axes[1, 1].set_ylabel('Average Reward')
                axes[1, 1].set_title('Final 10 Episodes Average Reward (Mean ± Std)')
                axes[1, 1].grid(True, axis='y')
            elif lcsac_final_10:
                # Only LC-SAC has data
                lcsac_mean_val = np.mean(lcsac_final_10)
                lcsac_std_val = np.std(lcsac_final_10)
                axes[1, 1].bar(['LC-SAC'],
                          [lcsac_mean_val],
                          yerr=[lcsac_std_val],
                          capsize=10, alpha=0.7, color=['C1'])
                axes[1, 1].set_ylabel('Average Reward')
                axes[1, 1].set_title('Final 10 Episodes Average Reward (Mean ± Std)')
                axes[1, 1].grid(True, axis='y')

        plt.tight_layout()

        comparison_plot_path = BASE_OUTPUT_DIR / "reward_comparison.png"
        plt.savefig(comparison_plot_path, dpi=300, bbox_inches='tight')
        print(f"Saved reward comparison plot to {comparison_plot_path}")

        # Close comparison figure
        plt.close()

    # Plot 2: Trajectory Comparison
    if baseline_trajectories or lcsac_trajectories:
        print("\nGenerating trajectory comparison plot...")

        fig, axes = plt.subplots(2, 1, figsize=(12, 10))

        # Plot 1: All trials overlay
        ref_positions = None

        # Plot baseline trajectories
        if baseline_trajectories:
            # for i, traj_data in enumerate(baseline_trajectories):
            #     ref_positions = np.array(traj_data['reference_positions'])
            #     agent_positions = np.array(traj_data['agent_positions'])

            #     if i == 0:
            #         axes[0].plot(ref_positions[:, 0], ref_positions[:, 1],
            #                    'b-', linewidth=3, label='Reference', alpha=0.8, zorder=3)
            #     axes[0].plot(agent_positions[:, 0], agent_positions[:, 1],
            #                 '--', linewidth=1.5, color='C0', alpha=0.5, zorder=1)

            # Plot mean baseline trajectory if multiple trials
            if len(baseline_trajectories) > 1:
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
            # for i, traj_data in enumerate(lcsac_trajectories):
            #     if ref_positions is None:
            #         ref_positions = np.array(traj_data['reference_positions'])
            #     agent_positions = np.array(traj_data['agent_positions'])

            #     if i == 0 and not baseline_trajectories:
            #         axes[0].plot(ref_positions[:, 0], ref_positions[:, 1],
            #                    'b-', linewidth=3, label='Reference', alpha=0.8, zorder=3)
            #     axes[0].plot(agent_positions[:, 0], agent_positions[:, 1],
            #                 '--', linewidth=1.5, color='C1', alpha=0.5, zorder=1)

            # Plot mean LC-SAC trajectory if multiple trials
            if len(lcsac_trajectories) > 1:
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

            # Reference
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

        trajectory_plot_path = BASE_OUTPUT_DIR / "trajectory_comparison.png"
        plt.savefig(trajectory_plot_path, dpi=300, bbox_inches='tight')
        print(f"Saved trajectory comparison plot to {trajectory_plot_path}")

        # Close trajectory figure
        plt.close()
    else:
        print("\nWarning: No trajectory data found for comparison.")
        print("Trajectory data is generated during trial evaluation.")

    print(f"\n{'='*60}")
    print("ALL TRIALS COMPLETED!")
    print(f"{'='*60}")
    print(f"Results saved to: {BASE_OUTPUT_DIR.absolute()}")


if __name__ == "__main__":
    main()
