"""
PID Controller and Data Collection for EDMD (2D Quadrotor)

This script:
1. Sets up a 2D quadrotor environment with PID control
2. Runs a single episode and plots the trajectory
3. Collects data for EDMD (Extended Dynamic Mode Decomposition) training
"""

import numpy as np
import yaml
from pathlib import Path
import matplotlib.pyplot as plt

from safe_control_gym.envs.gym_pybullet_drones.quadrotor import Quadrotor
from safe_control_gym.envs.benchmark_env import Task, Cost
from safe_control_gym.envs.gym_pybullet_drones.quadrotor_utils import QuadType
from safe_control_gym.controllers.pid.pid import PID as PIDController


def make_quadrotor_2d_env(
    traj_type="circle",
    gui=False,
    normalized_action_space=False,
    env_config=None
):
    """
    Create a Quadrotor 2D environment.

    Args:
        traj_type: Trajectory type ('circle', 'figure8', 'square')
        gui: Whether to show PyBullet GUI
        normalized_action_space: Whether to use normalized action space
        env_config: Environment configuration dictionary
    """
    # Start with the loaded config
    config = env_config.copy()

    # Override trajectory type if specified
    if traj_type is not None:
        if traj_type not in ['circle', 'figure8', 'square']:
            raise ValueError(f"traj_type must be one of ['circle', 'figure8', 'square'], got '{traj_type}'")
        if 'task_info' not in config:
            config['task_info'] = {}
        config['task_info']['trajectory_type'] = traj_type

    # Convert quad_type integer to QuadType enum
    quad_type = QuadType(config["quad_type"])

    # Convert task string to Task enum
    task = Task(config["task"])

    # Convert cost string to Cost enum
    cost = Cost(config["cost"])

    # Extract init_state and inertial_prop
    init_state = config.get("init_state")
    inertial_prop = config.get("inertial_prop")

    # Build environment kwargs from config
    env_kwargs = {
        "seed": config.get("seed"),
        "quad_type": quad_type,
        "task": task,
        "physics": config.get("physics"),
        "task_info": config.get("task_info"),
        "cost": cost,
        "gui": gui,
        "normalized_rl_action_space": normalized_action_space,
        "pyb_freq": config.get("pyb_freq"),
        "ctrl_freq": config.get("ctrl_freq"),
        "episode_len_sec": config.get("episode_len_sec"),
        "randomized_init": config.get("randomized_init", True),
        "randomized_inertial_prop": config.get("randomized_inertial_prop", False),
        "init_state_randomization_info": config.get("init_state_randomization_info"),
        "obs_goal_horizon": config.get("obs_goal_horizon"),
        "rew_state_weight": config.get("rew_state_weight"),
        "rew_act_weight": config.get("rew_act_weight"),
        "rew_exponential": config.get("rew_exponential", True),
        "done_on_out_of_bound": config.get("done_on_out_of_bound", True),
        "info_mse_metric_state_weight": config.get("info_mse_metric_state_weight"),
        "constraints": config.get("constraints"),
        "done_on_violation": config.get("done_on_violation", False),
    }

    # Add optional parameters only if they exist
    if init_state is not None:
        env_kwargs["init_state"] = init_state
    if inertial_prop is not None:
        env_kwargs["inertial_prop"] = inertial_prop

    env = Quadrotor(**env_kwargs)
    return env


def run_single_episode(env, pid_controller, step_limit=500):
    """
    Run a single episode with PID control.

    Args:
        env: Quadrotor environment
        pid_controller: PID controller instance
        step_limit: Maximum number of steps

    Returns:
        states: Array of states
        actions: Array of actions
        rewards: Array of rewards
        x_ref_full: Reference trajectory
    """
    obs, info = env.reset()
    x_ref_full = info.get("x_reference", None)
    pid_controller.reset_before_run(obs, info)

    states, actions, rewards = [], [], []
    episode_reward = 0.0

    for t in range(step_limit):
        action = pid_controller.select_action(obs, info)

        # Check for NaN or invalid actions
        if np.any(np.isnan(action)) or np.any(np.isinf(action)):
            print(f"Warning: Invalid action at step {t}: {action}")
            action = np.nan_to_num(action, nan=0.0, posinf=env.action_space.high[0], neginf=env.action_space.low[0])
            action = np.clip(action, env.action_space.low, env.action_space.high)

        next_obs, reward, done, info = env.step(action)

        # Check for NaN reward
        if np.isnan(reward) or np.isinf(reward):
            if t < 10 or t % 50 == 0:  # Print first 10 and then every 50th
                print(f"Warning: Invalid reward at step {t}: reward={reward}")
                print(f"  Action: {action}")
                if hasattr(env, 'state'):
                    print(f"  State: {env.state}")
                    if np.any(np.isnan(env.state)) or np.any(np.isinf(env.state)):
                        print(f"  State contains NaN/Inf!")
                if hasattr(env, 'ctrl_step_counter') and hasattr(env, 'X_GOAL'):
                    print(f"  Step counter: {env.ctrl_step_counter}, X_GOAL shape: {env.X_GOAL.shape}")
            reward = 0.0  # Replace NaN reward with 0

        states.append(obs)
        actions.append(action)
        rewards.append(reward)

        obs = next_obs
        # Only add valid rewards to episode_reward (NaN/inf already replaced with 0 above)
        # Double-check to ensure reward is valid before adding
        if not (np.isnan(reward) or np.isinf(reward)):
            episode_reward += reward
        else:
            # This shouldn't happen if the check above worked, but just in case
            episode_reward += 0.0

        if done:
            print(f"Episode terminated at step {t}: done={done}")
            if 'out_of_bounds' in info:
                print(f"  Out of bounds: {info.get('out_of_bounds', False)}")
            if 'goal_reached' in info:
                print(f"  Goal reached: {info.get('goal_reached', False)}")
            break

    states = np.array(states)
    actions = np.array(actions)
    rewards = np.array(rewards)

    # Check for NaN in rewards array and fix
    if np.any(np.isnan(rewards)) or np.any(np.isinf(rewards)):
        nan_count = np.sum(np.isnan(rewards)) + np.sum(np.isinf(rewards))
        print(f"\nWarning: Found {nan_count} NaN/inf rewards in array")
        rewards = np.nan_to_num(rewards, nan=0.0, posinf=0.0, neginf=0.0)
        episode_reward = np.sum(rewards)
        print(f"Corrected episode reward: {episode_reward:.2f}")

    print("\nEpisode reward:", episode_reward)
    print("Steps:", len(states))
    print("State shape:", states.shape)
    print("Action shape:", actions.shape)
    print(f"Reward stats: min={np.min(rewards):.4f}, max={np.max(rewards):.4f}, mean={np.mean(rewards):.4f}")

    return states, actions, rewards, x_ref_full


def plot_trajectory(states, x_ref_full, traj_type="figure8"):
    """
    Plot the trajectory comparison between reference and actual.

    Args:
        states: Array of actual states
        x_ref_full: Reference trajectory
        traj_type: Type of trajectory (for title)
    """
    # Plot trajectory (2D quadrotor moves in XZ plane)
    if x_ref_full is not None:
        # For 2D: state is [x, x_dot, z, z_dot, theta, theta_dot]
        # Extract positions: x (index 0) and z (index 2)
        ref_pos = x_ref_full[:, [0, 2]]
    else:
        ref_pos = None

    # Extract actual positions from states
    # States are observations which may be 24D, but actual state is first 6D
    if states.shape[1] > 6:
        actual_pos = states[:, [0, 2]]  # Extract x and z from first 6 dimensions
    else:
        actual_pos = states[:, [0, 2]]  # x and z positions

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)

    if ref_pos is not None:
        ax.plot(ref_pos[:, 0], ref_pos[:, 1],
                'r--', linewidth=2, label='Reference Trajectory')

    ax.plot(actual_pos[:, 0], actual_pos[:, 1],
            'b-', linewidth=2, label='PID Quadrotor Trajectory')

    ax.set_title(f"PID: Reference vs Quadrotor Trajectory (XZ plane) - {traj_type} Trajectory")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Z (m)")
    ax.legend()
    ax.grid(True)
    ax.set_aspect('equal', adjustable='box')
    plt.show()


def collect_edmd_data(env_config, pid_cfg, trajectories=['figure8', 'circle', 'square'],
                      num_episodes=20, step_limit=500):
    """
    Collect data for EDMD training.

    Args:
        env_config: Environment configuration dictionary
        pid_cfg: PID controller configuration dictionary
        trajectories: List of trajectory types to collect data for
        num_episodes: Number of episodes per trajectory
        step_limit: Maximum steps per episode

    Returns:
        Dictionary containing collected data arrays
    """
    # Storage for EDMD (X, U, X_next)
    data_X = []
    data_U = []
    data_X_prime = []
    data_tracking_error = []
    data_tracking_error_next = []

    # Collect data for each trajectory
    for traj in trajectories:
        env = make_quadrotor_2d_env(gui=False, traj_type=traj, env_config=env_config)

        # Create PID controller for this environment
        def env_func():
            return env

        pid_controller_traj = PIDController(
            env_func=env_func,
            g=pid_cfg["g"],
            kf=pid_cfg["KF"],
            km=pid_cfg["KM"],
            p_coeff_for=np.array(pid_cfg["P_COEFF_FOR"]),
            i_coeff_for=np.array(pid_cfg["I_COEFF_FOR"]),
            d_coeff_for=np.array(pid_cfg["D_COEFF_FOR"]),
            p_coeff_tor=np.array(pid_cfg["P_COEFF_TOR"]),
            i_coeff_tor=np.array(pid_cfg["I_COEFF_TOR"]),
            d_coeff_tor=np.array(pid_cfg["D_COEFF_TOR"]),
            pwm2rpm_scale=pid_cfg["PWM2RPM_SCALE"],
            pwm2rpm_const=pid_cfg["PWM2RPM_CONST"],
            min_pwm=pid_cfg["MIN_PWM"],
            max_pwm=pid_cfg["MAX_PWM"],
        )

        # Run the environment for specified number of episodes
        for episode in range(num_episodes):
            obs, info = env.reset()
            x_ref_full = info.get("x_reference", None)
            pid_controller_traj.reset_before_run(obs, info)

            # Run the environment for step limit
            for step in range(step_limit):
                # Get the current state
                # For 2D, extract first 6 dimensions (actual state) from observation
                if obs.shape[0] > 6:
                    X = obs[:6]  # Extract state portion
                else:
                    X = obs

                # Select action using PID controller
                action = pid_controller_traj.select_action(obs, info)
                # Add noise to the action
                action += np.random.normal(0, 0.001, action.shape)
                # Clip action to valid range
                action_low = env.action_space.low
                action_high = env.action_space.high
                noisy_action = np.clip(action, action_low, action_high)

                # Apply action to the environment
                next_obs, reward, done, info = env.step(noisy_action)

                # Extract next state
                if next_obs.shape[0] > 6:
                    X_next = next_obs[:6]  # Extract state portion
                else:
                    X_next = next_obs

                # Store the data
                data_X.append(X)
                data_U.append(noisy_action)
                data_X_prime.append(X_next)

                # Store current and next tracking errors for all states
                # Get the reference state at the current step
                current_step = info.get('current_step', 0)
                # Ensure we don't go out of bounds
                if current_step >= x_ref_full.shape[0]:
                    current_step = x_ref_full.shape[0] - 1

                x_ref_current = x_ref_full[current_step]
                # Get next reference state (handle boundary)
                next_step = min(current_step + 1, x_ref_full.shape[0] - 1)
                x_ref_next = x_ref_full[next_step]

                # Calculate tracking error for each state dimension (not norm)
                tracking_error = X - x_ref_current
                tracking_error_next = X_next - x_ref_next

                data_tracking_error.append(tracking_error)
                data_tracking_error_next.append(tracking_error_next)

                # Update the current state
                obs = next_obs

                # Break if done
                if done:
                    break

            # Episode completed (either done or reached step_limit)
            if episode % 5 == 0:
                print(f"Completed episode {episode+1}/{num_episodes} for trajectory {traj}")

        # Close the environment for this trajectory
        env.close()
        print(f"Completed data collection for trajectory: {traj}")

    print("\nData collection completed for all trajectories.")

    # Convert data to numpy arrays
    data_X = np.array(data_X)
    data_U = np.array(data_U)
    data_X_prime = np.array(data_X_prime)
    data_tracking_error = np.array(data_tracking_error)
    data_tracking_error_next = np.array(data_tracking_error_next)

    return {
        'X': data_X,
        'U': data_U,
        'X_prime': data_X_prime,
        'tracking_error': data_tracking_error,
        'tracking_error_next': data_tracking_error_next
    }


def main():
    """Main execution function."""
    # Load PID configuration
    PID_CONFIG_PATH = Path("./Params/Controllers/pid.yaml")
    assert PID_CONFIG_PATH.exists(), f"Missing {PID_CONFIG_PATH}"

    with open(PID_CONFIG_PATH, "r") as f:
        pid_cfg = yaml.safe_load(f)

    print("Loaded PID config from pid.yaml:")
    print(pid_cfg)

    # Load environment configuration from YAML (2D config)
    ENV_CONFIG_PATH = Path("./Params/Quadrotor_2D/PID/quadrotor_2D_track.yaml")
    assert ENV_CONFIG_PATH.exists(), f"Missing {ENV_CONFIG_PATH}"

    with open(ENV_CONFIG_PATH, "r") as f:
        env_config = yaml.safe_load(f)["task_config"]

    print("\nLoaded environment config from quadrotor_2D_track.yaml")

    # ============================================================================
    # Part 1: Run a single episode and plot trajectory
    # ============================================================================
    print("\n" + "="*70)
    print("Part 1: Running single episode with PID control")
    print("="*70)

    # Select the trajectory type
    traj = "figure8"
    env = make_quadrotor_2d_env(gui=False, traj_type=traj, env_config=env_config)

    def env_func():
        return env

    # Create PID controller (it will reset the environment internally during initialization)
    pid_controller = PIDController(
        env_func=env_func,
        g=pid_cfg["g"],
        kf=pid_cfg["KF"],
        km=pid_cfg["KM"],
        p_coeff_for=np.array(pid_cfg["P_COEFF_FOR"]),
        i_coeff_for=np.array(pid_cfg["I_COEFF_FOR"]),
        d_coeff_for=np.array(pid_cfg["D_COEFF_FOR"]),
        p_coeff_tor=np.array(pid_cfg["P_COEFF_TOR"]),
        i_coeff_tor=np.array(pid_cfg["I_COEFF_TOR"]),
        d_coeff_tor=np.array(pid_cfg["D_COEFF_TOR"]),
        pwm2rpm_scale=pid_cfg["PWM2RPM_SCALE"],
        pwm2rpm_const=pid_cfg["PWM2RPM_CONST"],
        min_pwm=pid_cfg["MIN_PWM"],
        max_pwm=pid_cfg["MAX_PWM"],
    )

    print("\nPID controller created.")

    # Get initial observation and info after PID controller initialization
    obs, info = env.reset()
    print("\nObs shape:", obs.shape)
    print("Action space:", env.action_space)

    # Run single episode
    step_limit = 500
    states, actions, rewards, x_ref_full = run_single_episode(
        env, pid_controller, step_limit
    )

    # Plot trajectory
    print("\nPlotting trajectory...")
    plot_trajectory(states, x_ref_full, traj)

    # ============================================================================
    # Part 2: Collect data for EDMD
    # ============================================================================
    print("\n" + "="*70)
    print("Part 2: Collecting data for EDMD training")
    print("="*70)

    TRAJECTORIES = ['figure8', 'circle', 'square']
    edmd_data = collect_edmd_data(
        env_config=env_config,
        pid_cfg=pid_cfg,
        trajectories=TRAJECTORIES,
        num_episodes=20,
        step_limit=step_limit
    )

    # Create save_data directory
    SAVE_DATA_DIR = Path("Saved_data")
    SAVE_DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\nSave data directory: {SAVE_DATA_DIR.absolute()}")

    # Save the data for EDMD in save_data folder
    data_file_path = SAVE_DATA_DIR / 'data_EDMD_2D.npz'
    np.savez(data_file_path,
             X=edmd_data['X'],
             U=edmd_data['U'],
             X_prime=edmd_data['X_prime'],
             tracking_error=edmd_data['tracking_error'],
             tracking_error_next=edmd_data['tracking_error_next'])

    # Check shape and type of data
    print(f"Data X shape: {edmd_data['X'].shape}, type: {edmd_data['X'].dtype}")
    print(f"Data U shape: {edmd_data['U'].shape}, type: {edmd_data['U'].dtype}")
    print(f"Data X_prime shape: {edmd_data['X_prime'].shape}, type: {edmd_data['X_prime'].dtype}")
    print(f"Data tracking_error shape: {edmd_data['tracking_error'].shape}, type: {edmd_data['tracking_error'].dtype}")
    print(f"Data tracking_error_next shape: {edmd_data['tracking_error_next'].shape}, type: {edmd_data['tracking_error_next'].dtype}")
    print(f"\nSaved data to {data_file_path}")

    # Close the environment
    env.close()
    print("\nScript completed successfully!")


if __name__ == "__main__":
    main()
