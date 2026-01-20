# Lyapunov-Constrained SAC for 2D Quadrotor Control

This repository implements and compares **Baseline SAC (Soft Actor-Critic)** and **LC-SAC (Lyapunov-Constrained SAC)** for 2D quadrotor trajectory tracking. The LC-SAC algorithm incorporates Lyapunov stability constraints using Extended Dynamic Mode Decomposition (EDMD) to ensure safety during learning.

## Features

- **Baseline SAC**: Standard Soft Actor-Critic implementation for quadrotor control
- **LC-SAC**: Lyapunov-constrained SAC with stability guarantees
- **EDMD Model**: Extended Dynamic Mode Decomposition for system identification
- **Multiple Trials Support**: Automated multi-trial experiments with statistical analysis
- **Comprehensive Visualization**: Reward comparisons, trajectory tracking, and Lyapunov loss analysis

## Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (optional, but recommended for faster training)

### Dependencies

Install the required Python packages:

```bash
pip install torch torchvision
pip install numpy scipy matplotlib
pip install pyyaml
pip install scikit-learn
pip install pykoopman
pip install control  # python-control
pip install gymnasium
```

### Safe Control Gym

This project uses the `safe-control-gym` library for the quadrotor environment and SAC implementation. Install it from source:

```bash
git clone https://github.com/utiasDSL/safe-control-gym.git
cd safe-control-gym
pip install -e .
```

### Project Setup

1. Clone this repository:
```bash
git clone <repository-url>
cd Multiple_trialV2
```

2. Ensure all dependencies are installed (see above)

3. The project structure should look like:
```
Multiple_trialV2/
├── Params/
│   └── Quadrotor_2D/
│       ├── quadrotor_2D_track.yaml
│       └── sac_quadrotor_2D.yaml
├── Saved_data/          # Will contain EDMD models and data
├── RL_Model/           # Will contain training results
├── baseline_SAC_2D.py
├── LC_SAC_2D_discrete.py
├── LC_SAC.py
├── EDMD_2D_discrete.py
├── run_multiple_trials.py
└── plot_multiple_trials_results.py
```

## Usage

### 1. Training EDMD Model (Required for LC-SAC)

Before training LC-SAC, you need to train an EDMD model for system identification:

```bash
python EDMD_2D_discrete.py
```

This will:
- Load tracking error data from PID controller (stored in `Saved_data/data_EDMD_2D.npz`)
- Train an EDMD model using Radial Basis Functions (RBF)
- Compute LQR gains using the Riccati equation
- Save the model to `Saved_data/edmd_model_2D.pkl` and LQR matrices to `Saved_data/lqr_matrices_2D.npz`

**Note**: You need to have collected tracking error data using a PID controller first. The data should be saved as `Saved_data/data_EDMD_2D.npz` with keys:
- `tracking_error`: Shape (N, 6) - tracking errors for all states
- `tracking_error_next`: Shape (N, 6) - next tracking errors
- `U`: Shape (N, 2) - control inputs

### 2. Training Individual Models

#### Baseline SAC
```bash
python baseline_SAC_2D.py
```

#### LC-SAC
```bash
python LC_SAC_2D_discrete.py
```

Both scripts will:
- Train the respective agent on the 2D quadrotor environment
- Save training results, models, and plots to `RL_Model/`
- Evaluate the trained agent on the specified trajectory (default: circle)

### 3. Running Multiple Trials

To run multiple trials with different random seeds for statistical analysis:

```bash
python run_multiple_trials.py
```

This script will:
- Run 5 trials each for Baseline SAC and LC-SAC (configurable in the script)
- Generate mean-variance plots for each algorithm
- Create comparison plots showing:
  - Episode rewards comparison
  - Evaluation rewards comparison
  - Best evaluation reward comparison
  - Final 10 episodes average reward
  - Trajectory tracking comparison
- Save aggregated data to `RL_Model/Multiple_Trials/`

### 4. Plotting Multiple Trials Results

To visualize and analyze results from multiple trials:

```bash
python plot_multiple_trials_results.py
```

This will:
- Load aggregated data from `RL_Model/Multiple_Trials/`
- Generate comprehensive comparison plots:
  - Episode rewards comparison (Mean ± Std)
  - Evaluation rewards comparison
  - Best evaluation reward bar chart
  - Final 10 episodes average reward
  - Trajectory comparison (XZ plane)
  - Lyapunov loss decay (LC-SAC only)
- Compute and display numerical metrics comparison
- Save plots to `RL_Model/Multiple_Trials/Comparison_Plots/`

## Configuration

### Environment Configuration

Edit `Params/Quadrotor_2D/quadrotor_2D_track.yaml` to configure:
- Quadrotor physical parameters
- Task configuration (trajectory type, etc.)
- Episode settings

### Algorithm Configuration

Edit `Params/Quadrotor_2D/sac_quadrotor_2D.yaml` to configure:
- Network architecture (hidden dimensions, activation)
- Learning rates (actor, critic, entropy)
- Training hyperparameters (gamma, tau, batch size, etc.)
- Training duration (max_env_steps)

### Running Different Trajectories

To change the trajectory type, modify the `trajectory_type` parameter in:
- `run_multiple_trials.py` (line 756): `sac_config['trajectory_type'] = 'circle'`

Supported trajectory types: `'circle'`, `'figure8'`, `'square'`.

## Results

### Performance Comparison

The LC-SAC algorithm demonstrates improved safety and stability compared to Baseline SAC while maintaining competitive reward performance. Results from multiple trials show:

- **Convergence**: LC-SAC achieves stable convergence with Lyapunov constraints
- **Safety**: Lyapunov loss decays over training, ensuring stability
- **Performance**: Both algorithms achieve similar peak rewards, with LC-SAC providing better stability guarantees

### Example Results Structure

After running multiple trials, the results are organized as:

```
RL_Model/Multiple_Trials/
├── baseline_SAC/
│   ├── baseline_sac_episode_rewards_all_trials.npy
│   ├── baseline_sac_episode_rewards_mean.npy
│   ├── baseline_sac_eval_rewards_all_trials.npy
│   ├── baseline_sac_summary.json
│   └── trial_1_seed_1/
│       └── ...
├── LC_SAC/
│   └── ...
└── Comparison_Plots/
    ├── reward_comparison.png
    ├── trajectory_comparison.png
    ├── lyapunov_loss_decay.png
    └── numerical_metrics_comparison.json
```

## Project Structure

```
Multiple_trialV2/
├── baseline_SAC_2D.py              # Baseline SAC training script
├── LC_SAC_2D_discrete.py           # LC-SAC training script
├── LC_SAC.py                        # LC-SAC agent implementation
├── Modified_SAC_Buffer.py          # Modified replay buffer for LC-SAC
├── EDMD_2D_discrete.py             # EDMD model training
├── PID_controller_2D.py            # PID controller for data collection
├── run_multiple_trials.py          # Multi-trial experiment runner
├── plot_multiple_trials_results.py # Results visualization
├── Params/                          # Configuration files
│   └── Quadrotor_2D/
├── Saved_data/                      # EDMD models and collected data
└── RL_Model/                        # Training results and models
```

## Citations

If you use this code in your research, please cite the following:

### Safe Control Gym
```bibtex
@article{safe_control_gym,
  title={safe-control-gym: A Unified Benchmark Suite for Safe Learning-based Control and Reinforcement Learning},
  author={Yuan, Jingyun and Carrillo, Luis and Leung, Chi Hay and Abbeel, Pieter},
  journal={arXiv preprint arXiv:2109.12325},
  year={2021}
}
```

### PyKoopman
```bibtex
@software{pykoopman,
  author = {E. Kaiser and J. N. Kutz and S. L. Brunton},
  title = {PyKoopman: A Python Package for Data-Driven Approximation of the Koopman Operator},
  year = {2022},
  url = {https://github.com/dynamicslab/pykoopman}
}
```

### SAC Algorithm
```bibtex
@inproceedings{haarnoja2018soft,
  title={Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor},
  author={Haarnoja, Tuomas and Zhou, Aurick and Abbeel, Pieter and Levine, Sergey},
  booktitle={International conference on machine learning},
  year={2018},
  organization={PMLR}
}
```

### EDMD
```bibtex
@article{williams2015data,
  title={A data-driven approximation of the Koopman operator: Extending dynamic mode decomposition},
  author={Williams, Matthew O and Kevrekidis, Ioannis G and Rowley, Clarence W},
  journal={Journal of Nonlinear Science},
  volume={25},
  number={6},
  pages={1307--1346},
  year={2015},
  publisher={Springer}
}
```

## Authors

- Dhruv Kushwaha

## Acknowledgments

- Built on top of the [safe-control-gym](https://github.com/utiasDSL/safe-control-gym) framework
- Uses [PyKoopman](https://github.com/dynamicslab/pykoopman) for EDMD implementation
