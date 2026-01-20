"""
EDMD (Extended Dynamic Mode Decomposition) Training for 2D Quadrotor

This script:
1. Loads tracking error data collected from PID controller
2. Trains an EDMD model using Radial Basis Functions (RBF)
3. Computes discrete-time LQR gains using the Riccati equation
4. Saves the trained model and LQR matrices for use in LC-SAC training
"""

import numpy as np
import pykoopman as pk
from pykoopman.observables import RadialBasisFunction
from pykoopman.regression import EDMDc
from sklearn.cluster import KMeans
from scipy.linalg import solve_discrete_are
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import control as ctrl


def load_data(data_file):
    """
    Load tracking error data from saved file.

    Args:
        data_file: Path to .npz file containing tracking error data

    Returns:
        Dictionary with tracking_error, tracking_error_next, and U
    """
    print("--- Loading Data ---")
    data = np.load(data_file)

    tracking_error = data["tracking_error"]          # (N, 6) - tracking error for all states
    tracking_error_next = data["tracking_error_next"]  # (N, 6) - next tracking error for all states
    U = data["U"]          # (N, 2)

    print(f"Loaded tracking_error shape: {tracking_error.shape}")
    print(f"Loaded tracking_error_next shape: {tracking_error_next.shape}")
    print(f"Loaded U shape: {U.shape}")

    return {
        'tracking_error': tracking_error,
        'tracking_error_next': tracking_error_next,
        'U': U
    }


def train_edmd_model(X, X_next, U, n_rbf_centers=3, rbf_width=0.25,
                     regularization=1e-5, dt=1/50.0):
    """
    Train EDMD model on tracking error dynamics.

    Args:
        X: Current tracking error (N, 6)
        X_next: Next tracking error (N, 6)
        U: Control inputs (N, 2)
        n_rbf_centers: Number of RBF centers
        rbf_width: Width parameter for RBF
        regularization: Tikhonov regularization
        dt: Control frequency (time step)

    Returns:
        Trained Koopman model and RBF observables
    """
    print("--- Training Koopman Operator on Tracking Error ---")

    # Configure EDMDc regressor
    regressor = EDMDc()

    # KMeans centers in tracking error space (not state space)
    kmeans = KMeans(n_clusters=n_rbf_centers, random_state=42, n_init=12)
    kmeans.fit(X)  # X is now tracking_error
    centers = kmeans.cluster_centers_          # (n_centers, n_features)

    # PyKoopman RBF expects (n_features, n_centers)
    centers_pk = centers.T                     # (6, n_centers)

    RBF = pk.observables.RadialBasisFunction(
        rbf_type="thinplate",
        n_centers=centers_pk.shape[1],
        centers=centers_pk,
        kernel_width=rbf_width,
        polyharmonic_coeff=1.0,
        include_state=True,
    )

    print(f"Tracking error (X): {X.shape}, Tracking error next (X_next): {X_next.shape}, "
          f"U: {U.shape}, centers: {centers_pk.shape}")

    # Create and train Koopman model
    model = pk.Koopman(observables=RBF, regressor=regressor, quiet=True)

    # Training on tracking error dynamics: tracking_error_next = f(tracking_error, u)
    model.fit(X, y=X_next, u=U, dt=dt)

    return model, RBF


def compute_lqr_gains(A_lifted, B_lifted, n_error=6, q_x=1.0, q_phi=1e-6):
    """
    Compute discrete-time LQR gains using Riccati equation.

    Args:
        A_lifted: Lifted space A matrix
        B_lifted: Lifted space B matrix
        n_error: Original tracking error dimension (6 for 2D quadrotor)
        q_x: Weight for original tracking error components
        q_phi: Weight for lifted dimensions

    Returns:
        Dictionary with P, K, Q, R matrices
    """
    lifted_dim = A_lifted.shape[0]
    m = lifted_dim - n_error

    # Extended Q matrix: more weight to original tracking error components
    Q = np.diag([q_x]*n_error + [q_phi]*m)
    R = np.eye(B_lifted.shape[1])

    print("\n--- Pre-DARE Diagnostics ---")
    print("Checking system properties for DARE solvability...")

    # Check A_lifted properties
    A_lifted_eigenvals = np.linalg.eigvals(A_lifted)
    A_lifted_eigenvals_real = np.real(A_lifted_eigenvals)
    A_lifted_eigenvals_imag = np.imag(A_lifted_eigenvals)
    print(f"A_lifted eigenvalues (real part): min={np.min(A_lifted_eigenvals_real):.4f}, "
          f"max={np.max(A_lifted_eigenvals_real):.4f}")
    print(f"A_lifted eigenvalues (imag part): min={np.min(np.abs(A_lifted_eigenvals_imag)):.4f}, "
          f"max={np.max(np.abs(A_lifted_eigenvals_imag)):.4f}")

    max_eigenval_magnitude = np.max(np.abs(A_lifted_eigenvals))
    print(f"A_lifted max eigenvalue magnitude: {max_eigenval_magnitude:.4f}")

    # Check controllability
    controllability_matrix_lifted = ctrl.ctrb(A_lifted, B_lifted)
    controllability_rank_lifted = np.linalg.matrix_rank(controllability_matrix_lifted)
    print(f"Lifted system controllability rank: {controllability_rank_lifted} / {lifted_dim}")

    # Solve discrete algebraic Riccati equation
    try:
        print("\n--- Computing P matrix for lifted space ---")
        P_lifted = solve_discrete_are(A_lifted, B_lifted, Q, R)
        print(f"Computed P_lifted shape: {P_lifted.shape}")
        print(f"P_lifted trace: {np.trace(P_lifted):.4f}")

        # Check if P_lifted is positive definite
        P_lifted_eigenvals = np.linalg.eigvals(P_lifted)
        P_lifted_eigenvals_real = np.real(P_lifted_eigenvals)
        if np.any(P_lifted_eigenvals_real < 0):
            print(f"  ⚠️  WARNING: P_lifted has {np.sum(P_lifted_eigenvals_real < 0)} negative eigenvalues!")
        if np.max(np.abs(P_lifted_eigenvals_real)) > 1e4:
            print(f"  ⚠️  WARNING: P_lifted has very large eigenvalues "
                  f"(max: {np.max(P_lifted_eigenvals_real):.2e})")
    except Exception as e:
        print(f"Warning: Could not solve DARE for lifted space: {e}")
        print("Falling back to extended identity matrix")
        P_lifted = np.eye(lifted_dim)
        print(f"Using identity P_lifted: shape {P_lifted.shape}")

    # Compute LQR gain K
    K = np.linalg.inv((R + B_lifted.T @ P_lifted @ B_lifted)) @ (B_lifted.T @ P_lifted @ A_lifted)
    print(f"LQR gain K shape: {K.shape}")

    # Compute closed-loop A matrix
    A_cl = A_lifted - B_lifted @ K

    # Check stability
    eig_A = np.linalg.eigvals(A_lifted)
    eig_Acl = np.linalg.eigvals(A_cl)
    print(f"\nOpen-loop spectral radius: {np.max(np.abs(eig_A)):.4f}")
    print(f"Closed-loop spectral radius: {np.max(np.abs(eig_Acl)):.4f}")

    return {
        'P': P_lifted,
        'K': K,
        'Q': Q,
        'R': R,
        'A_cl': A_cl
    }


def analyze_p_matrix(P_lifted, A_lifted, B_lifted):
    """
    Perform diagnostic analysis on P matrix.

    Args:
        P_lifted: P matrix from Riccati equation
        A_lifted: Lifted space A matrix
        B_lifted: Lifted space B matrix
    """
    print("\n" + "="*60)
    print("P_MATRIX DIAGNOSTICS")
    print("="*60)

    P_to_analyze = P_lifted.copy()
    P_name = "P_lifted"

    # Check symmetry
    is_symmetric = np.allclose(P_to_analyze, P_to_analyze.T, rtol=1e-5, atol=1e-8)
    print(f" Symmetry Check ({P_name}):")
    print(f"   P is symmetric: {is_symmetric}")
    if not is_symmetric:
        max_asymmetry = np.max(np.abs(P_to_analyze - P_to_analyze.T))
        print(f"   WARNING: Max asymmetry = {max_asymmetry:.2e}")
        P_to_analyze = (P_to_analyze + P_to_analyze.T) / 2
        print(f"   Fixed: Symmetrized {P_name} as (P + P.T) / 2")

    # Check magnitude and eigenvalues
    eigenvals = np.linalg.eigvals(P_to_analyze)
    eigenvals_real = np.real(eigenvals)
    eigenvals_imag = np.imag(eigenvals)

    print(f" Magnitude Check ({P_name}):")
    print(f"   P matrix stats:")
    print(f"     - Min value: {np.min(P_to_analyze):.2e}")
    print(f"     - Max value: {np.max(P_to_analyze):.2e}")
    print(f"     - Mean |value|: {np.mean(np.abs(P_to_analyze)):.2e}")
    print(f"     - Trace: {np.trace(P_to_analyze):.2e}")
    print(f"     - Frobenius norm: {np.linalg.norm(P_to_analyze, 'fro'):.2e}")

    print(f" Eigenvalue Check ({P_name}):")
    print(f"     - Min eigenvalue (real): {np.min(eigenvals_real):.2e}")
    print(f"     - Max eigenvalue (real): {np.max(eigenvals_real):.2e}")
    condition_num = np.max(eigenvals_real) / (np.min(eigenvals_real) + 1e-10)
    print(f"     - Condition number: {condition_num:.2e}")
    if np.any(eigenvals_real < 0):
        print(f"     - WARNING: {np.sum(eigenvals_real < 0)} negative eigenvalues found!")
        print(f"       P should be positive definite for Lyapunov function")

    # Check A_lifted and B_lifted magnitudes
    print(f" Update Matrices Magnitude:")
    print(f"     - A_lifted: min={np.min(A_lifted):.2e}, max={np.max(A_lifted):.2e}, "
          f"mean|value|={np.mean(np.abs(A_lifted)):.2e}")
    print(f"     - B_lifted: min={np.min(B_lifted):.2e}, max={np.max(B_lifted):.2e}, "
          f"mean|value|={np.mean(np.abs(B_lifted)):.2e}")

    print("\n" + "="*60)


def plot_p_matrix_analysis(P_lifted, save_dir):
    """
    Visualize P matrix properties.

    Args:
        P_lifted: P matrix to visualize
        save_dir: Directory to save plots
    """
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))

    P_to_plot = P_lifted
    P_plot_name = "P_lifted"

    # 1. P matrix heatmap
    ax1 = axes[0]
    im1 = ax1.imshow(P_to_plot, cmap='viridis', aspect='auto')
    ax1.set_title(f'{P_plot_name} Matrix Heatmap')
    ax1.set_xlabel('Column')
    ax1.set_ylabel('Row')
    plt.colorbar(im1, ax=ax1, label='Value')

    # 2. Eigenvalue distribution
    ax2 = axes[1]
    eigenvals = np.linalg.eigvals(P_to_plot)
    eigenvals_real = np.real(eigenvals)
    eigenvals_imag = np.imag(eigenvals)
    ax2.scatter(eigenvals_real, eigenvals_imag, alpha=0.6, s=20)
    ax2.axhline(y=0, color='r', linestyle='--', linewidth=1, alpha=0.5)
    ax2.axvline(x=0, color='r', linestyle='--', linewidth=1, alpha=0.5)
    ax2.set_title(f'{P_plot_name} Eigenvalues')
    ax2.set_xlabel('Real Part')
    ax2.set_ylabel('Imaginary Part')
    ax2.grid(True, alpha=0.3)

    # 3. Eigenvalue magnitude (sorted)
    ax3 = axes[2]
    eigenvals_sorted = np.sort(eigenvals_real)[::-1]  # Sort descending
    ax3.plot(eigenvals_sorted, 'o-', markersize=3)
    ax3.set_title(f'{P_plot_name} Eigenvalues (Sorted)')
    ax3.set_xlabel('Index')
    ax3.set_ylabel('Eigenvalue (Real Part)')
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=0, color='r', linestyle='--', linewidth=1, alpha=0.5)

    plt.tight_layout()

    # Save plot
    plot_path = save_dir / f'P_matrix_analysis_{P_plot_name.lower()}.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Saved P_matrix analysis plot to {plot_path}")

    plt.show()


def evaluate_model(model, X, X_next, U, n_plot=100):
    """
    Evaluate model prediction accuracy.

    Args:
        model: Trained Koopman model
        X: Current tracking error
        X_next: Next tracking error (ground truth)
        U: Control inputs
        n_plot: Number of samples to plot

    Returns:
        RMSE per component and prediction
    """
    X_s = X[:n_plot]
    U_s = U[:n_plot]
    X_next_s = X_next[:n_plot]

    # Koopman model prediction
    X_pred_koop = model.predict(X_s, u=U_s)
    X_pred_koop = np.asarray(X_pred_koop)

    assert X_pred_koop.shape == X_next_s.shape

    # Compute RMSE per component
    rmse_koop = np.sqrt(np.mean((X_pred_koop - X_next_s) ** 2, axis=0))

    print("RMSE (6 tracking error components for 2D quadrotor) - Koopman:",
          np.round(rmse_koop, 4))

    return rmse_koop, X_pred_koop, X_next_s


def plot_prediction_comparison(X_pred, X_true, save_dir, n_plot=100):
    """
    Plot prediction comparison with zoomed insets.

    Args:
        X_pred: Predicted tracking error
        X_true: True tracking error
        save_dir: Directory to save plots
        n_plot: Number of samples
    """
    fig, axs = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    t = np.arange(n_plot)
    state_labels = ['x', 'x_dot', 'z', 'z_dot', 'theta', 'theta_dot']

    # Compute RMSE for each component
    rmse = np.sqrt(np.mean((X_pred - X_true) ** 2, axis=0))

    # Define zoom regions
    zoom_regions = [
        (20, 40),
        (60, 80),
    ]

    for i in range(3):
        axs[i].grid(True, alpha=0.3)
        axs[i].plot(t, X_true[:, i], label=f"True tracking error ({state_labels[i]})",
                   linewidth=2, color='blue')
        axs[i].plot(t, X_pred[:, i], '--', label=f"Koopman pred (RMSE={rmse[i]:.3f})",
                   linewidth=2, color='red', alpha=0.8)
        axs[i].set_ylabel(f"Tracking Error ({state_labels[i]})", fontsize=12)
        axs[i].legend(loc='upper right', fontsize=10)
        axs[i].set_title(f"True vs Koopman Prediction - Tracking Error for {state_labels[i]}",
                        fontsize=12, fontweight='bold')

        # Add zoomed insets
        for zoom_idx, (zoom_start, zoom_end) in enumerate(zoom_regions):
            axins = inset_axes(axs[i], width="30%", height="30%", loc='lower left',
                              bbox_to_anchor=(0.05 + zoom_idx*0.5, 0.05, 1, 1),
                              bbox_transform=axs[i].transAxes)

            zoom_mask = (t >= zoom_start) & (t <= zoom_end)
            axins.plot(t[zoom_mask], X_true[zoom_mask, i], linewidth=2, color='blue')
            axins.plot(t[zoom_mask], X_pred[zoom_mask, i], '--', linewidth=2, color='red', alpha=0.8)
            axins.grid(True, alpha=0.3)
            axins.set_xlim(zoom_start, zoom_end)

            y_min = min(X_true[zoom_mask, i].min(), X_pred[zoom_mask, i].min())
            y_max = max(X_true[zoom_mask, i].max(), X_pred[zoom_mask, i].max())
            y_range = y_max - y_min
            axins.set_ylim(y_min - 0.1*y_range, y_max + 0.1*y_range)

            axs[i].axvspan(zoom_start, zoom_end, alpha=0.1, color='gray')

    axs[-1].set_xlabel('sample', fontsize=12)
    plt.savefig(save_dir / 'EDMD_2D_tracking_error_comparison.png', dpi=150, bbox_inches='tight')
    plt.tight_layout()
    plt.show()


def main():
    """Main execution function."""
    # Configuration
    SAVE_DATA_DIR = Path("Saved_data/")
    SAVE_DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Save data directory: {SAVE_DATA_DIR.absolute()}")

    DATA_FILE = SAVE_DATA_DIR / 'data_EDMD_2D.npz'
    MODEL_SAVE_PATH = SAVE_DATA_DIR / 'edmd_model_2D.pkl'
    RICCATI_SAVE_PATH = SAVE_DATA_DIR / 'lqr_matrices_2D.npz'

    # Hyperparameters
    N_RBF_CENTERS = 3
    RBF_WIDTH = 0.25
    REGULARIZATION = 1e-5
    DT = 1/50.0

    # Load data
    data = load_data(DATA_FILE)
    X = data['tracking_error']
    X_next = data['tracking_error_next']
    U = data['U']

    # Train EDMD model
    model, RBF = train_edmd_model(
        X, X_next, U,
        n_rbf_centers=N_RBF_CENTERS,
        rbf_width=RBF_WIDTH,
        regularization=REGULARIZATION,
        dt=DT
    )

    # Save trained model
    with open(MODEL_SAVE_PATH, 'wb') as f:
        pickle.dump(model, f)
    print(f"Saved Koopman model to {MODEL_SAVE_PATH}")

    # Extract lifted space matrices
    A_lifted = model.A
    B_lifted = model.B
    lifted_dim = A_lifted.shape[0]

    print(f"Extracted A_lifted shape: {A_lifted.shape}")
    print(f"Extracted B_lifted shape: {B_lifted.shape}")
    print(f"Lifted space dimension: {lifted_dim}")

    # Compute LQR gains
    n_error = X.shape[1]  # 6 for 2D quadrotor
    lqr_results = compute_lqr_gains(A_lifted, B_lifted, n_error=n_error)

    P_lifted = lqr_results['P']
    K = lqr_results['K']
    Q = lqr_results['Q']
    R = lqr_results['R']
    A_cl = lqr_results['A_cl']

    # Save Riccati/LQR matrices
    print("\n--- Saving fixed matrices ---")
    save_dict = {
        'A_lifted': A_lifted,
        'B_lifted': B_lifted,
        'A_cl': A_cl,
        'P': P_lifted,
        'K': K,
        'Q': Q,
        'R': R
    }

    np.savez(RICCATI_SAVE_PATH, **save_dict)
    print(f"✓ Saved fixed Riccati/LQR matrices to {RICCATI_SAVE_PATH}")
    print(f"  A_lifted ({A_lifted.shape}), B_lifted ({B_lifted.shape}), P_lifted ({P_lifted.shape})")
    print(f"\nNote: Matrices represent tracking error dynamics (not state dynamics)")
    print(f"  - Normalized Q and R matrices")
    print(f"  - LQR gain K minimizes tracking error")

    # Analyze P matrix
    analyze_p_matrix(P_lifted, A_lifted, B_lifted)

    # Visualize P matrix
    plot_p_matrix_analysis(P_lifted, SAVE_DATA_DIR)

    # Evaluate model and plot predictions
    print("\n--- Evaluating Model Predictions ---")
    rmse, X_pred, X_true = evaluate_model(model, X, X_next, U, n_plot=100)
    print(f"RMSE: {rmse}")
    plot_prediction_comparison(X_pred, X_true, SAVE_DATA_DIR, n_plot=100)

    print("\nEDMD training completed successfully!")


if __name__ == "__main__":
    main()
