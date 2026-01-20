''' LC_SAC.py: Lyapunov Constrained SAC agent
    Author: Dhruv Kushwaha
    Date: 2026-01-06
'''
import torch
from torch import optim
import torch.nn.functional as F
import numpy as np

from safe_control_gym.controllers.sac.sac_utils import MLPActor, MLPQFunction

# Hyperparameters
ACTIVATION = "relu"
DECAY_RATE = 1e-3
EPS = 1e-6  # tolerance; can be small >0

# --- LC-SAC Agent (Lyapunov Constrained SAC) ---
class LCSAC:
    def __init__(self, state_dim, action_dim, action_range, hidden_dim, device, edmd_model,
                 P_lifted, A, B, gamma=0.99, tau=0.005, init_temperature=0.2, actor_lr=3e-4,
                 critic_lr=1e-4, entropy_lr=1e-4, use_entropy_tuning=False, quadtype="quadrotor_2D"):
        '''Initializes the agent (matches official implementation).'''
        # Initialize parameters
        self.device = device
        self.gamma = float(gamma)
        self.tau = float(tau)
        self.edmd_model = edmd_model
        self.P_lifted = torch.FloatTensor(P_lifted).to(self.device)
        self.A = torch.FloatTensor(A).to(self.device)
        self.B = torch.FloatTensor(B).to(self.device)
        self.use_entropy_tuning = use_entropy_tuning
        self.action_range = action_range
        self.init_temperature = init_temperature
        self.entropy_lr = entropy_lr
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.hidden_dim = hidden_dim
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.quadtype = quadtype
        self.q = 0.9 # CVaR level (Not too small to avoid underflow)

        # Initialize lagrange multiplier for Lyapunov constraint
        self.lam_lr = 1e-4
        self.lam = torch.nn.Parameter(torch.tensor(0.0, device=self.device))
        # self.lam_optim = torch.optim.Adam([self.lam], lr=self.lam_lr)
        self.lam_max = 1.1e-1 # optional constraint on lam

        # Networks (use safe_control_gym MLP implementations)
        self.low = torch.tensor(self.action_range[0], device=self.device)
        self.high = torch.tensor(self.action_range[1], device=self.device)

        def unscale_fn(x):  # Rescale action from [-1, 1] to [low, high]
            return self.low.to(x.device) + (0.5 * (x + 1.0) * (self.high.to(x.device) - self.low.to(x.device)))

        self.actor = MLPActor(self.state_dim, self.action_dim,
         hidden_dims=[self.hidden_dim, self.hidden_dim], activation=ACTIVATION,
         postprocess_fn=unscale_fn).to(self.device)
        self.critic_1 = MLPQFunction(self.state_dim, self.action_dim,
            hidden_dims=[self.hidden_dim, self.hidden_dim], activation=ACTIVATION).to(self.device)
        self.critic_2 = MLPQFunction(self.state_dim, self.action_dim,
            hidden_dims=[self.hidden_dim, self.hidden_dim], activation=ACTIVATION).to(self.device)

        # Target networks
        self.critic_1_target = MLPQFunction(self.state_dim, self.action_dim,
            hidden_dims=[self.hidden_dim, self.hidden_dim], activation=ACTIVATION).to(self.device)
        self.critic_2_target = MLPQFunction(self.state_dim, self.action_dim,
            hidden_dims=[self.hidden_dim, self.hidden_dim], activation=ACTIVATION).to(self.device)

        # Copy weights to target networks
        self.critic_1_target.load_state_dict(self.critic_1.state_dict())
        self.critic_2_target.load_state_dict(self.critic_2.state_dict())

        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        # Single optimizer for both critics (matches official implementation)
        self.critic_optimizer = optim.Adam(
            list(self.critic_1.parameters()) + list(self.critic_2.parameters()),
            lr=critic_lr
        )

        # Entropy tuning - use log_alpha approach (matches official implementation)
        # Initialize log_alpha as a leaf tensor (important for optimizer)
        # Match baseline approach: create as leaf tensor so requires_grad works properly
        self.log_alpha = torch.tensor(np.log(float(self.init_temperature)),
            device=self.device, dtype=torch.float32, requires_grad=self.use_entropy_tuning)
        if self.use_entropy_tuning:
            self.target_entropy = -np.prod(np.array([action_dim])).item()
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=self.entropy_lr)


    @property
    def alpha(self):
        '''Entropy-tuning parameter/temperature (matches official SAC implementation)'''
        return self.log_alpha.exp()

    def train(self):
        '''Sets training mode (matches official implementation).'''
        self.actor.train()
        self.critic_1.train()
        self.critic_2.train()
        # Official implementation always sets requires_grad = True in train mode
        if self.use_entropy_tuning:
            self.log_alpha.requires_grad_(True)
        else:
            self.log_alpha.requires_grad_(False)

    def eval(self):
        '''Sets evaluation mode (matches official implementation).'''
        self.actor.eval()
        self.critic_1.eval()
        self.critic_2.eval()
        if self.use_entropy_tuning:
            self.log_alpha.requires_grad = False

    def select_action(self, state, deterministic=False):
        '''Selects action for given state (matches official implementation).'''
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        if deterministic:
            action, _ = self.actor(state, deterministic=True, with_logprob=False)
        else:
            action, _ = self.actor(state, deterministic=False, with_logprob=True)

        action = action.detach().cpu().numpy()[0]

        # Actions are already scaled via postprocess_fn in MLPActor (from [-1, 1] to [low, high])
        # No need to clip - the actor network handles the scaling internally
        # Clipping would be redundant and can cause issues with gradient flow

        return action

    def lift_state(self, state_error):
        """Lift state to EDMD lifted space

        Args:
            state_batch_np: Observation batch (n_batch, obs_dim) where obs_dim may be 24 for 2D
                            The first 6 dimensions are the actual state for 2D quadrotor
        """
        # Extract state portion from observation
        # For 2D: state is 6D [x, x_dot, z, z_dot, theta, theta_dot]
        # For 2D EDMD model, we can directly use the 6D state
        # The EDMD model was trained on 2D data, so it expects 6D input
        if self.quadtype == "quadrotor_2D":
            state_error_feats = state_error[:, :6].detach().cpu().numpy()
        elif self.quadtype == "quadrotor_3D":
            state_error_feats = state_error[:, :12].detach().cpu().numpy()
        else:
            raise ValueError(f"Invalid quadtype: {self.quadtype}")

        if self.edmd_model is not None:
            # Transform state to lifted space using EDMD observables
            lifted_error = self.edmd_model.observables.transform(state_error_feats)
        else:
            # Fallback: if model not loaded, we can't lift (shouldn't happen if matrices are saved)
            raise RuntimeError("EDMD model not loaded and cannot lift state")
        return torch.FloatTensor(lifted_error).to(self.device)

    def Lyapunov_fn(self, x: torch.Tensor) -> torch.Tensor:
        '''Lyapunov function V(x) = x^T P x'''
        assert x.ndim == 2, f"Expected 2D tensor, got {x.ndim}D"
        # ensure P_lifted is float32
        return torch.einsum('bi,ij,bj->b', x, self.P_lifted, x).unsqueeze(-1)

    def update(self, replay_buffer, batch_size):
        '''Updates the agent (matches official implementation).'''
        # Now set training mode (log_alpha is guaranteed to exist now)
        self.train()

        # Sample batch (SACBuffer returns a dict with keys: 'obs', 'act', 'rew', 'next_obs', 'mask', 'X_error')
        batch = replay_buffer.sample(batch_size, self.device)
        state = batch['obs']
        action = batch['act']
        reward = batch['rew']
        next_state = batch['next_obs']
        mask = batch['mask']  # SACBuffer uses mask (1.0 for not done, 0.0 for done)

        # Extract tracking error
        X_error = batch['X_error']

        assert X_error.shape[0] == state.shape[0]

        # Update critics first (combine losses and optimize together
        # - matches official implementation)
        current_q1 = self.critic_1(state, action)
        current_q2 = self.critic_2(state, action)

        with torch.no_grad():
            # Sample next action from actor for target computation
            action_next, next_log_prob = self.actor(next_state,
            deterministic=False, with_logprob=True)
            # Not using LQR action for critic target computation

            # Compute target Q-values using target networks
            target_q1 = self.critic_1_target(next_state, action_next)
            target_q2 = self.critic_2_target(next_state, action_next)
            min_target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_prob
            # Q value regression target
            # (standard SAC formula: reward + gamma * mask * (min_q - alpha * log_prob))
            #min_target_q = torch.clamp(min_target_q, -50.0, 50.0)  # ADD: Clip targets
            target_q = reward + self.gamma * mask * (min_target_q)


        # Cross check shapes
        assert reward.ndim == 2 and reward.shape[1] == 1
        assert mask.ndim == 2 and mask.shape[1] == 1
        assert current_q1.shape == target_q.shape
        assert current_q2.shape == target_q.shape

        # Compute critic losses and combine them
        q1_loss = F.huber_loss(current_q1, target_q)
        q2_loss = F.huber_loss(current_q2, target_q)
        critic_loss = q1_loss + q2_loss

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(list(self.critic_1.parameters()) + list(self.critic_2.parameters()), max_norm=10.0)
        self.critic_optimizer.step()

        # Update actor with Lyapunov constraint
        action_new, log_prob = self.actor(state, deterministic=False, with_logprob=True)
        # Compute critic values for new action

        # Not using Lyapunov constraint action for critic value computation
        for p in self.critic_1.parameters():
            p.requires_grad_(False)
        for p in self.critic_2.parameters():
            p.requires_grad_(False)

        q1_new = self.critic_1(state, action_new)
        q2_new = self.critic_2(state, action_new)

        for p in self.critic_1.parameters():
            p.requires_grad_(True)
        for p in self.critic_2.parameters():
            p.requires_grad_(True)

        # Standard SAC policy loss
        q_new = torch.min(q1_new, q2_new)
        policy_loss = ((self.alpha.detach() * log_prob) - q_new).mean()


        # Computed lifted tracking error
        z_error = self.lift_state(X_error)

        # Compute next tracking error prediction for Lyapunov decrease condition
        # Use the EDMD dynamics: tracking_error_next_lifted = A * tracking_error_current_lifted + B * action
        # Note: Gradients flow through action_new here, allowing the actor to learn actions that reduce violations
        z_next_error = z_error @ self.A.T + action_new @ self.B.T
        # Compute Lyapunov function values for tracking errors
        V_current = self.Lyapunov_fn(z_error)
        V_next = self.Lyapunov_fn(z_next_error)

        # Margin based on tracking error magnitude (decay rate)
        # Use the lifted tracking error magnitude for margin
        # y_lifted_tracking_error shape: (batch_size, lifted_dim)
        margin = DECAY_RATE * V_current
        lyap_violation = F.relu(V_next - V_current + margin)*mask  # zero out terminal states

        # Use CVaR loss instead of mean
        q = self.q
        temp_size = lyap_violation.shape[0]
        k = max(1, int((1 - q) * temp_size))
        lyap_topk = torch.topk(lyap_violation.view(-1), k, largest=True).values
        lyap_loss = lyap_topk.mean()
        #lyap_loss = lyap_violation.mean()

        # Combined actor loss: weighted combination of SAC loss and Lyapunov loss
        #total actor loss = policy loss + lambda * (lyap loss - eps)
        total_actor_loss = policy_loss + self.lam.detach() * (lyap_loss - EPS)
        self.actor_optimizer.zero_grad()
        total_actor_loss.backward()

        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=10.0)
        self.actor_optimizer.step()

        # Update Lagrange multiplier
        # self.lam_optim.zero_grad(set_to_none=True)
        # dual_loss = -(self.lam * (lyap_loss.detach() - EPS))
        # dual_loss.backward()
        # self.lam_optim.step()
        # with torch.no_grad():
        #     self.lam.clamp_(0.0, self.lam_max)  # keep nonnegative
        with torch.no_grad():
            self.lam.data += self.lam_lr * (lyap_loss.detach() - EPS)
            self.lam.data.clamp_(0.0, self.lam_max)


        # Update entropy coefficient (if enabled)
        entropy_loss = torch.zeros(1)
        if self.use_entropy_tuning:
            entropy_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            entropy_loss.backward()
            self.alpha_optimizer.step()

        # Soft update target networks
        self._soft_update(self.critic_1_target, self.critic_1)
        self._soft_update(self.critic_2_target, self.critic_2)

        return {
            'critic_loss': critic_loss.item(),
            'actor_loss': total_actor_loss.item(),
            'policy_loss': policy_loss.item(),
            'lyap_loss': lyap_loss.item(),
            'entropy_loss': entropy_loss.item() if isinstance(entropy_loss, torch.Tensor) else 0.0,
            'alpha': self.alpha.detach().item(),
            'actor_lr': self.actor_optimizer.param_groups[0]['lr'],
            'critic_lr': self.critic_optimizer.param_groups[0]['lr']
        }

    def _soft_update(self, target, source):
        '''Soft updates the target networks (matches official implementation).'''
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)

    def save(self, filepath):
        '''Saves the agent (matches official implementation).'''
        state_dict = {
            'actor': self.actor.state_dict(),
            'critic_1': self.critic_1.state_dict(),
            'critic_2': self.critic_2.state_dict(),
            'critic_1_target': self.critic_1_target.state_dict(),
            'critic_2_target': self.critic_2_target.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'log_alpha': self.log_alpha,
            'lam': self.lam,
        }
        if self.use_entropy_tuning:
            state_dict['alpha_optimizer'] = self.alpha_optimizer.state_dict()
        torch.save(state_dict, filepath)

    def load(self, filepath):
        '''Loads the agent (matches official implementation).'''
        checkpoint = torch.load(filepath, weights_only=False)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic_1.load_state_dict(checkpoint['critic_1'])
        self.critic_2.load_state_dict(checkpoint['critic_2'])
        self.critic_1_target.load_state_dict(checkpoint.get('critic_1_target',
        checkpoint['critic_1']))
        self.critic_2_target.load_state_dict(checkpoint.get('critic_2_target',
        checkpoint['critic_2']))
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        if 'critic_optimizer' in checkpoint:
            self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
        if 'log_alpha' in checkpoint and checkpoint['log_alpha'] is not None:
            self.log_alpha.data.copy_(checkpoint['log_alpha'].to(self.device).data)
        if self.use_entropy_tuning and 'alpha_optimizer' in checkpoint:
            self.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer'])
        if 'lam' in checkpoint and checkpoint['lam'] is not None:
            self.lam.data.copy_(checkpoint['lam'].to(self.device).data)