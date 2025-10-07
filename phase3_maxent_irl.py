# phase3_maxent_irl.py  (multi-agent / conditional IRL)
import os, math, json, random
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------
# Utilities
# ----------------------------
def set_seed(seed: int = 42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def to_torch(x, device):
    return torch.from_numpy(x).float().to(device)

def detach_cpu(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()

# ----------------------------
# Load Phase-1 data (S, E, A)  -- multi-agent aware
# ----------------------------
class MultiEmbActWindows:
    """
    Wrapper for Phase-1 NPZ with:
      S:[N,T,D_s]  full state (concat [E ; G]) -- if missing, falls back to E
      E:[N,T,D_e]  embedding-only
      A:[N,T-1,D_e] actions in embedding space (e_{t+1} - e_t)
      y:[N], meta:[N]
    Provides:
      - demo window sampling (for μ̂ and dyn updates)
      - seed sequences for conditional rollouts (returns e0 and g-seq)
    """
    def __init__(self, npz_path: str):
        d = np.load(npz_path, allow_pickle=True)
        if 'S' in d.files:
            self.S = d["S"]        # [N,T,D_s]
        else:
            self.S = d["E"]        # fallback to single-agent
        self.E = d["E"]            # [N,T,D_e]
        self.A = d["A"]            # [N,T-1,D_e]
        self.y = d["y"]
        self.meta = d["meta"]

        self.N, self.T, self.D_s = self.S.shape
        self.D_e = self.E.shape[-1]
        assert self.A.shape == (self.N, self.T-1, self.D_e)

        self.D_g = max(0, self.D_s - self.D_e)   # social feature dim (could be 0 if single-agent fallback)

        # indices of normal windows if available
        self.normal_idxs = np.where(self.y == 0)[0] if (self.y is not None) else np.arange(self.N)
        if len(self.normal_idxs) == 0:
            self.normal_idxs = np.arange(self.N)

    def sample_normal_windows(self, batch_windows: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        sel = np.random.choice(self.normal_idxs, size=min(batch_windows, len(self.normal_idxs)), replace=False)
        return self.S[sel], self.E[sel], self.A[sel]  # [B,T,D_s], [B,T,D_e], [B,T-1,D_e]

    def sample_seed_sequences(self, batch_size: int, horizon: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns:
          e0:   [B, D_e]
          gseq: [B, horizon+1, D_g]  (exogenous social context), if D_g==0 -> zeros
        We ensure t0 + horizon <= T-1 so rollouts have H steps.
        """
        B = batch_size
        e0 = np.zeros((B, self.D_e), dtype=np.float32)
        gseq = np.zeros((B, horizon+1, self.D_g), dtype=np.float32) if self.D_g > 0 else np.zeros((B, horizon+1, 0), dtype=np.float32)

        # draw (window, start) pairs
        for i in range(B):
            w = np.random.choice(self.normal_idxs)
            # valid t0 in [0, T-1-horizon]
            max_start = max(0, self.T - 1 - horizon)
            t0 = np.random.randint(0, max_start + 1) if (max_start > 0) else 0
            e0[i] = self.E[w, t0]
            if self.D_g > 0:
                # collect g_t from S: S = [E ; G] -> take last D_g dims
                gseq[i] = self.S[w, t0:t0+horizon+1, self.D_e:]
        return e0, gseq

# ----------------------------
# Phase-2 models (multi-agent)
# ----------------------------
class RewardFeaturesMLP(nn.Module):
    """ φ_ψ(s,a) = MLP_ψ([s,a]) """
    def __init__(self, state_dim:int, act_dim:int, feat_dim:int=64):
        super().__init__()
        self.f = nn.Sequential(
            nn.Linear(state_dim + act_dim, 64),
            nn.ELU(inplace=True),
            nn.Linear(64, 64),
            nn.ELU(inplace=True),
            nn.Linear(64, feat_dim)
        )
        self.ln = nn.LayerNorm(feat_dim)

    def forward(self, s, a):
        z = torch.cat([s, a], dim=-1)
        return self.ln(self.f(z))

class RewardLinearHead(nn.Module):
    """ r_θ = θ^T φ """
    def __init__(self, feat_dim:int):
        super().__init__()
        self.theta = nn.Parameter(torch.zeros(feat_dim))
    def forward(self, phi):
        return (phi * self.theta).sum(dim=-1, keepdim=True)

class RewardModel(nn.Module):
    """ r(s,a) = <θ, φ_ψ(s,a)> """
    def __init__(self, state_dim:int, act_dim:int, feat_dim:int=64):
        super().__init__()
        self.feat = RewardFeaturesMLP(state_dim, act_dim, feat_dim)
        self.head = RewardLinearHead(feat_dim)
    def forward(self, s, a, return_phi=False):
        phi = self.feat(s,a)
        r = self.head(phi)
        return (r, phi) if return_phi else r

class GaussianPolicy(nn.Module):
    """
    π_ω(a|s) = N( μ_ω(s), diag(σ^2) ), with learned state-independent log_std
      s ∈ ℝ^{D_s}, a ∈ ℝ^{D_e}
    """
    def __init__(self, state_dim:int, act_dim:int, hidden:int=64, init_log_std:float=-1.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ELU(inplace=True),
            nn.Linear(hidden, hidden),    nn.ELU(inplace=True),
            nn.Linear(hidden, act_dim)
        )
        self.log_std = nn.Parameter(torch.ones(act_dim) * init_log_std)

    def forward(self, s):
        mu = self.net(s)                 # [B,D_e]
        log_std = self.log_std.expand_as(mu)
        return mu, log_std

    def sample(self, s):
        mu, log_std = self.forward(s)
        std = torch.exp(log_std)
        eps = torch.randn_like(mu)
        return mu + std * eps, mu, log_std

    def log_prob_and_entropy(self, s, a):
        mu, log_std = self.forward(s)
        var = torch.exp(2*log_std)
        lp = -0.5 * (((a - mu)**2)/var + 2*log_std + math.log(2*math.pi))
        lp = lp.sum(dim=-1)  # [B]
        # entropy of diag Gaussian per sample (log_std is shared across s)
        ent = 0.5 * (1.0 + math.log(2*math.pi)) * a.size(-1) + log_std.sum()
        ent = ent.expand_as(lp)
        return lp, ent

class ConditionalEDynamics(nn.Module):
    """
    p_ψ(e'|s,a) = N( mean = e + a + g_ψ([s,a]), diag(σ_e^2) )
      - s = [e ; g], we model only e' (embedding next state).
    """
    def __init__(self, state_dim:int, act_dim:int, embed_dim:int,
                 hidden:int=64, init_log_sigma:float=-2.0,
                 learn_residual:bool=True, learn_sigma:bool=True):
        super().__init__()
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.embed_dim = embed_dim
        self.learn_residual = learn_residual
        self.learn_sigma = learn_sigma

        if learn_residual:
            self.res = nn.Sequential(
                nn.Linear(state_dim + act_dim, hidden), nn.ELU(inplace=True),
                nn.Linear(hidden, hidden),              nn.ELU(inplace=True),
                nn.Linear(hidden, embed_dim)
            )
        else:
            self.res = None

        if learn_sigma:
            self.log_sigma_e = nn.Parameter(torch.ones(embed_dim) * init_log_sigma)
        else:
            self.register_buffer("log_sigma_e", torch.ones(embed_dim) * init_log_sigma)

    def mean_e(self, s, a):
        e = s[..., :self.embed_dim]
        base = e + a
        if self.res is None: return base
        return base + self.res(torch.cat([s, a], dim=-1))

    def sample_e(self, s, a):
        mu_e = self.mean_e(s, a)
        std = torch.exp(self.log_sigma_e)
        return mu_e + std * torch.randn_like(mu_e)

    def nll(self, s, a, e_next):
        mu_e = self.mean_e(s, a)
        log_var = 2*self.log_sigma_e
        var = torch.exp(log_var)
        nll = 0.5 * ((e_next - mu_e)**2 / var + log_var + math.log(2*math.pi))
        return nll.mean()

# ----------------------------
# Combined reward (learned φ + simple hand features)
# ----------------------------
class CombinedReward(nn.Module):
    """
    r(s,a,sp,a_prev) = <theta_all, concat( φ_ψ(s,a),  hand_feats(s,a,sp,a_prev) )>
    hand_feats = [s, a, ||a||, ||a||^2, cos(s,sp), ||sp-s||, ||a-a_prev||]
                 dims = D_s + D_e + 5
    """
    def __init__(self, reward_model: RewardModel, state_dim:int, act_dim:int, init_theta_hand: Optional[torch.Tensor]=None):
        super().__init__()
        self.reward = reward_model
        F_learn = reward_model.head.theta.numel()
        F_hand = state_dim + act_dim + 5
        self.theta_all = nn.Parameter(torch.zeros(F_learn + F_hand))

        # initialize learned-part with pre-trained theta, keep hand part ~0
        with torch.no_grad():
            self.theta_all[:F_learn].copy_(reward_model.head.theta.data)
            if init_theta_hand is not None and init_theta_hand.numel() == F_hand:
                self.theta_all[F_learn:].copy_(init_theta_hand)

        self.F_learn = F_learn
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.F_hand = F_hand

    @staticmethod
    def _hand_features(s, a, sp, a_prev, eps=1e-6):
        a_norm = torch.linalg.norm(a, dim=-1, keepdim=True)
        a_norm2= (a**2).sum(dim=-1, keepdim=True)
        cos   = (s*sp).sum(-1, keepdim=True) / (torch.linalg.norm(s,dim=-1,keepdim=True)*torch.linalg.norm(sp,dim=-1,keepdim=True) + eps)
        speed = torch.linalg.norm(sp - s, dim=-1, keepdim=True)
        smooth= torch.linalg.norm(a - a_prev, dim=-1, keepdim=True)
        return torch.cat([s, a, a_norm, a_norm2, cos, speed, smooth], dim=-1)

    def forward(self, s, a, sp, a_prev):
        phi_learn = self.reward.feat(s,a)             # [B, F_learn]
        phi_hand  = self._hand_features(s,a,sp,a_prev)# [B, F_hand]
        phi_all   = torch.cat([phi_learn, phi_hand], dim=-1)
        r = (phi_all * self.theta_all).sum(dim=-1, keepdim=True)
        return r, phi_all

# ----------------------------
# Config
# ----------------------------
@dataclass
class IRLCfg:
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42
    # horizons / sampling
    horizon: int = 63          # typically T-1 if T=64
    rollouts_per_iter: int = 256
    s0_batch: int = 256
    # temps / weights
    temperature: float = 5.0   # weight w_i = exp(R_i / tau)
    tau_anneal: float = 0.98   # per-iter anneal
    # learning rates
    lr_reward: float = 1e-3
    lr_policy: float = 5e-4
    weight_decay: float = 1e-6
    # policy entropy coefficient
    alpha_entropy: float = 0.01
    # reward regularization
    l2_theta: float = 1e-4
    reward_clip: float = 10.0
    # iterations
    iters: int = 80
    # demo batch for μ_hat / dyn update
    demo_windows_per_batch: int = 128
    # dynamics update
    update_dynamics: bool = False
    dyn_lr: float = 1e-3
    # gradient safety
    grad_clip: float = 1.0

# ----------------------------
# Demo feature expectation μ̂  (uses real social context)
# ----------------------------
def demo_feature_expectations(data: MultiEmbActWindows,
                              comb_reward: CombinedReward,
                              cfg: IRLCfg) -> torch.Tensor:
    """
    μ̂ ≈ E_normal[ sum_t φ(s_t,a_t, s_{t+1}, a_{t-1}) ]
    """
    S, E, A = data.sample_normal_windows(cfg.demo_windows_per_batch)   # [B,T,D_s], [B,T,D_e], [B,T-1,D_e]
    B, T, D_s = S.shape
    D_e = E.shape[-1]
    device = cfg.device

    s  = to_torch(S[:, :-1, :].reshape(-1, D_s), device)  # [B*(T-1), D_s]
    a  = to_torch(A.reshape(-1, D_e), device)             # [B*(T-1), D_e]
    sp = to_torch(S[:, 1:,  :].reshape(-1, D_s), device)  # [B*(T-1), D_s]

    a_prev_np = np.concatenate([np.zeros((B,1,D_e), dtype=np.float32),
                                A[:, :-1, :]], axis=1).reshape(-1, D_e)
    a_prev = to_torch(a_prev_np, device)

    with torch.no_grad():
        _, phi = comb_reward(s, a, sp, a_prev)  # [B*(T-1), F]
        F = phi.shape[-1]
        phi_step = phi.view(B, T-1, F).sum(dim=1)   # [B,F]
        mu_hat = phi_step.mean(dim=0)               # [F]
    return mu_hat

# ----------------------------
# Conditional rollouts with exogenous g-sequence
# ----------------------------
@torch.no_grad()
def sample_rollouts_cond(policy: GaussianPolicy,
                         dyn: ConditionalEDynamics,
                         e0: torch.Tensor,          # [B, D_e]
                         gseq: torch.Tensor,        # [B, H+1, D_g]
                         horizon: int) -> Dict[str, torch.Tensor]:
    """
    Roll out for H steps:
      s_t = [e_t ; g_t], a_t ~ π(a|s_t), e_{t+1} ~ p(e'|s_t, a_t), s_{t+1} = [e_{t+1} ; g_{t+1}]
    Returns dict with s:[B,H,D_s], a:[B,H,D_e], sp:[B,H,D_s], a_prev:[B,H,D_e]
    """
    device = e0.device
    B, D_e = e0.shape
    D_g = gseq.shape[-1] if gseq.ndim == 3 else 0
    D_s = D_e + D_g

    e_t = e0                              # [B,D_e]
    a_prev = torch.zeros(B, D_e, device=device)

    s_list, a_list, sp_list = [], [], []

    for t in range(horizon):
        g_t  = gseq[:, t, :] if D_g > 0 else torch.zeros(B, 0, device=device)
        s_t  = torch.cat([e_t, g_t], dim=-1) if D_g > 0 else e_t  # [B,D_s]
        a_t, _, _ = policy.sample(s_t)                             # [B,D_e]
        e_next = dyn.sample_e(s_t, a_t)                            # [B,D_e]
        g_next = gseq[:, t+1, :] if D_g > 0 else torch.zeros(B, 0, device=device)
        s_next = torch.cat([e_next, g_next], dim=-1) if D_g > 0 else e_next

        s_list.append(s_t); a_list.append(a_t); sp_list.append(s_next)
        e_t = e_next

    s  = torch.stack(s_list, dim=1)        # [B,H,D_s]
    a  = torch.stack(a_list, dim=1)        # [B,H,D_e]
    sp = torch.stack(sp_list, dim=1)       # [B,H,D_s]
    a_prev_seq = torch.cat([torch.zeros(B,1,D_e, device=device), a[:, :-1, :]], dim=1)  # [B,H,D_e]
    return {"s": s, "a": a, "sp": sp, "a_prev": a_prev_seq}

# ----------------------------
# Compute rewards, weights, and μ_theta from rollouts
# ----------------------------
def rollout_stats(comb_reward: CombinedReward,
                  ro: Dict[str, torch.Tensor],
                  cfg: IRLCfg) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    returns:
      R: [B] total return per traj
      w: [B] normalized weights
      mu_theta: [F] weighted feature expectation
      r_per_step: [B,H,1] unclipped per-step rewards (for policy update)
    """
    B, H, D_s = ro["s"].shape
    D_e = ro["a"].shape[-1]
    s  = ro["s"].reshape(-1, D_s)
    a  = ro["a"].reshape(-1, D_e)
    sp = ro["sp"].reshape(-1, D_s)
    ap = ro["a_prev"].reshape(-1, D_e)

    r, phi = comb_reward(s, a, sp, ap)                 # r:[B*H,1], phi:[B*H,F]
    r_clipped = r.clamp(-cfg.reward_clip, cfg.reward_clip)
    Fdim = phi.size(-1)
    r_step = r_clipped.view(B, H, 1)                   # [B,H,1]
    R = r_step.sum(dim=1).squeeze(-1)                  # [B]

    with torch.no_grad():
        tau = cfg.temperature
        Rn = R - R.max()                                # stabilize
        w = torch.exp(Rn / max(1e-6, tau))             # [B]
        w = w / (w.sum() + 1e-8)

    phi_sum = phi.view(B, H, Fdim).sum(dim=1)          # [B,F]
    mu_theta = (w.view(B,1) * phi_sum).sum(dim=0)      # [F]
    return R, w, mu_theta, r_step

# ----------------------------
# Trainer
# ----------------------------
class MaxEntIRLTrainer:
    def __init__(self,
                 train_npz: str,
                 policy_ckpt: str,
                 dyn_ckpt: str,
                 reward_ckpt: str,
                 use_handcrafted: bool = True,
                 cfg: IRLCfg = IRLCfg()):
        set_seed(cfg.seed)
        self.cfg = cfg
        self.device = cfg.device

        # Data
        self.data = MultiEmbActWindows(train_npz)
        self.D_s = self.data.D_s
        self.D_e = self.data.D_e
        self.D_g = self.data.D_g
        self.H = min(cfg.horizon, self.data.T - 1)

        # Load policy
        pck = torch.load(policy_ckpt, map_location="cpu")
        self.policy = GaussianPolicy(pck["state_dim"], pck["act_dim"])
        self.policy.load_state_dict(pck["model"])
        self.policy.to(self.device)

        # Load dynamics (conditional e-dynamics)
        dck = torch.load(dyn_ckpt, map_location="cpu")
        self.dyn = ConditionalEDynamics(dck["state_dim"], dck["act_dim"], dck["embed_dim"],
                                        learn_residual=dck.get("learn_residual", True),
                                        learn_sigma=dck.get("learn_sigma", True))
        self.dyn.load_state_dict(dck["model"])
        self.dyn.to(self.device)

        # Load reward init
        rck = torch.load(reward_ckpt, map_location="cpu")
        rmodel = RewardModel(rck["state_dim"], rck["act_dim"], feat_dim=rck["feat_dim"])
        rmodel.load_state_dict(rck["model"])
        rmodel.to(self.device)

        self.use_hand = use_handcrafted
        self.comb_reward = CombinedReward(rmodel, state_dim=self.D_s, act_dim=self.D_e).to(self.device)

        # Optimizers
        self.opt_reward = torch.optim.Adam(
            [{"params": self.comb_reward.reward.parameters(), "lr": cfg.lr_reward, "weight_decay": cfg.weight_decay},
             {"params": [self.comb_reward.theta_all], "lr": cfg.lr_reward, "weight_decay": cfg.weight_decay}]
        )
        self.opt_policy = torch.optim.Adam(self.policy.parameters(), lr=cfg.lr_policy, weight_decay=cfg.weight_decay)
        if cfg.update_dynamics:
            self.opt_dyn = torch.optim.Adam(self.dyn.parameters(), lr=cfg.dyn_lr, weight_decay=cfg.weight_decay)
        else:
            self.opt_dyn = None

        # Precompute μ̂
        self.mu_hat = demo_feature_expectations(self.data, self.comb_reward, cfg).to(self.device)

    def step_reward(self, mu_theta: torch.Tensor):
        # Feature matching: (μθ − μ̂)·θ + λ||θ||^2
        theta = self.comb_reward.theta_all
        loss = torch.dot(mu_theta - self.mu_hat, theta) + self.cfg.l2_theta * (theta @ theta)
        self.opt_reward.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.comb_reward.parameters(), self.cfg.grad_clip)
        self.opt_reward.step()
        return loss.item()

    def step_policy(self, ro: Dict[str, torch.Tensor], r_per_step: torch.Tensor):
        """
        Entropy-regularized REINFORCE with trajectory returns as weight (no baseline).
        """
        B, H, D_s = ro["s"].shape
        D_e = ro["a"].shape[-1]
        s = ro["s"].reshape(-1, D_s); a = ro["a"].reshape(-1, D_e)
        logp, ent = self.policy.log_prob_and_entropy(s, a)    # [B*H], [B*H]

        Rtraj = r_per_step.sum(dim=1, keepdim=True)           # [B,1]
        Rrep  = Rtraj.repeat(1, H).reshape(-1)                # [B*H]
        loss = -( (Rrep / max(1e-6, self.cfg.temperature)) * logp ).mean() - self.cfg.alpha_entropy * ent.mean()

        self.opt_policy.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.cfg.grad_clip)
        self.opt_policy.step()
        return loss.item()

    def step_dynamics(self):
        if self.opt_dyn is None: return None
        # Update on real transitions (uses full s and a, but predicts e_next)
        S, E, A = self.data.sample_normal_windows(self.cfg.demo_windows_per_batch)
        B, T, D_s = S.shape; D_e = E.shape[-1]
        s  = to_torch(S[:, :-1, :].reshape(-1, D_s), self.device)
        a  = to_torch(A.reshape(-1, D_e), self.device)
        e_next = to_torch(E[:, 1:, :].reshape(-1, D_e), self.device)
        loss = self.dyn.nll(s, a, e_next)

        self.opt_dyn.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(self.dyn.parameters(), self.cfg.grad_clip)
        self.opt_dyn.step()
        return loss.item()

    def train(self, out_dir: str = "phase3_ckpt"):
        os.makedirs(out_dir, exist_ok=True)
        log = []
        for it in range(1, self.cfg.iters+1):
            # 1) Seed states (e0) & exogenous g-sequences; sample conditional rollouts
            e0_np, gseq_np = self.data.sample_seed_sequences(self.cfg.s0_batch, self.H)
            e0   = to_torch(e0_np,   self.device)                # [B,D_e]
            gseq = to_torch(gseq_np, self.device)                # [B,H+1,D_g]
            ro = sample_rollouts_cond(self.policy, self.dyn, e0, gseq, horizon=self.H)

            # 2) Compute rollout stats (R, weights, μθ, per-step rewards)
            R, w, mu_theta, r_step = rollout_stats(self.comb_reward, ro, self.cfg)

            # 3) Reward update (feature matching)
            loss_r = self.step_reward(mu_theta)

            # 4) Policy improvement
            loss_pi = self.step_policy(ro, r_step)

            # 5) (Optional) update dynamics on real data
            loss_dyn = self.step_dynamics()

            # 6) Anneal temperature
            self.cfg.temperature = max(1.0, self.cfg.temperature * self.cfg.tau_anneal)

            # 7) Log & save
            log_entry = {
                "iter": it,
                "mean_R": float(R.mean().item()),
                "max_R": float(R.max().item()),
                "reward_loss": loss_r,
                "policy_loss": loss_pi,
                "dyn_loss": None if loss_dyn is None else float(loss_dyn),
                "temperature": float(self.cfg.temperature),
            }
            log.append(log_entry)
            if it % 5 == 0 or it == self.cfg.iters:
                torch.save({
                    "comb_reward": self.comb_reward.state_dict(),
                    "policy": self.policy.state_dict(),
                    "dynamics": self.dyn.state_dict(),
                    "cfg": self.cfg.__dict__,
                    "D_s": self.D_s, "D_e": self.D_e, "D_g": self.D_g
                }, os.path.join(out_dir, f"iter_{it:03d}.pt"))
            print(f"[Iter {it:03d}] R(mean/max)={log_entry['mean_R']:.3f}/{log_entry['max_R']:.3f} "
                  f"| Lr={loss_r:.4f} | Lpi={loss_pi:.4f} | Ldyn={log_entry['dyn_loss']} | tau={self.cfg.temperature:.2f}")

        # final save
        final_path = os.path.join(out_dir, "final.pt")
        torch.save({
            "comb_reward": self.comb_reward.state_dict(),
            "policy": self.policy.state_dict(),
            "dynamics": self.dyn.state_dict(),
            "cfg": self.cfg.__dict__,
            "D_s": self.D_s, "D_e": self.D_e, "D_g": self.D_g
        }, final_path)
        with open(os.path.join(out_dir, "train_log.json"), "w") as f:
            json.dump(log, f, indent=2)
        print(f"✅ Phase-3 finished. Saved to {final_path}")

    @torch.no_grad()
    def score_windows(self, npz_path: str, out_path: str):
        """
        Per-window anomaly score (multi-agent):
          score = - sum_t r([e_t,g_t], a_t, [e_{t+1},g_{t+1}], a_{t-1})
        Saves {scores, y, meta}.
        """
        d = np.load(npz_path, allow_pickle=True)
        if 'S' in d.files:
            S = d["S"]   # [N,T,D_s]
        else:
            S = d["E"]   # fallback
        E = d["E"]       # [N,T,D_e]
        A = d["A"]       # [N,T-1,D_e]
        y = d["y"]; meta = d["meta"]

        N, T, D_s = S.shape
        D_e = E.shape[-1]
        H = T - 1
        scores = np.zeros((N,), dtype=np.float32)

        for i in tqdm(range(N), desc="Scoring"):
            s  = to_torch(S[i, :-1, :], self.device)          # [H,D_s]
            a  = to_torch(A[i],           self.device)        # [H,D_e]
            sp = to_torch(S[i, 1:,  :], self.device)          # [H,D_s]
            ap = torch.cat([torch.zeros(1, D_e, device=self.device), a[:-1]], dim=0)  # [H,D_e]
            r, _ = self.comb_reward(s, a, sp, ap)             # [H,1]
            r = r.clamp(-self.cfg.reward_clip, self.cfg.reward_clip)
            scores[i] = float((-r.sum()).item())

        np.savez_compressed(out_path, scores=scores, y=y, meta=meta)
        print(f"✅ Saved anomaly scores: {out_path}")

# ----------------------------
# CLI
# ----------------------------
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--train_npz",   type=str, required=True, help="Phase-1 train npz with S,E,A (multi-agent)")
    p.add_argument("--test_npz",    type=str, default=None,  help="Phase-1 test npz with S,E,A for scoring")
    p.add_argument("--policy_ckpt", type=str, required=True, help="phase2_ckpt/policy_bc.pt")
    p.add_argument("--dyn_ckpt",    type=str, required=True, help="phase2_ckpt/dynamics.pt (Conditional E-dynamics)")
    p.add_argument("--reward_ckpt", type=str, required=True, help="phase2_ckpt/reward_init.pt")
    p.add_argument("--iters",       type=int, default=80)
    p.add_argument("--rollouts",    type=int, default=256)
    p.add_argument("--horizon",     type=int, default=63)
    p.add_argument("--tau",         type=float, default=5.0)
    p.add_argument("--alpha",       type=float, default=0.01)
    p.add_argument("--lr_reward",   type=float, default=1e-3)
    p.add_argument("--lr_policy",   type=float, default=5e-4)
    p.add_argument("--update_dynamics", action="store_true")
    p.add_argument("--out_dir",     type=str, default="phase3_ckpt")
    args = p.parse_args()

    cfg = IRLCfg(
        iters=args.iters,
        rollouts_per_iter=args.rollouts,
        horizon=args.horizon,
        temperature=args.tau,
        alpha_entropy=args.alpha,
        lr_reward=args.lr_reward,
        lr_policy=args.lr_policy,
        update_dynamics=args.update_dynamics
    )

    trainer = MaxEntIRLTrainer(
        train_npz=args.train_npz,
        policy_ckpt=args.policy_ckpt,
        dyn_ckpt=args.dyn_ckpt,
        reward_ckpt=args.reward_ckpt,
        use_handcrafted=True,
        cfg=cfg
    )
    trainer.train(out_dir=args.out_dir)

    if args.test_npz:
        out_scores = os.path.join(args.out_dir, "scores_test.npz")
        trainer.score_windows(args.test_npz, out_scores)
