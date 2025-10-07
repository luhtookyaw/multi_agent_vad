# phase2_irl_models.py (multi-agent aware)
import os, json, math
from typing import Tuple, Dict, Any
from dataclasses import dataclass

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

# ============================================================
# 0) Dataset from Phase-1 outputs (S, E, A)  -- multi-agent
#    S: [N, T, D_s]  (state = [E ; G] if G exists, else E)
#    E: [N, T, D_e]  (embedding)
#    A: [N, T-1, D_e] (actions in embedding space)
# ============================================================

class EmbActPairsMulti(Dataset):
    """
    Builds (s_t, a_t, s_{t+1}, e_{t+1}) tuples from Phase-1 NPZ:
      S: [N,T,D_s]  full state (embedding + social features if available)
      E: [N,T,D_e]  embedding only
      A: [N,T-1,D_e] action = e_{t+1} - e_t
      y: [N]        labels (0 normal, 1 anomaly, -1 unknown)

    For policy/dynamics warm-start we usually filter to y==0.
    Dynamics is *conditional on neighbors*: we model only e_{t+1}
    as N(mean(s,a), diag(σ^2)); social part is treated exogenous.
    """
    def __init__(self, npz_path: str, only_label_zero: bool = True):
        dat = np.load(npz_path, allow_pickle=True)

        # Backward compatibility: if S not present, use E as S.
        if 'S' in dat.files:
            self.S = dat['S']  # [N,T,D_s]
        else:
            self.S = dat['E']  # [N,T,D_e] (single-agent fallback)

        self.E = dat['E']     # [N,T,D_e]
        self.A = dat['A']     # [N,T-1,D_e]
        self.y = dat['y']     # [N]
        self.meta = dat['meta']

        # Filter to normal windows if requested
        if only_label_zero and (self.y is not None):
            keep = (self.y == 0)
            self.S = self.S[keep]
            self.E = self.E[keep]
            self.A = self.A[keep]
            self.y = self.y[keep]
            self.meta = self.meta[keep]

        N, T, self.D_s = self.S.shape
        _, _, self.D_e = self.E.shape
        assert self.A.shape == (N, T-1, self.D_e), "E and A shapes are misaligned"

        # Flatten time across windows
        self.s  = self.S[:, :-1, :].reshape(-1, self.D_s)   # [M, D_s]
        self.a  = self.A.reshape(-1, self.D_e)              # [M, D_e]
        self.sp = self.S[:,  1:, :].reshape(-1, self.D_s)   # [M, D_s]
        self.e_next = self.E[:, 1:, :].reshape(-1, self.D_e)# [M, D_e]
        self.M = self.s.shape[0]

    def __len__(self): return self.M

    def __getitem__(self, i):
        # Return e_{t+1} for dynamics; sp is available for completeness
        return (
            torch.from_numpy(self.s[i]).float(),       # s_t
            torch.from_numpy(self.a[i]).float(),       # a_t
            torch.from_numpy(self.sp[i]).float(),      # s_{t+1} (unused by dyn loss)
            torch.from_numpy(self.e_next[i]).float()   # e_{t+1}
        )

# ============================================================
# 1) Models: Reward (linear on learned features), Policy, Dynamics
# ============================================================

class RewardFeaturesMLP(nn.Module):
    """
    φ_ψ(s,a) = MLP_ψ([s,a])  -> ℝ^F
    Keep reward linear on top: r_θ = θ^T φ_ψ.
    """
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
        f = self.f(z)
        return self.ln(f)

class RewardLinearHead(nn.Module):
    """ r_θ = θ^T φ   (no bias) """
    def __init__(self, feat_dim:int):
        super().__init__()
        self.theta = nn.Parameter(torch.zeros(feat_dim))

    def forward(self, phi):
        return (phi * self.theta).sum(dim=-1, keepdim=True)  # [B,1]

class RewardModel(nn.Module):
    """
    Full reward: r(s,a) = <θ, φ_ψ(s,a)>
    """
    def __init__(self, state_dim:int, act_dim:int, feat_dim:int=64):
        super().__init__()
        self.feat = RewardFeaturesMLP(state_dim, act_dim, feat_dim)
        self.head = RewardLinearHead(feat_dim)

    def forward(self, s, a, return_phi=False):
        phi = self.feat(s, a)     # [B,F]
        r = self.head(phi)        # [B,1]
        return (r, phi) if return_phi else r

class GaussianPolicy(nn.Module):
    """
    π_ω(a|s) = N( μ_ω(s), diag(σ^2) ), with learned state-independent log_std
    - Input:  s ∈ ℝ^{D_s}
    - Output: a ∈ ℝ^{D_e}
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

    def nll(self, s, a):
        mu, log_std = self.forward(s)
        var = torch.exp(2*log_std)
        nll = 0.5 * ((a - mu)**2 / var + 2*log_std + math.log(2*math.pi))
        return nll.mean()

    def sample(self, s):
        mu, log_std = self.forward(s)
        std = torch.exp(log_std)
        eps = torch.randn_like(mu)
        return mu + std * eps

class ConditionalEDynamics(nn.Module):
    """
    p_ψ(e'|s,a) = N( mean = e + a + g_ψ([s,a]), diag(σ_e^2) )
      - We model only the embedding next-state, treating social G as exogenous.
      - 's' contains [e ; g]; we extract the first D_e dims as e.

    Args:
      state_dim: D_s
      act_dim:   D_e  (action in embedding space)
      embed_dim: D_e  (first D_e dims of state)
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
        # s: [B,D_s] ; a: [B,D_e]; e = s[..., :D_e]
        e = s[..., :self.embed_dim]
        base = e + a
        if self.res is None:
            return base
        return base + self.res(torch.cat([s, a], dim=-1))

    def nll(self, s, a, e_next):
        mu_e = self.mean_e(s, a)
        log_var = 2 * self.log_sigma_e
        var = torch.exp(log_var)
        nll = 0.5 * ((e_next - mu_e)**2 / var + log_var + math.log(2*math.pi))
        return nll.mean()

    def sample_e(self, s, a):
        mu_e = self.mean_e(s, a)
        std = torch.exp(self.log_sigma_e)
        return mu_e + std * torch.randn_like(mu_e)

# ============================================================
# 2) Training utilities (BC for policy; NLL for dynamics)
# ============================================================

@dataclass
class TrainCfg:
    batch_size: int = 1024
    lr: float = 1e-3
    weight_decay: float = 1e-6
    epochs: int = 30
    val_ratio: float = 0.1
    num_workers: int = 4
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42

def set_seed(seed:int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def split_dataset(ds: Dataset, val_ratio: float):
    n = len(ds)
    n_val = int(n * val_ratio)
    n_tr = n - n_val
    return random_split(ds, [n_tr, n_val], generator=torch.Generator().manual_seed(123))

def train_policy_bc(npz_path: str, cfg:TrainCfg=TrainCfg(), out_dir:str="phase2_ckpt") -> str:
    """
    Behavior cloning on (s_t -> a_t) with a Gaussian policy in embedding action space.
    """
    set_seed(cfg.seed); os.makedirs(out_dir, exist_ok=True)
    ds = EmbActPairsMulti(npz_path, only_label_zero=True)
    tr, va = split_dataset(ds, cfg.val_ratio)
    dl_tr = DataLoader(tr, batch_size=cfg.batch_size, shuffle=True,  num_workers=cfg.num_workers, pin_memory=True)
    dl_va = DataLoader(va, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=True)

    state_dim = ds.D_s
    act_dim   = ds.D_e

    policy = GaussianPolicy(state_dim, act_dim).to(cfg.device)
    opt = torch.optim.Adam(policy.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    best = math.inf; best_path = os.path.join(out_dir, "policy_bc.pt")
    for ep in range(1, cfg.epochs+1):
        policy.train(); tr_loss=0; n=0
        for batch in dl_tr:
            s,a,_,_ = batch
            s,a = s.to(cfg.device), a.to(cfg.device)
            loss = policy.nll(s,a)
            opt.zero_grad(); loss.backward(); opt.step()
            tr_loss += loss.item() * s.size(0); n += s.size(0)
        tr_loss /= max(1,n)

        policy.eval(); va_loss=0; m=0
        with torch.no_grad():
            for batch in dl_va:
                s,a,_,_ = batch
                s,a = s.to(cfg.device), a.to(cfg.device)
                loss = policy.nll(s,a)
                va_loss += loss.item() * s.size(0); m += s.size(0)
        va_loss /= max(1,m)

        print(f"[Policy-BC] ep {ep:03d} | train {tr_loss:.6f} | val {va_loss:.6f}")
        if va_loss < best:
            best = va_loss
            torch.save({"state_dim":state_dim, "act_dim":act_dim,
                        "model": policy.state_dict(),
                        "cfg": cfg.__dict__},
                       best_path)

    print(f"✅ Saved policy: {best_path}")
    return best_path

def train_dynamics(npz_path: str, cfg:TrainCfg=TrainCfg(), out_dir:str="phase2_ckpt",
                   learn_residual:bool=True, learn_sigma:bool=True, init_log_sigma:float=-2.0) -> str:
    """
    Train conditional embedding dynamics p(e'|s,a).
    """
    set_seed(cfg.seed); os.makedirs(out_dir, exist_ok=True)
    ds = EmbActPairsMulti(npz_path, only_label_zero=True)
    tr, va = split_dataset(ds, cfg.val_ratio)
    dl_tr = DataLoader(tr, batch_size=cfg.batch_size, shuffle=True,  num_workers=cfg.num_workers, pin_memory=True)
    dl_va = DataLoader(va, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=True)

    state_dim = ds.D_s
    act_dim   = ds.D_e
    embed_dim = ds.D_e

    dyn = ConditionalEDynamics(
        state_dim, act_dim, embed_dim,
        learn_residual=learn_residual,
        learn_sigma=learn_sigma,
        init_log_sigma=init_log_sigma
    ).to(cfg.device)

    opt = torch.optim.Adam(dyn.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    best = math.inf; best_path = os.path.join(out_dir, "dynamics.pt")
    for ep in range(1, cfg.epochs+1):
        dyn.train(); tr_loss=0; n=0
        for batch in dl_tr:
            s,a,_,e_next = batch
            s,a,e_next = s.to(cfg.device), a.to(cfg.device), e_next.to(cfg.device)
            loss = dyn.nll(s,a,e_next)
            opt.zero_grad(); loss.backward(); opt.step()
            tr_loss += loss.item() * s.size(0); n += s.size(0)
        tr_loss /= max(1,n)

        dyn.eval(); va_loss=0; m=0
        with torch.no_grad():
            for batch in dl_va:
                s,a,_,e_next = batch
                s,a,e_next = s.to(cfg.device), a.to(cfg.device), e_next.to(cfg.device)
                loss = dyn.nll(s,a,e_next)
                va_loss += loss.item() * s.size(0); m += s.size(0)
        va_loss /= max(1,m)

        print(f"[Dynamics] ep {ep:03d} | train {tr_loss:.6f} | val {va_loss:.6f}")
        if va_loss < best:
            best = va_loss
            torch.save({"state_dim":state_dim, "act_dim":act_dim, "embed_dim":embed_dim,
                        "model": dyn.state_dict(),
                        "cfg": cfg.__dict__,
                        "learn_residual": learn_residual,
                        "learn_sigma": learn_sigma},
                       best_path)

    print(f"✅ Saved dynamics: {best_path}")
    return best_path

# ============================================================
# 3) Reward model init (no training yet—Phase-3 will update θ, ψ)
# ============================================================

def init_reward(npz_path: str, feat_dim:int=64, out_dir:str="phase2_ckpt") -> str:
    """
    Initialize reward model for multi-agent: r(s,a) = θ^T φ_ψ([s,a]).
    Uses state_dim = dim(S), act_dim = dim(E).
    """
    os.makedirs(out_dir, exist_ok=True)
    dat = np.load(npz_path, allow_pickle=True)
    if 'S' in dat.files:
        D_s = dat['S'].shape[-1]
    else:
        D_s = dat['E'].shape[-1]  # fallback
    D_e = dat['E'].shape[-1]

    reward = RewardModel(state_dim=D_s, act_dim=D_e, feat_dim=feat_dim)
    reward_path = os.path.join(out_dir, "reward_init.pt")
    torch.save({"state_dim":D_s, "act_dim":D_e, "feat_dim":feat_dim,
                "model": reward.state_dict()},
               reward_path)
    print(f"✅ Initialized reward (linear-on-MLP features): {reward_path}")
    return reward_path

# ============================================================
# 4) CLI
# ============================================================

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--train_npz", type=str, required=True,
                   help="Phase-1 output (emb_act_state_train_*.npz) with S,E,A")
    p.add_argument("--out_dir",   type=str, default="phase2_ckpt")
    p.add_argument("--epochs",    type=int, default=30)
    p.add_argument("--batch",     type=int, default=1024)
    p.add_argument("--lr",        type=float, default=1e-3)
    p.add_argument("--val_ratio", type=float, default=0.1)
    p.add_argument("--init_log_sigma", type=float, default=-2.0)
    p.add_argument("--no_dyn_residual", action="store_true",
                   help="Disable residual g_ψ; mean = e + a")
    p.add_argument("--no_dyn_learn_sigma", action="store_true",
                   help="Freeze σ_e (use init_log_sigma)")
    p.add_argument("--feat_dim", type=int, default=64)
    args = p.parse_args()

    cfg = TrainCfg(
        batch_size=args.batch,
        lr=args.lr,
        epochs=args.epochs,
        val_ratio=args.val_ratio
    )

    # 1) Policy warm-start (behavior cloning) on (s -> a)
    policy_ckpt = train_policy_bc(args.train_npz, cfg=cfg, out_dir=args.out_dir)

    # 2) Dynamics training: p(e'|s,a)
    dyn_ckpt = train_dynamics(
        args.train_npz, cfg=cfg, out_dir=args.out_dir,
        learn_residual=not args.no_dyn_residual,
        learn_sigma=not args.no_dyn_learn_sigma,
        init_log_sigma=args.init_log_sigma
    )

    # 3) Initialize reward (linear-on-features)
    reward_ckpt = init_reward(args.train_npz, feat_dim=args.feat_dim, out_dir=args.out_dir)

    print("\n=== Phase-2 (multi-agent) summary ===")
    print("Policy:", policy_ckpt)
    print("Dynamics:", dyn_ckpt)
    print("Reward init:", reward_ckpt)
