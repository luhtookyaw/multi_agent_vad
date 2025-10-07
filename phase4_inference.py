# phase4_inference.py (multi-agent)
import os, math, json, re, argparse
from typing import Dict, Any, Tuple, Optional, List

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

# ===========================
# Small utilities
# ===========================

def set_seed(seed: int = 42):
    import random
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def safe_numpy(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()

def extract_camera(meta_i) -> str:
    """
    Try hard to get a camera/group key from meta (dict or string).
    Falls back to 'all' if unknown.
    """
    try:
        if isinstance(meta_i, dict):
            for k in ["camera", "cam", "camera_id", "cam_id", "c"]:
                if k in meta_i: return str(meta_i[k])
            # try path-like fields
            for k in ["path", "file", "video", "vid", "key"]:
                if k in meta_i and isinstance(meta_i[k], str):
                    m = re.search(r"/(c\d+|csc)/", meta_i[k])
                    if m: return m.group(1)
        elif isinstance(meta_i, str):
            m = re.search(r"/(c\d+|csc)/", meta_i)
            if m: return m.group(1)
    except Exception:
        pass
    return "all"

# ===========================
# Modules (mirror Phase-3)
# ===========================

class RewardFeaturesMLP(nn.Module):
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
        return self.ln(self.f(torch.cat([s,a], dim=-1)))

class RewardLinearHead(nn.Module):
    def __init__(self, feat_dim:int):
        super().__init__()
        self.theta = nn.Parameter(torch.zeros(feat_dim))
    def forward(self, phi):
        return (phi * self.theta).sum(dim=-1, keepdim=True)

class RewardModel(nn.Module):
    def __init__(self, state_dim:int, act_dim:int, feat_dim:int=64):
        super().__init__()
        self.feat = RewardFeaturesMLP(state_dim, act_dim, feat_dim)
        self.head = RewardLinearHead(feat_dim)
    def forward(self, s, a, return_phi=False):
        phi = self.feat(s,a)
        r = self.head(phi)
        return (r, phi) if return_phi else r

class CombinedReward(nn.Module):
    """
    r = <theta_all, concat( φ_ψ(s,a), hand(s,a,s',a_prev) )>
    hand features: [s, a, ||a||, ||a||^2, cos(s,sp), ||sp-s||, ||a-a_prev||]
    """
    def __init__(self, reward_model: RewardModel, state_dim:int, act_dim:int):
        super().__init__()
        self.reward = reward_model
        self.F_learn = reward_model.head.theta.numel()
        self.F_hand  = state_dim + act_dim + 5
        self.theta_all = nn.Parameter(torch.zeros(self.F_learn + self.F_hand))

    @staticmethod
    def _hand_features(s, a, sp, a_prev, eps=1e-6):
        a_norm  = torch.linalg.norm(a,  dim=-1, keepdim=True)
        a_norm2 = (a**2).sum(dim=-1, keepdim=True)
        cos = (s*sp).sum(-1, keepdim=True) / (
            torch.linalg.norm(s, dim=-1, keepdim=True) * torch.linalg.norm(sp, dim=-1, keepdim=True) + eps
        )
        speed = torch.linalg.norm(sp - s, dim=-1, keepdim=True)
        smooth= torch.linalg.norm(a - a_prev, dim=-1, keepdim=True)
        return torch.cat([s, a, a_norm, a_norm2, cos, speed, smooth], dim=-1)

    def forward(self, s, a, sp, a_prev):
        phi_learn = self.reward.feat(s, a)
        phi_hand  = self._hand_features(s, a, sp, a_prev)
        phi_all   = torch.cat([phi_learn, phi_hand], dim=-1)
        r = (phi_all * self.theta_all).sum(dim=-1, keepdim=True)
        return r, phi_all

class ConditionalEDynamics(nn.Module):
    """
    p(e'|s,a) = N( e + a + res([s,a]), diag(σ_e^2) )
    - s = [e ; g], predict next embedding only.
    """
    def __init__(self, state_dim:int, act_dim:int, embed_dim:int,
                 hidden:int=64, init_log_sigma:float=-2.0,
                 learn_residual:bool=True, learn_sigma:bool=True):
        super().__init__()
        self.state_dim = state_dim
        self.act_dim   = act_dim
        self.embed_dim = embed_dim
        self.learn_residual = learn_residual
        self.learn_sigma    = learn_sigma
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
        return base if self.res is None else base + self.res(torch.cat([s, a], dim=-1))

# -- Loader for Phase-3 (multi-agent) ckpt
def load_phase3_ckpt(ckpt_path: str, device: str = None):
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_path, map_location="cpu")

    D_s = int(ckpt["D_s"]); D_e = int(ckpt["D_e"]); D_g = int(ckpt.get("D_g", max(0, D_s - D_e)))
    comb_state = ckpt["comb_reward"]
    dyn_state  = ckpt["dynamics"]
    cfg        = ckpt.get("cfg", {})

    # infer learned feature dim from theta size
    F_learn = comb_state["reward.head.theta"].numel()

    reward = RewardModel(D_s, D_e, feat_dim=F_learn)
    comb   = CombinedReward(reward, state_dim=D_s, act_dim=D_e)
    comb.load_state_dict(comb_state)
    comb.to(dev).eval()

    dyn = ConditionalEDynamics(state_dim=D_s, act_dim=D_e, embed_dim=D_e)
    dyn.load_state_dict(dyn_state)
    dyn.to(dev).eval()

    return comb, dyn, D_s, D_e, D_g, cfg, dev

# ===========================
# Scoring (window → scalar)
# ===========================

@torch.no_grad()
def per_step_costs_multi(
    comb: CombinedReward,
    dyn: Optional[ConditionalEDynamics],
    S: np.ndarray,  # [T, D_s]  (S = [E ; G])
    E: np.ndarray,  # [T, D_e]
    A: np.ndarray,  # [T-1, D_e]
    lam_dyn: float = 0.0,
    reward_clip: float = 10.0,
    device: str = "cpu",
    reduce_nll: str = "mean",    # 'mean' or 'sum'
) -> np.ndarray:
    """
    Returns c_t for t=0..T-2 as numpy array [T-1]
    c_t = -r_theta(s_t, a_t) [+ lam_dyn * NLL_dyn(e_{t+1}|s_t,a_t)]
    """
    T, D_s = S.shape
    H = T - 1
    D_e = E.shape[-1]

    s  = torch.from_numpy(S[:-1]).float().to(device)  # [H,D_s]
    a  = torch.from_numpy(A).float().to(device)       # [H,D_e]
    sp = torch.from_numpy(S[1:]).float().to(device)   # [H,D_s]
    ap = torch.cat([torch.zeros(1, D_e, device=device), a[:-1]], dim=0)

    r, _ = comb(s, a, sp, ap)                         # [H,1]
    r = r.clamp(-reward_clip, reward_clip)
    cost = (-r).squeeze(-1)                           # [H]

    if dyn is not None and lam_dyn > 0:
        e_next = torch.from_numpy(E[1:]).float().to(device)  # [H,D_e]
        mu_e   = dyn.mean_e(s, a)
        log_var = 2 * dyn.log_sigma_e
        var = torch.exp(log_var)
        nll = 0.5 * (((e_next - mu_e)**2)/var + log_var + math.log(2*math.pi))  # [H,D_e]
        nll = nll.mean(dim=-1) if reduce_nll == "mean" else nll.sum(dim=-1)
        cost = cost + lam_dyn * nll

    return safe_numpy(cost)

def window_score_from_costs(costs: np.ndarray, topk: float = 0.10) -> float:
    """
    Score = mean of top-k% costs (peak anomaly focus).
    """
    H = len(costs)
    k = max(1, int(np.ceil(topk * H)))
    idx = np.argpartition(costs, -k)[-k:]
    return float(costs[idx].mean())

# ===========================
# Evaluation helpers
# ===========================

def pr_roc(scores: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    out = {"ap": float("nan"), "roc_auc": float("nan")}
    try:
        from sklearn.metrics import average_precision_score, roc_auc_score
        mask = (labels == 0) | (labels == 1)
        if mask.sum() >= 2 and np.unique(labels[mask]).size > 1:
            out["ap"] = float(average_precision_score(labels[mask], scores[mask]))
            out["roc_auc"] = float(roc_auc_score(labels[mask], scores[mask]))
    except Exception as e:
        print(f"[warn] sklearn not available or failed: {e}")
    return out

def per_camera_thresholds(
    scores: np.ndarray,
    labels: Optional[np.ndarray],
    cameras: List[str],
    method: str = "quantile",   # 'quantile' or 'f1'
    q: float = 0.95
) -> Dict[str, float]:
    """
    Returns {camera: threshold}
    - 'quantile': threshold = q-quantile of normal (label==0) scores for that camera
    - 'f1':       threshold maximizing F1 on that camera (requires labels in {0,1})
    """
    cams = np.array(cameras)
    uniq = np.unique(cams)
    th = {}
    for c in uniq:
        m = (cams == c)
        sc = scores[m]
        if method == "quantile":
            if labels is not None:
                lab = labels[m]
                sc = sc[(lab == 0)] if np.any(lab == 0) else sc
            th[c] = float(np.quantile(sc, q)) if sc.size else float("inf")
        else:  # F1 sweep
            if labels is None:
                th[c] = float(np.quantile(sc, q)) if sc.size else float("inf")
                continue
            lab = labels[m]
            mask = (lab == 0) | (lab == 1)
            sc = sc[mask]; lab = lab[mask]
            if sc.size == 0 or np.unique(lab).size < 2:
                th[c] = float(np.quantile(sc, q)) if sc.size else float("inf")
                continue
            best_f1, best_t = -1.0, float("inf")
            for t in np.unique(sc):
                pred = (sc >= t).astype(np.int32)
                tp = int(((pred == 1) & (lab == 1)).sum())
                fp = int(((pred == 1) & (lab == 0)).sum())
                fn = int(((pred == 0) & (lab == 1)).sum())
                prec = tp / max(1, tp + fp)
                rec  = tp / max(1, tp + fn)
                f1 = 2*prec*rec/max(1e-12, (prec+rec)) if (prec+rec) > 0 else 0.0
                if f1 > best_f1:
                    best_f1, best_t = f1, float(t)
            th[c] = best_t
    return th

def apply_thresholds(scores: np.ndarray, cameras: List[str], th: Dict[str, float]) -> np.ndarray:
    pred = np.zeros_like(scores, dtype=np.int32)
    for i, c in enumerate(cameras):
        t = th.get(c, th.get("all", np.inf))
        pred[i] = 1 if scores[i] >= t else 0
    return pred

def basic_report(scores: np.ndarray, labels: Optional[np.ndarray], cameras: List[str]) -> Dict[str, Any]:
    rep = {"global": {}, "per_camera": {}}
    rep["global"].update(pr_roc(scores, labels if labels is not None else np.zeros_like(scores)))
    cams = np.array(cameras)
    for c in np.unique(cams):
        idx = (cams == c)
        rep["per_camera"][c] = pr_roc(scores[idx], labels[idx] if labels is not None else np.zeros(idx.sum()))
    return rep

# ===========================
# Main scorer (multi-agent)
# ===========================

def score_npz(
    ckpt_path: str,
    npz_path: str,
    out_path: str,
    topk: float = 0.10,
    lam_dyn: float = 0.0,
    reward_clip: float = 10.0,
    reduce_nll: str = "mean",
    thresh_method: str = "quantile",  # 'quantile' or 'f1'
    thresh_q: float = 0.95,
    seed: int = 42
):
    """
    Produces per-window anomaly scores and (optionally) predictions via per-camera thresholds.
    Saves results to NPZ + JSON (report).
    """
    set_seed(seed)
    comb, dyn, D_s, D_e, D_g, cfg, device = load_phase3_ckpt(ckpt_path)

    dat = np.load(npz_path, allow_pickle=True)
    if 'S' in dat.files:
        S = dat['S']        # [N,T,D_s]
    else:
        # fall back to single-agent style: S == E
        S = dat['E']
    E = dat['E']            # [N,T,D_e]
    A = dat['A']            # [N,T-1,D_e]
    y = dat.get("y", None)
    meta = dat.get("meta", None)

    N, T, Ds_npz = S.shape
    De_npz = E.shape[-1]
    assert Ds_npz == D_s, f"State dim mismatch: NPZ S={Ds_npz} vs ckpt D_s={D_s}"
    assert De_npz == D_e, f"Embed dim mismatch: NPZ E={De_npz} vs ckpt D_e={D_e}"

    cameras = [extract_camera(m) for m in meta] if meta is not None else ["all"] * N

    scores = np.zeros((N,), dtype=np.float32)
    topk_used = []
    for i in tqdm(range(N), desc="Scoring windows"):
        c_t = per_step_costs_multi(
            comb=comb, dyn=dyn if lam_dyn > 0 else None,
            S=S[i], E=E[i], A=A[i],
            lam_dyn=lam_dyn, reward_clip=reward_clip,
            device=device, reduce_nll=reduce_nll
        )
        s = window_score_from_costs(c_t, topk=topk)
        scores[i] = s
        topk_used.append(int(max(1, int(np.ceil(topk * (T-1))))))

    # global & per-camera PR/ROC
    report = basic_report(scores, y if y is not None else None, cameras)

    # per-camera thresholds
    th = per_camera_thresholds(scores, y if y is not None else None, cameras, method=thresh_method, q=thresh_q)
    preds = apply_thresholds(scores, cameras, th)

    # compute per-camera precision/recall/F1 if labels exist
    per_cam_stats = {}
    if y is not None:
        cams = np.array(cameras)
        for c in np.unique(cams):
            m = (cams == c)
            yt = y[m]
            pr = preds[m]
            tp = int(((pr == 1) & (yt == 1)).sum())
            fp = int(((pr == 1) & (yt == 0)).sum())
            tn = int(((pr == 0) & (yt == 0)).sum())
            fn = int(((pr == 0) & (yt == 1)).sum())
            prec = tp / max(1, tp + fp)
            rec  = tp / max(1, tp + fn)
            f1   = 2*prec*rec/max(1e-12, (prec+rec)) if (prec+rec) > 0 else 0.0
            per_cam_stats[c] = dict(tp=tp, fp=fp, tn=tn, fn=fn, precision=prec, recall=rec, f1=f1)

    # save outputs
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    np.savez_compressed(out_path,
                        scores=scores,
                        preds=preds,
                        y=y if y is not None else np.array([]),
                        cameras=np.array(cameras, dtype=object),
                        topk=np.array(topk_used))
    with open(out_path.replace(".npz", "_report.json"), "w") as f:
        json.dump({
            "ckpt": ckpt_path,
            "npz":  npz_path,
            "N": int(N), "T": int(T),
            "D_s": int(D_s), "D_e": int(D_e), "D_g": int(D_g),
            "topk": topk, "lam_dyn": lam_dyn, "reduce_nll": reduce_nll,
            "thresholds": th,
            "report": report,
            "per_camera_stats": per_cam_stats
        }, f, indent=2)

    print(f"✅ Saved scores to {out_path}")
    print(f"✅ Saved report to {out_path.replace('.npz','_report.json')}")

# ===========================
# CLI
# ===========================
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt",      type=str, required=True, help="phase3_ckpt/final.pt (or iter_XXX.pt)")
    ap.add_argument("--npz",       type=str, required=True, help="Phase-1 NPZ with S,E,A to score (multi-agent)")
    ap.add_argument("--out",       type=str, required=True, help="Output NPZ path for scores/preds")
    ap.add_argument("--topk",      type=float, default=0.10, help="Top-k fraction for pooling (0.1=top10%)")
    ap.add_argument("--lam_dyn",   type=float, default=0.0,  help="Weight for dynamics surprise term")
    ap.add_argument("--reward_clip", type=float, default=10.0)
    ap.add_argument("--reduce_nll",  type=str,   default="mean", choices=["mean","sum"])
    ap.add_argument("--thresh_method", type=str, default="quantile", choices=["quantile","f1"])
    ap.add_argument("--thresh_q",     type=float, default=0.95)
    ap.add_argument("--seed",         type=int,   default=42)
    args = ap.parse_args()

    score_npz(
        ckpt_path=args.ckpt,
        npz_path=args.npz,
        out_path=args.out,
        topk=args.topk,
        lam_dyn=args.lam_dyn,
        reward_clip=args.reward_clip,
        reduce_nll=args.reduce_nll,
        thresh_method=args.thresh_method,
        thresh_q=args.thresh_q,
        seed=args.seed
    )
