# realtime_irl_multi.py
import os, math, time, argparse, collections
from dataclasses import dataclass
from typing import Dict, Deque, Tuple, Optional, List

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics import YOLO

# ===========================
# Small utilities
# ===========================
def set_seed(seed=42):
    import random
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def to_torch(x, device):
    return torch.from_numpy(x).float().to(device)

def as_numpy(x: torch.Tensor):
    return x.detach().cpu().numpy()

def l2norm(v, eps=1e-6):
    n = np.linalg.norm(v)
    return v/(n+eps)

# HuVAD uses (y,x,c); YOLOv8 pose yields (x,y,conf). For LIVE inference we already have (x,y,conf).
# We'll normalize by bbox center/height per-frame.
def normalize_xyc(kpts_xyc: np.ndarray, box_xyxy: np.ndarray) -> np.ndarray:
    """
    kpts_xyc: [V,3] with columns (x,y,conf)
    box_xyxy: [4] (x1,y1,x2,y2)
    returns normed [V,3] with (x',y',conf), where x'=(x-cx)/h, y'=(y-cy)/h
    """
    x1,y1,x2,y2 = box_xyxy
    cx, cy = (x1+x2)/2.0, (y1+y2)/2.0
    h = max(1.0, (y2-y1))
    out = kpts_xyc.copy()
    out[:,0] = (out[:,0] - cx)/h
    out[:,1] = (out[:,1] - cy)/h
    return out

# ===========================
# ST-GCN (causal, lightweight reference)
# ===========================
class SimpleCausalSTGCN(nn.Module):
    """
    Minimal causal temporal encoder: (B,C,T,V,M)->(B,T,D)
    - C = 2 or 3 (x,y[,conf])
    - V = 17 joints
    - M = 1 person (we already isolate per-track)
    This is a *reference* encoder for inference. Replace with your Phase-1 encoder if needed.
    """
    def __init__(self, in_channels=2, V=17, D=64):
        super().__init__()
        self.V = V
        self.D = D
        hidden = 128
        self.proj = nn.Conv1d(in_channels*V, hidden, kernel_size=1)
        self.temporal1 = nn.Conv1d(hidden, hidden, kernel_size=5, padding=4, dilation=1)
        self.temporal2 = nn.Conv1d(hidden, hidden, kernel_size=5, padding=8, dilation=2)
        self.temporal3 = nn.Conv1d(hidden, hidden, kernel_size=5, padding=16, dilation=4)
        self.head = nn.Conv1d(hidden, D, kernel_size=1)
        self.act = nn.ELU(inplace=True)
        self.ln = nn.GroupNorm(8, hidden)

    def forward(self, x):
        """
        x shape: (B,C,T,V,M) with M=1
        returns: (B,T,D) per-frame embedding, causal (depends only on <=t)
        """
        B,C,T,V,M = x.shape
        assert M == 1 and V == self.V
        x = x[...,0]                     # (B,C,T,V)
        x = x.permute(0,3,1,2)           # (B,V,C,T)
        x = x.reshape(B, V*C, T)         # (B,VC,T)

        h = self.proj(x)                 # (B,H,T)
        h = self.act(h)

        def causal_conv(conv, h):
            pad = conv.padding[0]
            y = conv(h)
            if pad > 0:
                y = y[:, :, :-(pad)] if y.size(-1) > pad else y*0
            return y

        h = self.ln(causal_conv(self.temporal1, h)); h = self.act(h)
        h = self.ln(causal_conv(self.temporal2, h)); h = self.act(h)
        h = self.ln(causal_conv(self.temporal3, h)); h = self.act(h)
        e = self.head(h)                  # (B,D,T)
        e = e.permute(0,2,1)             # (B,T,D)
        return e

def load_encoder(encoder_ckpt: str, in_channels=2, V=17, D=64, device="cpu"):
    """
    Tries to load:
      1) TorchScript module (jit) -> must output (B,T,D)
      2) State dict into SimpleCausalSTGCN with matching D
    Replace this with your Phase-1 encoder loader if you have a custom model.
    """
    try:
        enc = torch.jit.load(encoder_ckpt, map_location="cpu")
        enc.eval().to(device)
        return enc, D
    except Exception:
        enc = SimpleCausalSTGCN(in_channels=in_channels, V=V, D=D)
        if os.path.isfile(encoder_ckpt):
            sd = torch.load(encoder_ckpt, map_location="cpu")
            if isinstance(sd, dict) and "model" in sd:
                sd = sd["model"]
            try:
                enc.load_state_dict(sd, strict=False)
            except Exception:
                pass
        enc.eval().to(device)
        return enc, D

# ===========================
# Reward / Dynamics (multi-agent loader like Phase-4)
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

def load_phase3_ckpt(ckpt_path: str, device: str):
    ckpt = torch.load(ckpt_path, map_location="cpu")

    # Multi-agent dims if present; fall back if not.
    D_s = int(ckpt.get("D_s", ckpt.get("D", 64)))
    D_e = int(ckpt.get("D_e", D_s))     # if single-agent, D_e == D_s
    D_g = int(ckpt.get("D_g", max(0, D_s - D_e)))

    comb_state = ckpt["comb_reward"]
    dyn_state  = ckpt["dynamics"]

    F_learn = comb_state["reward.head.theta"].numel()
    reward = RewardModel(D_s, D_e, feat_dim=F_learn)
    comb   = CombinedReward(reward, state_dim=D_s, act_dim=D_e)
    comb.load_state_dict(comb_state)
    comb.eval().to(device)

    dyn = ConditionalEDynamics(state_dim=D_s, act_dim=D_e, embed_dim=D_e)
    dyn.load_state_dict(dyn_state)
    dyn.eval().to(device)

    return comb, dyn, D_s, D_e, D_g

# ===========================
# Social features (pairwise) & aggregation
# ===========================
# COCO17 index helpers
KP = {
    "nose":0, "leye":1, "reye":2, "lear":3, "rear":4,
    "lsho":5, "rsho":6, "lelb":7, "relb":8, "lwri":9, "rwri":10,
    "lhip":11, "rhip":12, "lknee":13, "rknee":14, "lank":15, "rank":16
}

def torso_box_from_kpts_xy(k: np.ndarray, pad: float=0.15) -> np.ndarray:
    """
    k: [17,2] pixel coordinates
    returns xyxy for a small torso box around shoulders/hips
    """
    sh = k[[KP["lsho"], KP["rsho"]], :]
    hp = k[[KP["lhip"], KP["rhip"]], :]
    allp = np.vstack([sh, hp])
    x1,y1 = allp.min(axis=0); x2,y2 = allp.max(axis=0)
    dx,dy = x2-x1, y2-y1
    x1 -= pad*dx; x2 += pad*dx; y1 -= pad*dy; y2 += pad*dy
    return np.array([x1,y1,x2,y2], dtype=np.float32)

def iou_xyxy(a: np.ndarray, b: np.ndarray, eps=1e-6) -> float:
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    inter = max(0.0, x2-x1) * max(0.0, y2-y1)
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return float(inter / (ua + eps))

def min_pair_dist(A: np.ndarray, B: np.ndarray) -> float:
    # A:[m,2], B:[n,2]
    da = A[:,None,:] - B[None,:,:]   # [m,n,2]
    d2 = np.sum(da**2, axis=-1)      # [m,n]
    return float(np.sqrt(np.min(d2))) if d2.size else 1e6

def hands_pts(k: np.ndarray):  # [17,2] -> [2,2]
    return k[[KP["lwri"], KP["rwri"]], :]

def feet_pts(k: np.ndarray):
    return k[[KP["lank"], KP["rank"]], :]

def neck_head_pts(k: np.ndarray):
    # rough neck/head cluster: nose + shoulders mid
    neck = (k[KP["lsho"]] + k[KP["rsho"]]) / 2.0
    return np.vstack([k[KP["nose"]], neck[None,:]])

def torso_pts(k: np.ndarray):
    # hips center + shoulders center
    hc = (k[KP["lhip"]] + k[KP["rhip"]]) / 2.0
    sc = (k[KP["lsho"]] + k[KP["rsho"]]) / 2.0
    return np.vstack([hc[None,:], sc[None,:]])

@dataclass
class SocialCfg:
    K: int = 3                 # nearest neighbors
    R: float = 3.0             # radius in body-heights
    use_radius: bool = True
    beta1: float = 1.0         # attention weight on 1/distance
    beta2: float = 1.0         # attention weight on closing speed
    # feature scaling guards
    eps: float = 1e-6

def pair_features(i_box, i_h, i_c, i_v, i_head, i_kpix,
                  j_box, j_c, j_v, j_head, j_kpix, eps=1e-6) -> np.ndarray:
    """
    Compute 7D pairwise features for (i <- j)
    Returns: [d_scaled, close, relspeed, face_ij, hand2neck, foot2torso, torso_iou]
    """
    r = j_c - i_c
    d = np.linalg.norm(r) + eps
    d_scaled = d / (i_h + eps)
    vij = j_v - i_v
    close = - (r/d) @ vij                 # >0 when approaching
    face_ij = float(np.dot(i_head, r/d))  # i faces direction of j
    relspeed = float(np.linalg.norm(vij))
    hand2neck = min_pair_dist(hands_pts(i_kpix), neck_head_pts(j_kpix)) / (i_h + eps)
    foot2torso = min_pair_dist(feet_pts(i_kpix), torso_pts(j_kpix)) / (i_h + eps)
    torso_iou = iou_xyxy(torso_box_from_kpts_xy(i_kpix), torso_box_from_kpts_xy(j_kpix))
    return np.array([d_scaled, close, relspeed, face_ij, hand2neck, foot2torso, torso_iou], dtype=np.float32)

def aggregate_social(i_idx: int,
                     centers: List[np.ndarray],
                     heights: List[float],
                     vels: List[np.ndarray],
                     heads: List[np.ndarray],
                     kpix: List[np.ndarray],
                     social_cfg: SocialCfg) -> np.ndarray:
    """
    Build aggregated social vector g_i from neighbors of i.
    """
    N = len(centers)
    i_c = centers[i_idx]; i_h = heights[i_idx]; i_v = vels[i_idx]; i_head = heads[i_idx]; i_k = kpix[i_idx]

    # compute distances to others
    ds = [np.linalg.norm(centers[j]-i_c) if j!=i_idx else np.inf for j in range(N)]
    order = np.argsort(ds)
    neighs = [j for j in order if j != i_idx]

    if social_cfg.use_radius:
        neighs = [j for j in neighs if ds[j] <= social_cfg.R * i_h]
    neighs = neighs[:social_cfg.K]

    if len(neighs) == 0:
        return np.zeros((7,), dtype=np.float32)

    feats = []
    scores = []
    for j in neighs:
        f = pair_features(
            i_box=None, i_h=i_h, i_c=i_c, i_v=i_v, i_head=i_head, i_kpix=i_k,
            j_box=None, j_c=centers[j], j_v=vels[j], j_head=heads[j], j_kpix=kpix[j],
            eps=social_cfg.eps
        )
        feats.append(f)
        # attention score: beta1 * (1/d) + beta2 * closing_speed
        s = social_cfg.beta1 * (1.0 / max(ds[j], social_cfg.eps)) + social_cfg.beta2 * f[1]
        scores.append(s)

    feats = np.stack(feats, axis=0)    # [K,7]
    scores = np.array(scores, dtype=np.float32)
    # softmax weights
    w = np.exp(scores - scores.max())
    w = w / max(w.sum(), social_cfg.eps)
    g = (w[:,None] * feats).sum(axis=0)
    return g.astype(np.float32)        # [7]

# ===========================
# Live track state
# ===========================
@dataclass
class ScoreCfg:
    topk_frac: float = 0.10     # top-k pooling over trailing costs
    pool_len: int = 60          # frames kept for pooling/smoothing (~2s @30FPS)
    ema_alpha: float = 0.15     # EMA on per-frame cost
    reward_clip: float = 10.0
    lam_dyn: float = 0.0        # add dynamics surprise weight
    conf_min: float = 0.1       # drop joints below this conf (kept as last)

@dataclass
class TrackBuffers:
    kpts: Deque[np.ndarray]            # each: [V, C] (C=2 or 3) normalized
    kpix: Deque[np.ndarray]            # each: [V, 2] pixel coords (for social)
    conf: Deque[np.ndarray]            # each: [V]
    bboxes: Deque[np.ndarray]          # [4] xyxy (viz / height)
    centers: Deque[np.ndarray]         # [2] pixel center history
    e_prev: Optional[np.ndarray]       # [D_e]
    g_prev: Optional[np.ndarray]       # [D_g]
    a_prev: Optional[np.ndarray]       # [D_e]
    cost_hist: Deque[float]
    ema: float
    last_seen: int

def make_track_buffers(T:int, pool_len:int) -> TrackBuffers:
    return TrackBuffers(
        kpts=collections.deque(maxlen=T),
        kpix=collections.deque(maxlen=T),
        conf=collections.deque(maxlen=T),
        bboxes=collections.deque(maxlen=T),
        centers=collections.deque(maxlen=T),
        e_prev=None,
        g_prev=None,
        a_prev=None,
        cost_hist=collections.deque(maxlen=pool_len),
        ema=0.0,
        last_seen=0
    )

# ===========================
# Painter (skeleton + score)
# ===========================
COCO_SKELETON = [
    (5,7),(7,9), (6,8),(8,10), (5,6), (5,11),(6,12),(11,12),
    (11,13),(13,15), (12,14),(14,16), (0,1),(1,2),(2,3),(3,4),(1,5),(1,6)
]
def draw_pose(img, kpts_xyc: np.ndarray, color=(0,255,0)):
    V = kpts_xyc.shape[0]
    pts = kpts_xyc[:,:2].astype(int)
    conf = kpts_xyc[:,2] if kpts_xyc.shape[1]==3 else np.ones(V)
    for (u,v) in COCO_SKELETON:
        if u < V and v < V and conf[u]>0.1 and conf[v]>0.1:
            cv2.line(img, pts[u], pts[v], color, 2, cv2.LINE_AA)
    for i in range(V):
        if conf[i] > 0.1:
            cv2.circle(img, pts[i], 2, color, -1, lineType=cv2.LINE_AA)

def heat_color(v: float, vmin=0.0, vmax=1.0):
    x = (v - vmin) / max(1e-6, (vmax - vmin)); x = np.clip(x, 0, 1)
    return tuple(int(c) for c in cv2.applyColorMap(np.uint8([[x*255]]), cv2.COLORMAP_JET)[0,0,:3])

# ===========================
# Online loop (multi-agent)
# ===========================
def realtime_loop(
    rtsp: str,
    encoder_ckpt: str,
    phase3_ckpt: str,
    imgsz: int = 960,
    T: int = 64,
    use_conf_channel: bool = False,
    show: bool = True,
    save: Optional[str] = None,
    device_str: Optional[str] = None,
    score_cfg: ScoreCfg = ScoreCfg(),
    social_cfg: SocialCfg = SocialCfg(),
    max_tracks_per_frame: int = 20,
    ttl_frames: int = 30
):
    set_seed(42)
    device = device_str or ("cuda" if torch.cuda.is_available() else "cpu")

    # 1) Load ST-GCN encoder (Phase-1 encoder or reference)
    C = 3 if use_conf_channel else 2
    encoder, D_e = load_encoder(encoder_ckpt, in_channels=C, V=17, D=64, device=device)

    # 2) Load Phase-3 multi-agent reward & dynamics
    comb, dyn, D_s, D_e_ck, D_g = load_phase3_ckpt(phase3_ckpt, device)
    assert D_e == D_e_ck, f"Embed dim mismatch: encoder {D_e} vs Phase-3 {D_e_ck}"
    assert D_s == D_e + D_g, f"State dim mismatch: D_s={D_s} vs D_e+D_g={D_e + D_g}"

    # 3) YOLOv8 Pose with ByteTrack (Ultralytics)
    pose_model = YOLO("yolov8n-pose.pt")
    stream = pose_model.track(
        source=rtsp, imgsz=imgsz, conf=0.25, iou=0.5, device=0 if device=="cuda" else "cpu",
        tracker="bytetrack.yaml", stream=True, persist=True, verbose=False
    )

    # 4) Per-ID buffers
    tracks: Dict[int, TrackBuffers] = {}
    writer = None
    frame_idx = 0
    t0 = time.time()

    for r in stream:
        frame = r.orig_img
        H, W = frame.shape[:2]

        ids = r.boxes.id
        kpts = r.keypoints
        if (ids is None) or (kpts is None):
            cv2.putText(frame, "No people", (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,200,255), 2)
            if show: cv2.imshow("IRL Anomaly", frame)
            if save:
                if writer is None:
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(save, fourcc, 20.0, (W,H))
                writer.write(frame)
            if show and (cv2.waitKey(1) & 0xFF == 27): break
            frame_idx += 1
            continue

        ids = ids.int().cpu().numpy()  # [N]
        xyxy = r.boxes.xyxy.cpu().numpy()      # [N,4]
        kxy  = kpts.xy.cpu().numpy()           # [N,17,2]
        kcf  = (kpts.conf.cpu().numpy()
                if (hasattr(kpts, "conf") and kpts.conf is not None)
                else np.ones((kxy.shape[0], kxy.shape[1]), dtype=np.float32)) # [N,17]

        # Update last_seen counters
        for tid, tb in tracks.items():
            tb.last_seen += 1

        # Order by area (keep top-N)
        order = np.argsort((xyxy[:,2]-xyxy[:,0])*(xyxy[:,3]-xyxy[:,1]))[::-1]
        order = order[:min(len(order), max_tracks_per_frame)]

        # Ensure buffers exist & append frame-wise observations
        active_ids = []
        for j in order:
            tid = int(ids[j])
            box = xyxy[j]
            joints_xy = kxy[j]          # [17,2]
            conf_j = kcf[j]             # [17]
            cx, cy = (box[0]+box[2])/2.0, (box[1]+box[3])/2.0
            h = max(1.0, box[3]-box[1])

            k_xyc = np.concatenate([joints_xy, conf_j[:,None]], axis=1)  # [17,3]
            # carry-forward low-conf joints
            if tid in tracks and len(tracks[tid].kpix) > 0:
                last_xy = tracks[tid].kpix[-1]
                bad = (k_xyc[:,2] < score_cfg.conf_min)
                k_xyc[bad,:2] = last_xy[bad,:2]

            if tid not in tracks: tracks[tid] = make_track_buffers(T, score_cfg.pool_len)
            tb = tracks[tid]
            tb.last_seen = 0
            tb.kpix.append(k_xyc[:,:2].copy())
            tb.conf.append(k_xyc[:,2].copy())
            tb.bboxes.append(box.copy())
            tb.centers.append(np.array([cx,cy], dtype=np.float32))

            k_norm = normalize_xyc(k_xyc.copy(), box)              # [17,3]
            if not use_conf_channel:
                k_norm = k_norm[:,:2]
            tb.kpts.append(k_norm)

            active_ids.append(tid)

        # ---- Social pass: compute centers, velocities, headings for active tracks
        centers = []
        heights = []
        vels = []
        heads = []
        kpix_list = []
        tids_active = active_ids.copy()

        for tid in tids_active:
            tb = tracks[tid]
            c_now = tb.centers[-1]
            c_prev = tb.centers[-2] if len(tb.centers) >= 2 else tb.centers[-1]
            v = c_now - c_prev
            h = max(1.0, tb.bboxes[-1][3]-tb.bboxes[-1][1])
            head = l2norm(v) if np.linalg.norm(v) > 1e-4 else np.array([0.0,1.0], dtype=np.float32)
            centers.append(c_now)
            heights.append(h)
            vels.append(v)
            heads.append(head)
            kpix_list.append(tb.kpix[-1])

        # g_t for each active id
        g_curr: Dict[int, np.ndarray] = {}
        for idx, tid in enumerate(tids_active):
            g = aggregate_social(idx, centers, heights, vels, heads, kpix_list, social_cfg)  # [7]
            # If checkpoint expects a bigger D_g (e.g., you extended pair features),
            # you should ensure the same recipe was used in training. We pad/trim here defensively.
            if g.shape[0] < D_g:
                g = np.pad(g, (0, D_g - g.shape[0]), mode='constant')
            elif g.shape[0] > D_g:
                g = g[:D_g]
            g_curr[tid] = g

        # ---- Scoring pass (per-track)
        for tid in tids_active:
            tb = tracks[tid]
            # Need at least two frames to form a_{t-1}; and last e_prev,g_prev
            if len(tb.kpts) < 2:
                continue

            # Build causal window (1,C,T,V,1)
            K = len(tb.kpts); V = 17; C_in = 3 if use_conf_channel else 2
            window = np.zeros((T, V, C_in), dtype=np.float32)
            src = np.stack(tb.kpts, axis=0)  # [K,V,C]
            window[-K:] = src
            x = torch.from_numpy(window).permute(2,0,1).unsqueeze(0).unsqueeze(-1)  # (1,C,T,V,1)
            x = x.float().to(device)

            with torch.no_grad():
                e_seq = encoder(x)          # (1,T,D_e)
                e_t = e_seq[0,-1,:]         # [D_e]
                e_t_np = as_numpy(e_t)

            if tb.e_prev is None or tb.g_prev is None:
                tb.e_prev = e_t_np
                tb.g_prev = g_curr[tid]
                continue

            a_prev = e_t_np - tb.e_prev                      # [D_e]
            a_prev_prev = tb.a_prev if isinstance(tb.a_prev, np.ndarray) and tb.a_prev.shape == a_prev.shape else np.zeros_like(a_prev)

            # Build s_{t-1}, s_t
            s_im1 = np.concatenate([tb.e_prev, tb.g_prev], axis=0)   # [D_s]
            s_i   = np.concatenate([e_t_np, g_curr[tid]], axis=0)    # [D_s]

            s_im1_t = to_torch(s_im1[None,:], device)
            a_im1_t = to_torch(a_prev[None,:], device)
            s_i_t   = to_torch(s_i[None,:], device)
            a_im2_t = to_torch(a_prev_prev[None,:], device)

            with torch.no_grad():
                r_step, _ = comb(s_im1_t, a_im1_t, s_i_t, a_im2_t)  # [1,1]
                r_step = r_step.clamp(-score_cfg.reward_clip, score_cfg.reward_clip).item()
                cost = -r_step

                if score_cfg.lam_dyn > 0:
                    # dynamics surprise on embedding only: e_{t+1} | s_t, a_t
                    mu_e = dyn.mean_e(s_im1_t, a_im1_t)  # uses s_{t-1}, a_{t-1} → predicts e_t
                    log_var = 2*dyn.log_sigma_e
                    var = torch.exp(log_var)
                    e_next = to_torch(e_t_np[None,:], device)
                    nll = 0.5 * (((e_next - mu_e)**2)/var + log_var + math.log(2*math.pi))
                    nll = nll.mean().item()
                    cost += score_cfg.lam_dyn * nll

            # Update buffers
            tb.cost_hist.append(float(cost))
            tb.ema = (1 - score_cfg.ema_alpha)*tb.ema + score_cfg.ema_alpha*float(cost)
            tb.a_prev = a_prev.copy()
            tb.e_prev = e_t_np.copy()
            tb.g_prev = g_curr[tid].copy()

            # top-k pooling over trailing costs
            ch = np.array(tb.cost_hist, dtype=np.float32)
            if ch.size > 0:
                k = max(1, int(math.ceil(score_cfg.topk_frac * ch.size)))
                topk_idx = np.argpartition(ch, -k)[-k:]
                topk_mean = float(ch[topk_idx].mean())
                score = 0.3*tb.ema + 0.7*topk_mean
            else:
                score = tb.ema

            # Viz
            box = tb.bboxes[-1]
            k_last = np.concatenate([tb.kpix[-1], tb.conf[-1][:,None]], axis=1)
            clr = heat_color(score, vmin=0.0, vmax=3.0)  # tune vmax per camera
            x1,y1,x2,y2 = map(int, box)
            cv2.rectangle(frame, (x1,y1), (x2,y2), clr, 2)
            cv2.putText(frame, f"ID {tid}  S={score:.2f}", (x1, max(20,y1-8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, clr, 2)
            draw_pose(frame, k_last, color=clr)

        # Cleanup stale tracks
        drop_ids = [tid for tid,tb in tracks.items() if tb.last_seen > ttl_frames]
        for tid in drop_ids:
            tracks.pop(tid, None)

        # Compose and show
        fps = (frame_idx+1) / max(1e-6, (time.time()-t0))
        cv2.putText(frame, f"FPS: {fps:.1f}", (20,30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,255), 2)

        if show:
            cv2.imshow("IRL Anomaly (multi-agent)", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break

        if save:
            if writer is None:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(save, fourcc, 20.0, (frame.shape[1], frame.shape[0]))
            writer.write(frame)

        frame_idx += 1

    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()
    print("✅ Done.")

# ===========================
# CLI
# ===========================
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rtsp", required=True, help="RTSP URL (or video path / webcam index)")
    ap.add_argument("--encoder_ckpt", required=True, help="Path to ST-GCN encoder (TorchScript .pt or state_dict .pt)")
    ap.add_argument("--phase3_ckpt", required=True, help="Path to Phase-3 multi-agent final (e.g., phase3_ckpt/final.pt)")
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--T", type=int, default=64)
    ap.add_argument("--use_conf", action="store_true", help="Include keypoint confidence as 3rd input channel")
    ap.add_argument("--no_show", action="store_true")
    ap.add_argument("--save", type=str, default=None, help="Optional output video path")
    ap.add_argument("--device", type=str, default=None, help="'cuda' or 'cpu'")
    ap.add_argument("--topk", type=float, default=0.10)
    ap.add_argument("--pool_len", type=int, default=60)
    ap.add_argument("--ema_alpha", type=float, default=0.15)
    ap.add_argument("--lam_dyn", type=float, default=0.0)
    ap.add_argument("--reward_clip", type=float, default=10.0)
    ap.add_argument("--conf_min", type=float, default=0.1)
    ap.add_argument("--max_tracks", type=int, default=20)
    ap.add_argument("--ttl_frames", type=int, default=30)
    ap.add_argument("--K", type=int, default=3, help="Max neighbors per person")
    ap.add_argument("--R", type=float, default=3.0, help="Neighbor radius in body-heights")
    ap.add_argument("--no_radius", action="store_true", help="Use KNN only (ignore radius)")
    ap.add_argument("--beta1", type=float, default=1.0, help="Neighbor attention weight for 1/d")
    ap.add_argument("--beta2", type=float, default=1.0, help="Neighbor attention weight for closing speed")
    args = ap.parse_args()

    sc = ScoreCfg(
        topk_frac=args.topk,
        pool_len=args.pool_len,
        ema_alpha=args.ema_alpha,
        reward_clip=args.reward_clip,
        lam_dyn=args.lam_dyn,
        conf_min=args.conf_min
    )
    soc = SocialCfg(
        K=args.K,
        R=args.R,
        use_radius=not args.no_radius,
        beta1=args.beta1,
        beta2=args.beta2
    )
    realtime_loop(
        rtsp=args.rtsp,
        encoder_ckpt=args.encoder_ckpt,
        phase3_ckpt=args.phase3_ckpt,
        imgsz=args.imgsz,
        T=args.T,
        use_conf_channel=args.use_conf,
        show=not args.no_show,
        save=args.save,
        device_str=args.device,
        score_cfg=sc,
        social_cfg=soc,
        max_tracks_per_frame=args.max_tracks,
        ttl_frames=args.ttl_frames
    )
