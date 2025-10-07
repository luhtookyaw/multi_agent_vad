# huvad_phase0_preprocess.py  (multi-agent ready, fixed li/lj indexing)
import os, glob, pickle, warnings, math
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

import numpy as np
from tqdm import tqdm

# --------------------------
# Config
# --------------------------
T_WINDOW = 64          # frames per window
STRIDE   = 32          # hop size
USE_CONF = True        # include confidence channel in X
MIN_COVERAGE = 0.9     # at least 90% frames present in a window
JOINTS = 17

# --- Social config
K_NEIGHBORS   = 3        # per-frame max neighbors per person
RADIUS_BODY   = 3.0      # neighbor radius (in units of focal person's bbox height)
SOC_ATT_BETA1 = 1.0      # weight on 1/d term
SOC_ATT_BETA2 = 1.0      # weight on closing speed term
SOC_CONF_MIN  = 0.10     # keypoint confidence threshold for distance proxies
SOC_FEAT_NAMES = [
    "d_scaled", "closing", "rel_speed", "facing",
    "hand2neck", "foot2torso", "torso_iou"
]
F_SOC = len(SOC_FEAT_NAMES)

# --------------------------
# Helpers: I/O + geometry
# --------------------------
def load_pkl(path:str):
    with open(path, "rb") as f:
        return pickle.load(f)

def swap_keypoints_yx_to_xy(kps: np.ndarray) -> np.ndarray:
    # input (17,3) (Y, X, C) -> output (17,3) (X, Y, C)
    kps = np.asarray(kps, dtype=np.float32)
    if kps.ndim != 2 or kps.shape[0] != JOINTS:
        raise ValueError(f"Bad keypoints shape {kps.shape}")
    if kps.shape[1] == 2:
        kps = np.concatenate([kps[:,[1,0]], np.ones((JOINTS,1), np.float32)], axis=1)
    else:
        kps = kps[:, [1, 0, 2]]
    # sanitize conf
    kps[:,2] = np.nan_to_num(kps[:,2], nan=0.0, posinf=0.0, neginf=0.0)
    return kps

def bbox_xyxy(bbox):
    """Ensure bbox = [x1, y1, x2, y2] as ints."""
    b = np.array(bbox, dtype=np.float32).reshape(-1)
    if b.size != 4:
        raise ValueError(f"Bad bbox {bbox}")
    x1, y1, x2, y2 = b
    # If x2,y2 look like width,height, convert
    if x2 <= x1 or y2 <= y1:
        x2 = x1 + b[2]; y2 = y1 + b[3]
    return int(x1), int(y1), int(x2), int(y2)

def normalize_keypoints_by_bbox(kps_xyc: np.ndarray, bbox_xyxy_val: Tuple[int,int,int,int]) -> np.ndarray:
    """Translate by bbox center, scale by bbox height."""
    x1, y1, x2, y2 = bbox_xyxy_val
    w = max(1.0, float(x2 - x1))
    h = max(1.0, float(y2 - y1))
    cx = x1 + 0.5 * w
    cy = y1 + 0.5 * h
    out = kps_xyc.copy().astype(np.float32)
    out[:, 0] = (out[:, 0] - cx) / h
    out[:, 1] = (out[:, 1] - cy) / h
    # conf unchanged
    return out

def cam_of_path(path: str) -> str:
    # .../c0/0_10.pkl  -> "c0"
    return os.path.basename(os.path.dirname(path))

def video_stem(path: str) -> str:
    # .../c0/0_10.pkl -> "0_10"
    return os.path.splitext(os.path.basename(path))[0]

# --------------------------
# Indices for COCO-17
# --------------------------
# 0: nose, 1: LEye,2: REye,3: LEar,4: REar,
# 5: LShoulder,6: RShoulder,7: LElbow,8: RElbow,9: LWrist,10: RWrist,
# 11: LHip,12: RHip,13: LKnee,14: RKnee,15: LAnkle,16: RAnkle
IDX = {
    "LWrist":9, "RWrist":10, "LAnkle":15, "RAnkle":16,
    "LShoulder":5, "RShoulder":6, "LHip":11, "RHip":12
}

def mid_point(a, b): return (a + b) * 0.5
def l2(x): return float(np.linalg.norm(x))

def torso_box_from_kps(kps_xyc: np.ndarray, fallback_xyxy: Tuple[int,int,int,int]) -> np.ndarray:
    """Small torso rectangle using shoulders & hips; fallback to bbox if invalid."""
    conf = kps_xyc[:,2]
    ok = []
    for k in ["LShoulder","RShoulder","LHip","RHip"]:
        ok.append(conf[IDX[k]] > SOC_CONF_MIN)
    if sum(ok) < 3:
        return np.array(fallback_xyxy, dtype=np.float32)
    ls = kps_xyc[IDX["LShoulder"], :2]
    rs = kps_xyc[IDX["RShoulder"], :2]
    lh = kps_xyc[IDX["LHip"], :2]
    rh = kps_xyc[IDX["RHip"], :2]
    xs = np.array([ls[0], rs[0], lh[0], rh[0]])
    ys = np.array([ls[1], rs[1], lh[1], rh[1]])
    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()
    # pad a bit
    padx = 0.10 * (x2 - x1 + 1); pady = 0.10 * (y2 - y1 + 1)
    return np.array([x1-padx, y1-pady, x2+padx, y2+pady], dtype=np.float32)

def iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2-ix1), max(0.0, iy2-iy1)
    inter = iw * ih
    ua = max(0.0, ax2-ax1) * max(0.0, ay2-ay1)
    ub = max(0.0, bx2-bx1) * max(0.0, by2-by1)
    denom = ua + ub - inter + 1e-6
    return float(inter / denom)

def min_pair_dist(A: np.ndarray, B: np.ndarray) -> float:
    # A: [Na,2], B: [Nb,2]
    if A.size == 0 or B.size == 0: return 1e6
    # broadcast
    d = A[:,None,:] - B[None,:,:]
    d = np.sqrt((d**2).sum(axis=-1))
    return float(np.min(d))

def hands_xy(kps_xyc):  # [2,2]
    conf = kps_xyc[:,2]
    pts = []
    for k in ["LWrist","RWrist"]:
        if conf[IDX[k]] > SOC_CONF_MIN:
            pts.append(kps_xyc[IDX[k],:2])
    return np.array(pts, dtype=np.float32)

def feet_xy(kps_xyc):
    conf = kps_xyc[:,2]
    pts = []
    for k in ["LAnkle","RAnkle"]:
        if conf[IDX[k]] > SOC_CONF_MIN:
            pts.append(kps_xyc[IDX[k],:2])
    return np.array(pts, dtype=np.float32)

def neck_point(kps_xyc):
    conf = kps_xyc[:,2]
    ok = (conf[IDX["LShoulder"]] > SOC_CONF_MIN) and (conf[IDX["RShoulder"]] > SOC_CONF_MIN)
    if not ok: return None
    ls = kps_xyc[IDX["LShoulder"], :2]
    rs = kps_xyc[IDX["RShoulder"], :2]
    return mid_point(ls, rs)

def torso_region_points(kps_xyc, torso_box_xyxy, grid=3):
    # sample grid points in torso box for a more stable distance proxy
    x1,y1,x2,y2 = torso_box_xyxy
    xs = np.linspace(x1, x2, grid); ys = np.linspace(y1, y2, grid)
    pts = np.stack(np.meshgrid(xs, ys), axis=-1).reshape(-1,2)
    return pts.astype(np.float32)

# --------------------------
# Track building & per-frame cache
# --------------------------
def frames_sorted_keys(video_data: Dict) -> List:
    return sorted(video_data.keys(), key=lambda x: float(x))

def build_tracks_and_framecache(video_data: Dict):
    """
    Returns:
      tracks[pid] = dict with:
        frames: [indices 0..N-1]
        pose_norm: list[(17,C)] normalized by bbox (C=2 or 3)
        pose_xyc: list[(17,3)] original x,y,conf (NaN for missing frames)
        bbox_xyxy: list[(4,)]
        present_mask: list[bool]
      frame_cache[t] = list of tuples (pid, center_xy, height, bbox_xyxy, kps_xyc)
    """
    fkeys = frames_sorted_keys(video_data)
    idx_map = {fk: i for i, fk in enumerate(fkeys)}
    N = len(fkeys)

    tracks = defaultdict(lambda: dict(frames=[], pose_norm=[], pose_xyc=[],
                                      bbox_xyxy=[], present_mask=[], last_index=-2))
    frame_cache = {i: [] for i in range(N)}

    for fk in fkeys:
        fidx = idx_map[fk]
        persons = video_data[fk]
        for pid, (bbox, kps_raw) in persons.items():
            kps_xyc = swap_keypoints_yx_to_xy(kps_raw)
            x1, y1, x2, y2 = bbox_xyxy(bbox)
            kps_norm = normalize_keypoints_by_bbox(kps_xyc, (x1, y1, x2, y2))
            cx, cy = (x1+x2)/2.0, (y1+y2)/2.0
            h = max(1.0, (y2-y1))

            tr = tracks[pid]
            last = tr['last_index']
            # fill gaps with missings
            if fidx != last + 1 and last >= -1:
                for miss in range(last+1, fidx):
                    tr['frames'].append(miss)
                    tr['pose_norm'].append(None)
                    tr['pose_xyc'].append(None)
                    tr['bbox_xyxy'].append(None)
                    tr['present_mask'].append(False)

            tr['frames'].append(fidx)
            tr['pose_norm'].append(kps_norm)
            tr['pose_xyc'].append(kps_xyc)
            tr['bbox_xyxy'].append(np.array([x1,y1,x2,y2], dtype=np.float32))
            tr['present_mask'].append(True)
            tr['last_index'] = fidx

            # frame cache
            frame_cache[fidx].append((int(pid), np.array([cx,cy],np.float32), float(h),
                                      np.array([x1,y1,x2,y2],np.float32), kps_xyc))

    for pid in list(tracks.keys()):
        tracks[pid].pop('last_index', None)

    return tracks, frame_cache, N

# --------------------------
# Utilities: temporal interpolation
# --------------------------
def interpolate_1d_with_nans(x: np.ndarray):
    n = x.shape[0]
    idx = np.arange(n)
    mask = ~np.isnan(x)
    if mask.sum() == 0:
        return np.zeros_like(x)
    return np.interp(idx, idx[mask], x[mask])

def interpolate_seq(arr: np.ndarray):
    """arr: [N, D] possibly with NaNs -> linear interp per dim"""
    out = arr.copy().astype(np.float32)
    for d in range(out.shape[1]):
        out[:,d] = interpolate_1d_with_nans(out[:,d])
    return out

# --------------------------
# Social feature computation
# --------------------------
def compute_primitives_per_track(tr, N_total:int):
    """
    Build per-track arrays of centers, heights, torso boxes, velocities, headings.
    Missing frames filled with NaNs then interpolated.
    (Arrays are length == len(tr['frames']) in the track's own index.)
    """
    frames = tr['frames']
    present = np.array(tr['present_mask'], dtype=bool)
    N = len(frames)

    center = np.full((N,2), np.nan, np.float32)
    height = np.full((N,1), np.nan, np.float32)
    torso  = np.full((N,4), np.nan, np.float32)

    for i,(pose_xyc, bbox) in enumerate(zip(tr['pose_xyc'], tr['bbox_xyxy'])):
        if pose_xyc is None or bbox is None: continue
        x1,y1,x2,y2 = bbox
        cx, cy = (x1+x2)/2.0, (y1+y2)/2.0
        h = max(1.0, (y2-y1))
        center[i] = [cx, cy]
        height[i,0] = h
        torso[i] = torso_box_from_kps(pose_xyc, bbox)

    center = interpolate_seq(center)
    height[:,0] = interpolate_1d_with_nans(height[:,0])
    torso = interpolate_seq(torso)

    # velocities and headings (unit vel; use torso orientation fallback if almost static)
    vel = np.vstack([[0,0], np.diff(center, axis=0)]).astype(np.float32)
    vnorm = np.linalg.norm(vel, axis=1, keepdims=True) + 1e-6
    heading = vel / vnorm
    for i in range(N):
        if np.linalg.norm(vel[i]) < 1e-3 and tr['pose_xyc'][i] is not None:
            k = tr['pose_xyc'][i]
            conf = k[:,2]
            if conf[IDX["LShoulder"]]>SOC_CONF_MIN and conf[IDX["RShoulder"]]>SOC_CONF_MIN \
               and conf[IDX["LHip"]]>SOC_CONF_MIN and conf[IDX["RHip"]]>SOC_CONF_MIN:
                mid_sh = mid_point(k[IDX["LShoulder"],:2], k[IDX["RShoulder"],:2])
                mid_hip= mid_point(k[IDX["LHip"],:2],      k[IDX["RHip"],:2])
                vec = mid_hip - mid_sh
                n = np.linalg.norm(vec) + 1e-6
                heading[i] = vec / n

    return {
        "center": center,     # [N,2]   (track index)
        "height": height,     # [N,1]
        "torso":  torso,      # [N,4]
        "vel":    vel,        # [N,2]
        "heading":heading     # [N,2]
    }

def social_neighbors_for_frame(i_center, i_h, all_ids, all_centers, all_heights):
    """Select up to K nearest within radius (RADIUS_BODY * i_h)."""
    nbrs = []
    for pid_j, c_j, h_j in zip(all_ids, all_centers, all_heights):
        if np.any(np.isnan(c_j)) or math.isinf(i_h): continue
        d = np.linalg.norm(c_j - i_center)
        if d <= RADIUS_BODY * i_h:
            nbrs.append((pid_j, d))
    # sort by distance, take up to K, drop self if present
    nbrs = [(j,d) for (j,d) in nbrs if d > 1e-6]
    nbrs.sort(key=lambda x: x[1])
    return [j for (j,_) in nbrs[:K_NEIGHBORS]]

def pair_features(pid_i, pid_j, li, lj, prim_i, prim_j, kps_i_t, kps_j_t):
    """
    Returns 7-dim feature vector for pair (i,j) using *track-local* indices li, lj.
    All geometric quantities are in pixel space, distances normalized by i's height.
    """
    # --- safety: bounds check (defensive) ---
    for name, idx, arr in (("i.center", li, prim_i["center"]),
                           ("j.center", lj, prim_j["center"])):
        if idx < 0 or idx >= arr.shape[0]:
            return np.zeros((F_SOC,), np.float32)

    ci, vi, hi = prim_i["center"][li], prim_i["vel"][li], float(prim_i["height"][li,0] + 1e-6)
    cj, vj     = prim_j["center"][lj], prim_j["vel"][lj]
    ri = cj - ci
    d = np.linalg.norm(ri) + 1e-6
    d_scaled = d / hi

    vij = vj - vi
    closing = - float((ri / d) @ vij)                # >0 when approaching
    rel_speed = float(np.linalg.norm(vij))

    # facing: dot( heading_i , direction to j )
    hi_vec = prim_i["heading"][li]
    facing = float(np.clip((hi_vec @ (ri / d)), -1.0, 1.0))

    # contact proxies
    tb_i = prim_i["torso"][li]; tb_j = prim_j["torso"][lj]
    iou = iou_xyxy(tb_i, tb_j)

    # distances
    hpts_i = hands_xy(kps_i_t); fpts_i = feet_xy(kps_i_t)
    npt_j = neck_point(kps_j_t)
    if npt_j is None:
        conf = kps_j_t[:,2]
        if conf[IDX["LShoulder"]]>SOC_CONF_MIN and conf[IDX["RShoulder"]]>SOC_CONF_MIN:
            npt_j = mid_point(kps_j_t[IDX["LShoulder"],:2], kps_j_t[IDX["RShoulder"],:2])
    hand2neck = min_pair_dist(hpts_i, np.array([npt_j],np.float32)) if (hpts_i.size and npt_j is not None) else 1e6
    torso_pts_j = torso_region_points(kps_j_t, tb_j, grid=3)
    foot2torso = min_pair_dist(fpts_i, torso_pts_j) if fpts_i.size else 1e6

    # normalize distances by hi
    hand2neck /= hi
    foot2torso /= hi

    return np.array([d_scaled, closing, rel_speed, facing, hand2neck, foot2torso, iou], dtype=np.float32)

def attention_weights(feats_mat: np.ndarray):
    """
    feats_mat: [M, F] pairwise features for M neighbors.
    Score = beta1*(1/d_scaled) + beta2*closing
    """
    if feats_mat.shape[0] == 0:
        return np.zeros((0,), dtype=np.float32)
    d_scaled = np.clip(feats_mat[:,0], 1e-6, None)
    closing  = feats_mat[:,1]
    score = SOC_ATT_BETA1*(1.0/d_scaled) + SOC_ATT_BETA2*closing
    # softmax
    m = score.max()
    w = np.exp(score - m)
    w /= (w.sum() + 1e-6)
    return w.astype(np.float32)

def build_social_tensor_for_video(tracks, frame_cache, N_frames) -> Dict[int, np.ndarray]:
    """
    For each pid, returns G_pid: [len(track['frames']), F_SOC] aligned to the track's indexing.
    Missing frames are interpolated.
    """
    # precompute primitives per track (in track frame indexing)
    prim = {pid: compute_primitives_per_track(tr, N_frames) for pid,tr in tracks.items()}

    # pid -> {global_t: local_idx}
    local_idx = {pid: {g:i for i,g in enumerate(tr['frames'])} for pid,tr in tracks.items()}

    # storage (per-track length)
    G = {pid: np.full((len(tr['frames']), F_SOC), np.nan, np.float32) for pid,tr in tracks.items()}

    for t in range(N_frames):
        # items: (pid, center_xy, height, bbox_xyxy, kps_xyc)
        items = frame_cache[t]
        if not items: 
            continue

        # explicit unpacking to avoid tuple-unpack mistakes
        ids  = [pid for (pid, _, _, _, _) in items]
        ctrs = [c   for (_, c, _, _, _) in items]
        hts  = [h   for (_, _, h, _, _) in items]
        # bbxs = [b   for (_, _, _, b, _) in items]   # (kept if you need it)
        kps_map = {pid: kp for (pid, _, _, _, kp) in items}

        # per focal person
        for i_idx, pid_i in enumerate(ids):
            # only if this track has a local index at global t
            li = local_idx.get(pid_i, {}).get(t, None)
            if li is None:
                continue

            # select neighbors around person i
            nbr_ids = social_neighbors_for_frame(
                i_center=ctrs[i_idx], i_h=hts[i_idx],
                all_ids=ids, all_centers=ctrs, all_heights=hts
            )
            if len(nbr_ids) == 0:
                G[pid_i][li] = np.zeros((F_SOC,), np.float32)
                continue

            feats = []
            for pid_j in nbr_ids:
                lj = local_idx.get(pid_j, {}).get(t, None)
                if lj is None:
                    continue
                kpi = tracks[pid_i]['pose_xyc'][li]
                kpj = tracks[pid_j]['pose_xyc'][lj]
                if kpi is None or kpj is None:
                    continue
                feats.append(
                    pair_features(pid_i, pid_j, li, lj, prim[pid_i], prim[pid_j], kpi, kpj)
                )

            if len(feats) == 0:
                G[pid_i][li] = np.zeros((F_SOC,), np.float32)
                continue

            F_mat = np.stack(feats, axis=0)  # [M, F_SOC]
            w = attention_weights(F_mat)     # [M]
            g = (w[:,None] * F_mat).sum(axis=0)
            G[pid_i][li] = g.astype(np.float32)

    # interpolate missing g over time per pid
    for pid in G.keys():
        g = G[pid]
        for f in range(F_SOC):
            g[:,f] = interpolate_1d_with_nans(g[:,f])
        G[pid] = g
    return G

# --------------------------
# Windowing (now returns X and G)
# --------------------------
def windowize_track(tr: Dict, G_pid: np.ndarray, T=T_WINDOW, stride=STRIDE, use_conf=USE_CONF, min_coverage=MIN_COVERAGE):
    """
    Returns:
      X_windows: list[(T,V,C)]
      G_windows: list[(T,F_SOC)]
      metas    : list[dict]
    """
    frames = tr['frames']
    pose_seq = tr['pose_norm']
    present = np.array(tr['present_mask'], dtype=bool)

    V = JOINTS
    C = 3 if use_conf else 2

    N = len(pose_seq)
    X_arr = np.full((N, V, C), np.nan, dtype=np.float32)

    for i, kp in enumerate(pose_seq):
        if kp is None: 
            continue
        if not use_conf and kp.shape[1] >= 3:
            X_arr[i, :, :2] = kp[:, :2]
        else:
            if kp.shape[1] == 2 and use_conf:
                tmp = np.concatenate([kp, np.ones((V,1), dtype=np.float32)], axis=1)
                X_arr[i] = tmp
            else:
                X_arr[i, :, :kp.shape[1]] = kp

    # interp X over missing frames
    for j in range(V):
        for c in range(C):
            X_arr[:, j, c] = interpolate_1d_with_nans(X_arr[:, j, c])

    # slide windows
    X_windows, G_windows, metas = [], [], []
    for start in range(0, N - T + 1, stride):
        end = start + T
        cov = present[start:end].mean()
        if cov < min_coverage:
            continue
        X_windows.append(X_arr[start:end])      # (T,V,C)
        G_windows.append(G_pid[start:end])      # (T,F_SOC)
        metas.append({'start': start, 'end': end})
    return X_windows, G_windows, metas

# --------------------------
# GT utilities (for Test split)
# --------------------------
def load_gt_vector(gt_path: str) -> Optional[np.ndarray]:
    if not os.path.exists(gt_path):
        return None
    try:
        vec = np.load(gt_path)
        vec = np.asarray(vec).astype(np.int64).reshape(-1)
        return vec
    except Exception as e:
        warnings.warn(f"Failed to load GT {gt_path}: {e}")
        return None

def label_windows_from_gt(win_metas: List[Dict], gt_vec: Optional[np.ndarray], n_frames:int) -> List[int]:
    if gt_vec is None:
        return [-1] * len(win_metas)
    L = min(len(gt_vec), n_frames)
    out = []
    for m in win_metas:
        s, e = m['start'], m['end']
        s = max(0, min(s, L-1))
        e = max(0, min(e, L))
        lab = int((gt_vec[s:e].sum() > 0))
        out.append(lab)
    return out

# --------------------------
# Main API
# --------------------------
def process_split(
    root_dir: str,
    split: str,              # "Train" or "Test"
    subset: str = "Normal_Training",  # or "Online_Training"
    out_npz: str = "phase0_{}.npz",
    use_conf: bool = USE_CONF
):
    """
    Saves:
      - X: [N, T, V, C]  (normalized pose windows)
      - G: [N, T, F_soc] (aggregated social windows)
      - y: [N] window labels (Test via GT; Train uses 0)
      - meta: list[dict]
    """
    base = os.path.join(root_dir, subset)
    pkl_glob = os.path.join(base, split, "*", "*.pkl")
    pkl_files = sorted(glob.glob(pkl_glob))
    if len(pkl_files) == 0:
        raise FileNotFoundError(f"No PKL files found at {pkl_glob}")

    X_list, G_list, y_list, meta_list = [], [], [], []

    for pkl_path in tqdm(pkl_files, desc=f"[{subset}/{split}]"):
        vid = video_stem(pkl_path)     # "0_10"
        cam = cam_of_path(pkl_path)    # "c0"

        data = load_pkl(pkl_path)
        tracks, frame_cache, n_frames = build_tracks_and_framecache(video_data=data)

        # Precompute social tensors G_pid for all tracks in this video
        G_all = build_social_tensor_for_video(tracks, frame_cache, n_frames)

        # For Test split, load GT vector
        gt_vec = None
        if split.lower() == "test":
            gt_path = os.path.join(base, "GT", cam, f"{vid}.npy")
            gt_vec = load_gt_vector(gt_path)
            if gt_vec is not None and len(gt_vec) != n_frames:
                warnings.warn(f"GT length ({len(gt_vec)}) != frames ({n_frames}) for {cam}/{vid}. Aligning by min length.")

        # Windowize per person
        for pid, tr in tracks.items():
            Xw, Gw, metas = windowize_track(tr, G_all[pid], T=T_WINDOW, stride=STRIDE, use_conf=use_conf, min_coverage=MIN_COVERAGE)
            if not Xw:
                continue
            if split.lower() == "train":
                labels = [0] * len(Xw)
            else:
                labels = label_windows_from_gt(metas, gt_vec, n_frames)

            X_list.extend(Xw)
            G_list.extend(Gw)
            y_list.extend(labels)
            for m in metas:
                meta_list.append({
                    'camera': cam,
                    'video': vid,
                    'person_id': int(pid),
                    'start': m['start'],
                    'end': m['end'],
                    'n_frames_video': n_frames
                })

    # Stack & save
    C = 3 if use_conf else 2
    X = np.stack(X_list, axis=0).astype(np.float32) if X_list else np.zeros((0, T_WINDOW, JOINTS, C), dtype=np.float32)
    G = np.stack(G_list, axis=0).astype(np.float32) if G_list else np.zeros((0, T_WINDOW, F_SOC), dtype=np.float32)
    y = np.array(y_list, dtype=np.int64) if y_list else np.zeros((0,), dtype=np.int64)

    out_path = out_npz.format(split.lower())
    np.savez_compressed(out_path, X=X, G=G, y=y, meta=np.array(meta_list, dtype=object), soc_feat_names=np.array(SOC_FEAT_NAMES, dtype=object))
    print(f"✅ Saved {subset}/{split}: X{X.shape}  G{G.shape}  → {out_path}")

if __name__ == "__main__":
    ROOT = "./"  # folder that contains Normal_Training / Online_Training

    # Phase-0 for Normal_Training
    process_split(ROOT, split="Train", subset="Normal_Training", out_npz="phase0_multi_normal_train.npz", use_conf=USE_CONF)
    process_split(ROOT, split="Test",  subset="Normal_Training", out_npz="phase0_multi_normal_test.npz",  use_conf=USE_CONF)

    # (Optional) Phase-0 for Online_Training
    # process_split(ROOT, split="Train", subset="Online_Training", out_npz="phase0_multi_online_train.npz", use_conf=USE_CONF)
    # process_split(ROOT, split="Test",  subset="Online_Training", out_npz="phase0_multi_online_test.npz",  use_conf=USE_CONF)
