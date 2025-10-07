# phase1_stgcn_embeddings.py  (multi-agent ready)
import os
import math
import json
import numpy as np
from typing import Tuple, Dict, List
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ============================================================
# 0) Graph (COCO-17) and Utilities
# ============================================================

COCO_EDGES = [
    (0,1),(0,2),(1,3),(2,4),
    (5,6),
    (5,7),(7,9),
    (6,8),(8,10),
    (5,11),(6,12),(11,12),
    (11,13),(13,15),
    (12,14),(14,16)
]

def build_adjacency(num_nodes=17, edges=COCO_EDGES):
    A = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    for i, j in edges:
        A[i, j] = 1.0
        A[j, i] = 1.0
    # Add self connections (stability)
    for i in range(num_nodes):
        A[i, i] = 1.0
    # D^{-1/2} A D^{-1/2}
    D = np.diag(1.0 / np.sqrt(A.sum(1) + 1e-8))
    A_norm = D @ A @ D
    return torch.from_numpy(A_norm)  # [V,V]

# ============================================================
# 1) Minimal ST-GCN Backbone
#    Input:  (B, C, T, V)   (we ignore M since M=1)
#    Output: per-frame embedding E: (B, D, T)
# ============================================================

class STGCNBlock(nn.Module):
    """
    Very small ST-GCN block:
    - Temporal conv: Conv2d(C_in->C_out, kernel=(k,1))
    - Spatial GCN:   einsum over normalized adjacency
    - Residual + BN + ReLU
    """
    def __init__(self, c_in, c_out, A_mat, k_t=9, stride_t=1, dropout=0.0):
        super().__init__()
        padding_t = (k_t // 2)
        self.A = A_mat  # [V,V], normalized
        self.temporal = nn.Conv2d(c_in, c_out, kernel_size=(k_t,1),
                                  stride=(stride_t,1), padding=(padding_t,0), bias=False)
        self.theta = nn.Conv2d(c_out, c_out, kernel_size=(1,1), bias=False)  # 1x1 after spatial
        self.bn = nn.BatchNorm2d(c_out)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.act = nn.ReLU(inplace=True)
        # residual mapping
        if stride_t != 1 or c_in != c_out:
            self.residual = nn.Conv2d(c_in, c_out, kernel_size=1, stride=(stride_t,1), bias=False)
        else:
            self.residual = nn.Identity()

    def forward(self, x):
        # x: [B, C, T, V]
        res = self.residual(x)
        xt = self.temporal(x)      # [B, C_out, T, V]
        # spatial GCN: multiply along V dimension
        xs = torch.einsum('bctv,vw->bctw', xt, self.A.to(xt.device))
        xs = self.theta(xs)
        xs = self.bn(xs)
        xs = self.act(xs + res)
        xs = self.dropout(xs)
        return xs

class TinySTGCN(nn.Module):
    """
    A tiny 3-block ST-GCN with output embedding per frame via global-V pooling.
    - in_ch: 2 or 3 (x,y,(conf))
    - out_dim: embedding dimension D (e.g., 64)
    """
    def __init__(self, in_ch=2, out_dim=64, A_mat=None, k_t=9, dropout=0.1):
        super().__init__()
        assert A_mat is not None
        c1, c2, c3 = 64, 64, out_dim
        self.g1 = STGCNBlock(in_ch, c1, A_mat, k_t=k_t, stride_t=1, dropout=dropout)
        self.g2 = STGCNBlock(c1, c2, A_mat, k_t=k_t, stride_t=1, dropout=dropout)
        self.g3 = STGCNBlock(c2, c3, A_mat, k_t=k_t, stride_t=1, dropout=dropout)

    def forward(self, x):
        """
        x: [B, C, T, V]
        returns:
          feats: [B, D, T]  per-frame embeddings (global pooled over V)
          fmap:  [B, D, T, V] last feature map (optional)
        """
        x = self.g1(x)
        x = self.g2(x)
        x = self.g3(x)  # [B, D, T, V]
        # global avg over joints (V)
        E = x.mean(dim=-1)  # [B, D, T]
        return E, x

# ============================================================
# 2) Decoders for Pretext Tasks
#    - RECON: decode current joints from E_t
#    - NEXT:  decode next-step joints from E_{t-1}
# ============================================================

class JointDecoder(nn.Module):
    """
    Decode joints (V,C_out) from a per-frame embedding E_t (dim D).
    Implemented as 1x1 conv over time (linear per time-step).
    """
    def __init__(self, emb_dim=64, out_ch=2, V=17):
        super().__init__()
        self.V, self.Cout = V, out_ch
        self.fc = nn.Conv1d(emb_dim, V*out_ch, kernel_size=1)  # (B, D, T) -> (B, V*C, T)
    def forward(self, E):
        # E: [B, D, T]
        y = self.fc(E)  # [B, V*C, T]
        B, VC, T = y.shape
        y = y.view(B, self.V, self.Cout, T)  # [B, V, C, T]
        # return [B, T, V, C] to match keypoint target orientation
        return y.permute(0, 3, 1, 2).contiguous()

# ============================================================
# 3) Dataset wrapper for Phase-0 outputs (X only for pretext)
#    (G is not needed to train ST-GCN, but we handle it in export.)
# ============================================================

class HUVADPhase0Windows(Dataset):
    """
    Loads Phase-0 .npz with:
      X: [N, T, V, C], y: [N], meta: object array of dicts (camera, video, ...)
      (Optionally G: [N, T, F_soc] — ignored in training.)
    Provides tensors in ST-GCN format: [B, C, T, V].
    """
    def __init__(self, npz_path, use_conf=True):
        super().__init__()
        dat = np.load(npz_path, allow_pickle=True)
        self.X = dat['X']          # [N,T,V,C]
        self.y = dat['y']          # [N]
        self.meta = dat['meta']    # [N] (object)
        self.has_G = 'G' in dat.files  # we don't use G here
        self.use_conf = use_conf
        # ensure channels
        C = self.X.shape[-1]
        if not use_conf and C > 2:
            self.X = self.X[..., :2]   # drop conf
        self.N, self.T, self.V, self.C = self.X.shape

    def __len__(self): return self.N

    def __getitem__(self, idx):
        x = self.X[idx]  # [T,V,C]
        x = np.transpose(x, (2,0,1)).astype(np.float32)  # -> [C,T,V]
        return torch.from_numpy(x), int(self.y[idx]), self.meta[idx].item()

# ============================================================
# 4) Training loop (pretext = 'recon' or 'next')
# ============================================================

def train_stgcn_pretext(
    train_npz: str,
    val_npz: str = None,
    out_dir: str = "phase1_ckpt",
    pretext: str = "recon",             # 'recon' or 'next'
    emb_dim: int = 64,
    batch_size: int = 64,
    lr: float = 1e-3,
    max_epochs: int = 30,
    use_conf: bool = True,
    device: str = None
):
    os.makedirs(out_dir, exist_ok=True)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # Data
    ds_train = HUVADPhase0Windows(train_npz, use_conf=use_conf)
    dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)

    ds_val, dl_val = None, None
    if val_npz is not None and os.path.exists(val_npz):
        ds_val = HUVADPhase0Windows(val_npz, use_conf=use_conf)
        dl_val = DataLoader(ds_val, batch_size=batch_size, shuffle=False, num_workers=2)

    # Model
    A = build_adjacency(num_nodes=ds_train.V)  # [V,V]
    stgcn = TinySTGCN(in_ch=ds_train.C, out_dim=emb_dim, A_mat=A)
    dec = JointDecoder(emb_dim=emb_dim, out_ch=ds_train.C, V=ds_train.V)

    stgcn, dec = stgcn.to(device), dec.to(device)

    opt = torch.optim.Adam(list(stgcn.parameters()) + list(dec.parameters()), lr=lr, weight_decay=1e-5)

    def step(batch):
        x, _, _ = batch  # x:[B,C,T,V]
        x = x.to(device)
        E, _ = stgcn(x)  # E:[B,D,T]

        if pretext == "recon":
            # decode current joints
            pred = dec(E)          # [B,T,V,C]
            tgt  = x.permute(0,2,3,1).contiguous()  # [B,T,V,C]
            loss = F.mse_loss(pred, tgt)

        elif pretext == "next":
            # predict next-step joints: use E_{t-1} -> target joints at t
            pred = dec(E[:, :, :-1])       # [B,T-1,V,C]
            tgt  = x.permute(0,2,3,1)[:, 1:, ...].contiguous()  # [B,T-1,V,C]
            loss = F.mse_loss(pred, tgt)
        else:
            raise ValueError("pretext must be 'recon' or 'next'")
        return loss

    best_val = math.inf
    for epoch in range(1, max_epochs+1):
        stgcn.train(); dec.train()
        tr_loss, n = 0.0, 0
        for batch in dl_train:
            opt.zero_grad()
            loss = step(batch)
            loss.backward()
            opt.step()
            tr_loss += loss.item() * batch[0].shape[0]
            n += batch[0].shape[0]
        tr_loss /= max(1,n)

        msg = f"[Epoch {epoch:03d}] train loss: {tr_loss:.6f}"

        if dl_val is not None:
            stgcn.eval(); dec.eval()
            va_loss, m = 0.0, 0
            with torch.no_grad():
                for batch in dl_val:
                    loss = step(batch)
                    va_loss += loss.item() * batch[0].shape[0]
                    m += batch[0].shape[0]
            va_loss /= max(1,m)
            msg += f" | val loss: {va_loss:.6f}"
            if va_loss < best_val:
                best_val = va_loss
                torch.save({'stgcn': stgcn.state_dict(), 'dec': dec.state_dict(),
                            'cfg': {'emb_dim': emb_dim, 'in_ch': ds_train.C, 'V': ds_train.V}},
                           os.path.join(out_dir, f"best_{pretext}.pt"))
        else:
            torch.save({'stgcn': stgcn.state_dict(), 'dec': dec.state_dict(),
                        'cfg': {'emb_dim': emb_dim, 'in_ch': ds_train.C, 'V': ds_train.V}},
                       os.path.join(out_dir, f"last_{pretext}.pt"))

        print(msg)

    ckpt_path = os.path.join(out_dir, f"best_{pretext}.pt") if dl_val is not None else os.path.join(out_dir, f"last_{pretext}.pt")
    print(f"✅ Finished training. Checkpoint: {ckpt_path}")
    return ckpt_path

# ============================================================
# 5) Embedding export + per-camera z-score + actions (multi-agent)
#    - Reads optional G from Phase-0 NPZ
#    - Z-scores E and G per camera
#    - Builds state S_t = [e_t ; g_t]
#    - Actions A_t = e_{t+1} - e_t
# ============================================================

@torch.no_grad()
def export_embeddings_and_actions_multi(
    npz_path: str,
    ckpt_path: str,
    out_npz: str = "phase1_multi_embeddings_actions.npz",
    use_conf: bool = True,
    zscore_per_camera: bool = True,
    concat_state: bool = True
):
    dat = np.load(npz_path, allow_pickle=True)
    X = dat['X']     # [N,T,V,C]
    y = dat['y']     # [N]
    meta = dat['meta']
    has_G = 'G' in dat.files
    G = dat['G'] if has_G else None  # [N,T,F_soc] if present
    soc_feat_names = dat['soc_feat_names'] if ('soc_feat_names' in dat.files) else None

    N, T, V, C = X.shape
    F_soc = int(G.shape[-1]) if G is not None else 0

    # Load model
    ckpt = torch.load(ckpt_path, map_location='cpu')
    in_ch = ckpt['cfg']['in_ch']
    emb_dim = ckpt['cfg']['emb_dim']
    V_chk = ckpt['cfg'].get('V', V)
    if V_chk != V:
        raise ValueError(f"V mismatch: checkpoint expects V={V_chk}, but npz has V={V}")

    A_graph = build_adjacency(num_nodes=V)
    stgcn = TinySTGCN(in_ch=in_ch, out_dim=emb_dim, A_mat=A_graph)
    stgcn.load_state_dict(ckpt['stgcn'])
    stgcn.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    stgcn.to(device)

    # Infer embeddings in chunks to avoid OOM
    BATCH = 128
    E_all = np.zeros((N, T, emb_dim), dtype=np.float32)

    for i in range(0, N, BATCH):
        xb = X[i:i+BATCH]                         # [b,T,V,C]
        if not use_conf and xb.shape[-1] > 2:
            xb = xb[..., :2]
        xb = np.transpose(xb, (0,3,1,2))         # -> [b,C,T,V]
        xb = torch.from_numpy(xb).float().to(device)
        Eb, _ = stgcn(xb)                         # Eb: [b,D,T]
        Eb = Eb.permute(0, 2, 1).contiguous().cpu().numpy()  # [b,T,D]
        E_all[i:i+BATCH] = Eb

    cams = [m.item()['camera'] for m in meta]

    # Z-score per camera: E and (if present) G independently
    E_norm = E_all.copy()
    G_norm = None if G is None else G.copy()

    zstats = {'E': {}, 'G': {}}

    if zscore_per_camera:
        by_cam = defaultdict(list)
        for i, c in enumerate(cams): by_cam[c].append(i)

        # E stats
        for c, idxs in by_cam.items():
            Ec = E_all[idxs]                         # [Nc, T, D]
            mu = Ec.reshape(-1, emb_dim).mean(axis=0)
            sd = Ec.reshape(-1, emb_dim).std(axis=0) + 1e-6
            E_norm[idxs] = (Ec - mu) / sd
            zstats['E'][c] = {'mean': mu.tolist(), 'std': sd.tolist()}

        # G stats
        if G is not None:
            for c, idxs in by_cam.items():
                Gc = G[idxs]                         # [Nc, T, F]
                mu = Gc.reshape(-1, F_soc).mean(axis=0)
                sd = Gc.reshape(-1, F_soc).std(axis=0) + 1e-6
                G_norm[idxs] = (Gc - mu) / sd
                zstats['G'][c] = {'mean': mu.tolist(), 'std': sd.tolist()}
    else:
        # global (rare)
        muE = E_all.reshape(-1, emb_dim).mean(axis=0)
        sdE = E_all.reshape(-1, emb_dim).std(axis=0) + 1e-6
        E_norm = (E_all - muE) / sdE
        zstats['E']['global'] = {'mean': muE.tolist(), 'std': sdE.tolist()}
        if G is not None:
            muG = G.reshape(-1, F_soc).mean(axis=0)
            sdG = G.reshape(-1, F_soc).std(axis=0) + 1e-6
            G_norm = (G - muG) / sdG
            zstats['G']['global'] = {'mean': muG.tolist(), 'std': sdG.tolist()}

    # Continuous actions on embedding only (as per design)
    A_all = E_norm[:, 1:, :] - E_norm[:, :-1, :]  # [N, T-1, D]

    # Build state S = concat(E_norm, G_norm) if requested (and G exists)
    if concat_state and (G_norm is not None):
        S_all = np.concatenate([E_norm, G_norm], axis=-1)  # [N,T,D+F]
    else:
        S_all = E_norm  # if no G, state == embedding

    out = {
        'E': E_norm,         # [N,T,D]
        'A': A_all,          # [N,T-1,D]
        'S': S_all,          # [N,T,D(+F)]
        'y': y,
        'meta': meta
    }
    if G_norm is not None:
        out['G'] = G_norm    # [N,T,F]
        if soc_feat_names is not None:
            out['soc_feat_names'] = soc_feat_names

    out['zscore_stats_json'] = np.array(json.dumps(zstats), dtype=object)

    np.savez_compressed(out_npz, **out)
    print(f"✅ Saved multi-agent embeddings/actions: {out_npz}")
    print(f"   E: {E_norm.shape}, A: {A_all.shape}, S: {S_all.shape}" + (f", G: {G_norm.shape}" if G_norm is not None else ""))

# ============================================================
# 6) CLI usage example
# ============================================================

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--train_npz", type=str, default="phase0_multi_normal_train.npz")  # multi-agent Phase-0 by default
    p.add_argument("--val_npz",   type=str, default=None)   # optional
    p.add_argument("--test_npz",  type=str, default="phase0_multi_normal_test.npz")
    p.add_argument("--out_dir",   type=str, default="phase1_ckpt")
    p.add_argument("--pretext",   type=str, default="recon", choices=["recon","next"])
    p.add_argument("--emb_dim",   type=int, default=64)
    p.add_argument("--batch_size",type=int, default=64)
    p.add_argument("--lr",        type=float, default=1e-3)
    p.add_argument("--epochs",    type=int, default=30)
    p.add_argument("--use_conf",  action="store_true")
    p.add_argument("--no_use_conf", dest="use_conf", action="store_false")
    p.set_defaults(use_conf=True)
    args = p.parse_args()

    # 1) Train ST-GCN with selected pretext (only uses X)
    ckpt_path = train_stgcn_pretext(
        train_npz=args.train_npz,
        val_npz=args.val_npz,
        out_dir=args.out_dir,
        pretext=args.pretext,
        emb_dim=args.emb_dim,
        batch_size=args.batch_size,
        lr=args.lr,
        max_epochs=args.epochs,
        use_conf=args.use_conf
    )

    # 2) Export embeddings & actions & (optional) social-aware state
    os.makedirs(args.out_dir, exist_ok=True)
    export_embeddings_and_actions_multi(
        npz_path=args.train_npz,
        ckpt_path=ckpt_path,
        out_npz=os.path.join(args.out_dir, f"emb_act_state_train_{args.pretext}.npz"),
        use_conf=args.use_conf,
        zscore_per_camera=True,
        concat_state=True
    )
    if args.test_npz and os.path.exists(args.test_npz):
        export_embeddings_and_actions_multi(
            npz_path=args.test_npz,
            ckpt_path=ckpt_path,
            out_npz=os.path.join(args.out_dir, f"emb_act_state_test_{args.pretext}.npz"),
            use_conf=args.use_conf,
            zscore_per_camera=True,
            concat_state=True
        )
