# How To Run

## 1) Train with autoencoder pretext (reconstruct joints)
``` python
python phase1_stgcn_embeddings.py \
  --train_npz phase0_normal_train.npz \
  --val_npz   phase0_normal_test.npz \
  --test_npz  phase0_normal_test.npz \
  --pretext recon --emb_dim 64 --epochs 30
```

## 2) Or: train with next-step prediction
``` python
  python phase1_stgcn_embeddings.py \
    --train_npz phase0_normal_train.npz \
    --val_npz   phase0_normal_test.npz \
    --test_npz  phase0_normal_test.npz \
    --pretext next --emb_dim 64 --epochs 30
```

## 3) TRAIN policy (BC) + dynamics, and INIT reward on Phase-1 train NPZ
``` python
  python phase2_irl_models.py \
    --train_npz phase1_ckpt/emb_act_train_recon.npz \
    --out_dir   phase2_ckpt \
    --epochs    30 --batch 1024 --lr 1e-3 \
    --init_log_sigma -2.0
```
``` python
  # (Optional) turn off residual in dynamics to keep s'≈s+a:
  python phase2_irl_models.py \
    --train_npz phase1_ckpt/emb_act_train_recon.npz \
    --out_dir   phase2_ckpt \
    --no_dyn_residual
```

## 4) Train MaxEnt IRL (with hand-crafted extras + policy improvement)
``` python
  python phase3_maxent_irl.py \
    --train_npz phase1_ckpt/emb_act_train_recon.npz \
    --test_npz  phase1_ckpt/emb_act_test_recon.npz \
    --policy_ckpt phase2_ckpt/policy_bc.pt \
    --dyn_ckpt    phase2_ckpt/dynamics.pt \
    --reward_ckpt phase2_ckpt/reward_init.pt \
    --iters 80 --rollouts 256 --horizon 63 \
    --tau 5.0 --alpha 0.01 --use_handcrafted \
    --out_dir phase3_ckpt
```

## 5) Scores are in: phase3_ckpt/scores_test.npz  (higher = more anomalous)

## 6) Example: score your Phase-1 test windows with dynamics surprise and top-10% pooling
``` python
  python phase4_inference.py \
    --ckpt phase3_ckpt/final.pt \
    --npz  phase1_ckpt/emb_act_test_recon.npz \
    --out  phase4_ckpt/scores_test_top10_dyn05.npz \
    --topk 0.10 \
    --lam_dyn 0.5 \
    --thresh_method quantile --thresh_q 0.95
```

## 7) With CUDA (if available) and live RTSP
python realtime_irl.py \
  --rtsp "rtsp://user:pass@CAMERA_IP/stream" \
  --encoder_ckpt phase1_ckpt/stgcn_encoder.pt \
  --phase3_ckpt  phase3_ckpt/final.pt \
  --imgsz 960 --T 64 --use_conf \
  --lam_dyn 0.5 \
  --save outputs/live_anomaly.mp4

## 8) From a video file (for quick testing on CPU)
python realtime_irl.py \
  --rtsp samples/people.mp4 \
  --encoder_ckpt phase1_ckpt/stgcn_encoder.pt \
  --phase3_ckpt  phase3_ckpt/final.pt \
  --device cpu --no_show
