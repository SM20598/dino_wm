"""
Open-loop rollout evaluation for a trained DINO-WM checkpoint.

Samples trajectories from the held-out dataset, feeds the *ground-truth*
action sequence through the world model's autoregressive `rollout()` (no
live simulator involved), decodes the predicted latents back to images, and
compares them against the real trajectory: latent (embedding) error, image
error metrics (l1/l2/ssim/psnr/lpips), and side-by-side GT-vs-predicted grids
with the push action (start/end pusher pose + direction arrow) overlaid.

For obs_type=occupancy, the decoded image *is* the particle raster, so image
error against ground truth is a direct proxy for particle-prediction error.
There's no separate particle-state decoder, so raw state vectors aren't
compared numerically. The action overlay only applies to obs_type=occupancy,
since it relies on the same top-down meters->pixel projection Genesis used to
rasterize particles (verified empirically against states.pth); there's no
equivalent projection for obs_type=rendered (a perspective camera view).

Usage:
    python eval_open_loop.py --model_dir outputs/2026-08-24/13-36-19 --epoch latest
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import hydra
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Rectangle
from matplotlib.transforms import Affine2D
from omegaconf import OmegaConf

from metrics.image_metrics import eval_images
from utils import seed, slice_trajdict_with_t

ALL_MODEL_KEYS = ["encoder", "predictor", "decoder", "proprio_encoder", "action_encoder"]

GT_COLOR = "#2ca02c"    # green
PRED_COLOR = "#1f77b4"  # blue
CTX_COLOR = "#888888"   # gray, for context frames fed into the model (not real predictions)

# Approximate pusher plate footprint in meters (x, y), matching the "plate.size"
# used by the cube-pile Genesis configs this dataset was collected with
# (e.g. Genesis/data/corl/cube/n10/size0.005/_0_config.yaml: [0.04, 0.002, 0.01]).
# The exported dataset doesn't retain the exact per-episode config, so this is
# an approximation until we wire up real inference from the Genesis/plate yaml.
PLATE_SIZE_M = (0.04, 0.002)


def load_model(model_dir: Path, epoch, device):
    with open(model_dir / "hydra.yaml") as f:
        model_cfg = OmegaConf.load(f)

    ckpt_name = "model_latest.pth" if epoch == "latest" else f"model_{epoch}.pth"
    ckpt_path = model_dir / "checkpoints" / ckpt_name
    payload = torch.load(ckpt_path, map_location=device, weights_only=False)
    print(f"Loaded checkpoint from epoch {payload['epoch']}: {ckpt_path}")

    result = {k: v.to(device) for k, v in payload.items() if k in ALL_MODEL_KEYS}

    if "encoder" not in result:
        result["encoder"] = hydra.utils.instantiate(model_cfg.encoder).to(device)
    if "predictor" not in result:
        raise ValueError(f"No predictor found in checkpoint {ckpt_path}")
    if "decoder" not in result and model_cfg.has_decoder:
        raise ValueError(f"No decoder found in checkpoint {ckpt_path}")

    model = hydra.utils.instantiate(
        model_cfg.model,
        encoder=result["encoder"],
        proprio_encoder=result["proprio_encoder"],
        action_encoder=result["action_encoder"],
        predictor=result["predictor"],
        decoder=result.get("decoder"),
        proprio_dim=model_cfg.proprio_emb_dim,
        action_dim=model_cfg.action_emb_dim,
        concat_dim=model_cfg.concat_dim,
        num_action_repeat=model_cfg.num_action_repeat,
        num_proprio_repeat=model_cfg.num_proprio_repeat,
    )
    model.to(device)
    model.eval()
    return model, model_cfg


def err_eval_single(model, z_pred, z_tgt):
    return {k: model.emb_criterion(z_pred[k], z_tgt[k]).item() for k in z_pred.keys()}


class ActionCalibration:
    """
    Maps raw (meters) particle/pusher xy -> pixel coords in the *displayed*
    (post-transform) image, replicating Genesis's PileSweepData rasterizer:
        pixel = xy_meters * to_pxl + ctr_in_PXL,  to_pxl = 1000 * resolution_scale
        ctr_in_PXL = (raw_W / 2, raw_H / 2)
    then rescaled from the raw raster resolution to the model's img_size.
    Verified against states.pth: with resolution_scale=1.0 every particle in
    frame 0 of episode 0 projects exactly onto its rasterized pixel.
    """

    def __init__(self, raw_h, raw_w, display_size, resolution_scale=1.0):
        self.to_pxl = 1e3 * resolution_scale
        self.ctr = np.array([round(raw_w / 2), round(raw_h / 2)], dtype=np.float32)
        self.scale = display_size / raw_w
        plate_px = (PLATE_SIZE_M[0] * self.to_pxl, PLATE_SIZE_M[1] * self.to_pxl)
        self.marker_size = (plate_px[0] * self.scale, max(plate_px[1] * self.scale, 3.0))

    def project(self, xy_meters):
        """xy_meters: (..., 2) -> pixel coords in the displayed image, (..., 2)."""
        return (np.asarray(xy_meters) * self.to_pxl + self.ctr) * self.scale


def get_raw_raster_size(dset):
    obs_dir = Path(dset.data_path) / "obses"
    first_file = next(iter(sorted(obs_dir.glob("episode_*.pth"))))
    shape = torch.load(first_file, map_location="cpu").shape  # (T, H, W, 3)
    return shape[1], shape[2]


def tint(img_chw, color_hex):
    """img_chw: (3,H,W) tensor in [-1,1] (equal-ish RGB). Returns (H,W,3) uint8 in `color_hex`."""
    gray = ((img_chw.mean(0) + 1) / 2).clamp(0, 1).numpy()  # (H,W) in [0,1]
    rgb = np.array(matplotlib.colors.to_rgb(color_hex))
    out = gray[..., None] * rgb[None, None, :]
    return out


def draw_action(ax, start_px, end_px, angle_rad, color, marker_size):
    for center in (start_px, end_px):
        rect = Rectangle(
            (center[0] - marker_size[0] / 2, center[1] - marker_size[1] / 2),
            marker_size[0], marker_size[1],
            fill=False, edgecolor=color, linewidth=1.3,
        )
        rect.set_transform(
            Affine2D().rotate_around(center[0], center[1], angle_rad) + ax.transData
        )
        ax.add_patch(rect)
    ax.annotate(
        "", xy=end_px, xytext=start_px,
        arrowprops=dict(arrowstyle="->", color=color, linewidth=1.3),
    )


def plot_rollout(gt_imgs, pred_imgs, n_ctx, action_starts_px, action_ends_px, angles, marker_size, out_path):
    """
    gt_imgs: (T, 3, H, W) in [-1, 1], ground-truth frames.
    pred_imgs: (T - n_ctx, 3, H, W), predicted frames aligned to gt_imgs[n_ctx:].
    action_{starts,ends}_px: (T-1, 2) pixel coords of the pusher for the push
        taken FROM frame t (None if no action overlay available).
    angles: (T-1,) pusher orientation in radians, or None.
    """
    T = gt_imgs.shape[0]
    fig, axes = plt.subplots(2, T, figsize=(1.9 * T, 4.2), squeeze=False)

    for t in range(T):
        ax_gt, ax_pred = axes[0, t], axes[1, t]

        gt_rgb = tint(gt_imgs[t], GT_COLOR)
        ax_gt.imshow(gt_rgb)
        ax_gt.set_title(f"t={t}", fontsize=9)

        if t < T - 1 and action_starts_px is not None:
            angle = angles[t] if angles is not None else 0.0
            draw_action(ax_gt, action_starts_px[t], action_ends_px[t], angle, "white", marker_size)

        if t < n_ctx:
            ax_pred.set_facecolor("#222222")
            ax_pred.text(0.5, 0.5, "context", color=CTX_COLOR, ha="center", va="center",
                         fontsize=8, transform=ax_pred.transAxes)
        else:
            pred_rgb = tint(pred_imgs[t - n_ctx], PRED_COLOR)
            ax_pred.imshow(pred_rgb)

        for ax, color in ((ax_gt, GT_COLOR), (ax_pred, PRED_COLOR if t >= n_ctx else CTX_COLOR)):
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_edgecolor(color)
                spine.set_linewidth(2.5)

    axes[0, 0].set_ylabel("Ground Truth", color=GT_COLOR, fontsize=10, fontweight="bold")
    axes[1, 0].set_ylabel("Predicted", color=PRED_COLOR, fontsize=10, fontweight="bold")

    fig.tight_layout(w_pad=0.3, h_pad=0.6)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


@torch.no_grad()
def evaluate(model, dset, model_cfg, device, num_rollout, min_horizon, seed_val, output_dir):
    np.random.seed(seed_val)
    output_dir.mkdir(parents=True, exist_ok=True)
    for stale in output_dir.glob("rollout_*.png"):
        stale.unlink()

    obs_type = model_cfg.env.dataset.get("obs_type", None)
    calib = None
    if model_cfg.has_decoder and obs_type == "occupancy":
        raw_h, raw_w = get_raw_raster_size(dset)
        calib = ActionCalibration(raw_h, raw_w, model_cfg.img_size)
    elif model_cfg.has_decoder:
        print(f"obs_type={obs_type!r}: skipping action overlay (only supported for 'occupancy').")

    min_horizon = min_horizon + model_cfg.num_hist
    num_past_variants = [(model_cfg.num_hist, "full_context"), (1, "single_frame_start")]

    metrics_by_variant = defaultdict(lambda: defaultdict(list))

    for idx in range(num_rollout):
        valid_traj = False
        while not valid_traj:
            traj_idx = np.random.randint(0, len(dset))
            obs, act, state, _ = dset[traj_idx]
            act = act.to(device)
            if obs["visual"].shape[0] > min_horizon * model_cfg.frameskip + 1:
                start = np.random.randint(0, obs["visual"].shape[0] - min_horizon * model_cfg.frameskip - 1)
            else:
                start = 0
            max_horizon = (obs["visual"].shape[0] - start - 1) // model_cfg.frameskip
            if max_horizon > min_horizon:
                valid_traj = True
                horizon = np.random.randint(min_horizon, max_horizon + 1)

        for k in obs.keys():
            obs[k] = obs[k][start : start + horizon * model_cfg.frameskip + 1 : model_cfg.frameskip]
        act_slice = act[start : start + horizon * model_cfg.frameskip]  # (horizon*frameskip, action_dim)
        act_reshaped = act_slice.reshape(horizon, model_cfg.frameskip * act_slice.shape[-1])

        action_starts_px = action_ends_px = angles_raw = None
        if calib is not None:
            action_mean = torch.as_tensor(dset.action_mean).repeat(model_cfg.frameskip).to(device)
            action_std = torch.as_tensor(dset.action_std).repeat(model_cfg.frameskip).to(device)
            act_raw = (act_reshaped * action_std + action_mean).cpu().numpy()  # (horizon, frameskip*4)
            start_xy = act_raw[:, 0:2]
            end_xy = act_raw[:, -2:]
            action_starts_px = calib.project(start_xy)
            action_ends_px = calib.project(end_xy)

            proprio_mean = torch.as_tensor(dset.proprio_mean).to(device)
            proprio_std = torch.as_tensor(dset.proprio_std).to(device)
            proprio_raw = (obs["proprio"].to(device) * proprio_std + proprio_mean).cpu().numpy()  # (T, 3)
            angles_raw = proprio_raw[:-1, 2]

        obs_g = {k: v[-1].unsqueeze(0).unsqueeze(0).to(device) for k, v in obs.items()}
        z_g = model.encode_obs(obs_g)
        actions = act_reshaped.unsqueeze(0)

        for n_past, variant in num_past_variants:
            obs_0 = {k: v[:n_past].unsqueeze(0).to(device) for k, v in obs.items()}
            z_obses, _ = model.rollout(obs_0, actions)
            z_obs_last = slice_trajdict_with_t(z_obses, start_idx=-1, end_idx=None)
            div_loss = err_eval_single(model, z_obs_last, z_g)
            for k, v in div_loss.items():
                metrics_by_variant[variant][f"z_{k}_err"].append(v)

            if model_cfg.has_decoder:
                visuals = model.decode_obs(z_obses)[0]["visual"][0].cpu()  # (T, 3, H, W)
                gt_imgs = obs["visual"]  # (T, 3, H, W)
                pred_imgs = visuals[n_past:]  # predicted frames only, aligned with gt_imgs[n_past:]
                img_scores = eval_images(pred_imgs.to(device), gt_imgs[n_past:].to(device))
                for k, v in img_scores.items():
                    metrics_by_variant[variant][f"img_{k}"].append(v.item())

                plot_rollout(
                    gt_imgs, pred_imgs, n_past,
                    action_starts_px, action_ends_px, angles_raw,
                    calib.marker_size if calib is not None else (0, 0),
                    output_dir / f"rollout_{idx:03d}_{variant}.png",
                )

    summary = {
        variant: {k: float(np.mean(v)) for k, v in metric_dict.items()}
        for variant, metric_dict in metrics_by_variant.items()
    }
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"\nSaved rollout plots and metrics.json to {output_dir}")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True, help="Hydra train run dir, e.g. outputs/2026-08-24/13-36-19")
    parser.add_argument("--epoch", default="latest", help="epoch number or 'latest'")
    parser.add_argument("--split", default="valid", choices=["train", "valid"])
    parser.add_argument("--num_rollout", type=int, default=10)
    parser.add_argument("--min_horizon", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_dir", default=None)
    args = parser.parse_args()

    model_dir = Path(args.model_dir).resolve()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    seed(args.seed)
    model, model_cfg = load_model(model_dir, args.epoch, device)

    print(f"Loading dataset from {model_cfg.env.dataset.data_path} ...")
    _, traj_dsets = hydra.utils.call(
        model_cfg.env.dataset,
        num_hist=model_cfg.num_hist,
        num_pred=model_cfg.num_pred,
        frameskip=model_cfg.frameskip,
    )
    dset = traj_dsets[args.split]

    output_dir = Path(args.output_dir) if args.output_dir else model_dir / f"eval_rollout_e{args.epoch}"
    evaluate(
        model=model,
        dset=dset,
        model_cfg=model_cfg,
        device=device,
        num_rollout=args.num_rollout,
        min_horizon=args.min_horizon,
        seed_val=args.seed,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    main()
