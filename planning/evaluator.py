import os
import torch
import imageio
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from einops import rearrange, repeat
from utils import (
    cfg_to_dict,
    seed,
    slice_trajdict_with_t,
    aggregate_dct,
    move_to_device,
    concat_trajdict,
)
from torchvision import utils
from eval_open_loop import ActionCalibration, draw_action

GT_COLOR = "#2ca02c"    # green - executed in the real env
PRED_COLOR = "#1f77b4"  # blue - imagined by the world model
CTX_COLOR = "#888888"   # gray - frame the world model was given, not predicted
GOAL_COLOR = "#ff7f0e"  # orange - target the planner was optimizing for


def _get_action_calibration(env, display_size):
    """
    Builds an ActionCalibration (see eval_open_loop.py) from the live env if
    it's our Genesis-backed occupancy-raster granular env - duck-typed via
    the `rasterizer` attribute so this is a no-op (returns None) for every
    other env (wall, pusht, ...), which have no such projection.
    """
    try:
        raw_env = env.envs[0].unwrapped
        raw_h, raw_w = raw_env.rasterizer._output_grid.shape
        return ActionCalibration(raw_h, raw_w, display_size)
    except AttributeError:
        return None


def _tint(img_chw, color_hex):
    """img_chw: (3,H,W) tensor in [-1,1]. Returns (H,W,3) array in `color_hex`."""
    gray = ((img_chw.mean(0) + 1) / 2).clamp(0, 1).detach().numpy()
    rgb = np.array(matplotlib.colors.to_rgb(color_hex))
    return gray[..., None] * rgb[None, None, :]


def _plot_rollout_compare_labeled(
    e_visuals, i_visuals, goal_visual, successes, n_ctx, filename,
    exec_actions=None, calib=None,
):
    """
    e_visuals, i_visuals: (b, T, c, h, w) in [-1, 1] - executed vs. imagined rollout.
    goal_visual: (b, 1, c, h, w) in [-1, 1] - the planning target.
    successes: (b,) bool.
    n_ctx: how many of i_visuals' leading columns are the world model's given
        context (re-encoded/decoded, not genuine predictions).
    exec_actions: (b, T-1, action_dim) raw (denormalized, meters) actions
        actually executed, action t taking column t -> t+1. None if the env
        doesn't support the pixel calibration (see _get_action_calibration).
    calib: ActionCalibration for projecting exec_actions into pixel space, or None.
    """
    b, T = e_visuals.shape[:2]
    for idx in range(b):
        fig, axes = plt.subplots(2, T + 1, figsize=(1.9 * (T + 1), 4.2), squeeze=False)
        goal_chw = goal_visual[idx, 0]

        for t in range(T):
            ax_gt, ax_pred = axes[0, t], axes[1, t]
            ax_gt.imshow(_tint(e_visuals[idx, t], GT_COLOR))
            ax_gt.set_title(f"t={t}", fontsize=9)

            if calib is not None and exec_actions is not None and t < exec_actions.shape[1]:
                x0, y0, x1, y1 = exec_actions[idx, t, :4]
                if exec_actions.shape[-1] >= 5:
                    angle = float(exec_actions[idx, t, 4])  # true plate orientation
                else:
                    # legacy 4D actions have no separate angle dim - approximate
                    # with the travel direction (see granular_env.py's old step()).
                    angle = float(np.arctan2(y1 - y0, x1 - x0))
                start_px = calib.project(np.array([x0, y0]))
                end_px = calib.project(np.array([x1, y1]))
                draw_action(ax_gt, start_px, end_px, angle, "white", calib.marker_size)

            if t < n_ctx:
                ax_pred.set_facecolor("#222222")
                ax_pred.text(0.5, 0.5, "context", color=CTX_COLOR, ha="center", va="center",
                             fontsize=8, transform=ax_pred.transAxes)
                pred_color = CTX_COLOR
            else:
                ax_pred.imshow(_tint(i_visuals[idx, t], PRED_COLOR))
                pred_color = PRED_COLOR

            for ax, color in ((ax_gt, GT_COLOR), (ax_pred, pred_color)):
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_edgecolor(color)
                    spine.set_linewidth(2.5)

        for row in (0, 1):
            ax = axes[row, T]
            ax.imshow(_tint(goal_chw, GOAL_COLOR))
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_edgecolor(GOAL_COLOR)
                spine.set_linewidth(3.0)
        axes[0, T].set_title("GOAL", fontsize=9, color=GOAL_COLOR, fontweight="bold")

        axes[0, 0].set_ylabel("Ground Truth\n(executed)", color=GT_COLOR, fontsize=10, fontweight="bold")
        axes[1, 0].set_ylabel("Predicted\n(world model)", color=PRED_COLOR, fontsize=10, fontweight="bold")

        success_tag = "success" if successes[idx] else "failure"
        fig.suptitle(f"sample {idx} - {success_tag}", fontsize=11)
        fig.tight_layout(w_pad=0.3, h_pad=0.6, rect=[0, 0, 1, 0.95])
        out_path = f"{filename}.png" if b == 1 else f"{filename}_{idx}.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)


class PlanEvaluator:  # evaluator for planning
    def __init__(
        self,
        obs_0,
        obs_g,
        state_0,
        state_g,
        env,
        wm,
        frameskip,
        seed,
        preprocessor,
        n_plot_samples,
    ):
        self.obs_0 = obs_0
        self.obs_g = obs_g
        self.state_0 = state_0
        self.state_g = state_g
        self.env = env
        self.wm = wm
        self.frameskip = frameskip
        self.seed = seed
        self.preprocessor = preprocessor
        self.n_plot_samples = n_plot_samples
        self.device = next(wm.parameters()).device

        self.plot_full = False  # plot all frames or frames after frameskip

    def assign_init_cond(self, obs_0, state_0):
        self.obs_0 = obs_0
        self.state_0 = state_0

    def assign_goal_cond(self, obs_g, state_g):
        self.obs_g = obs_g
        self.state_g = state_g

    def get_init_cond(self):
        return self.obs_0, self.state_0

    def _get_trajdict_last(self, dct, length):
        new_dct = {}
        for key, value in dct.items():
            new_dct[key] = self._get_traj_last(value, length)
        return new_dct

    def _get_traj_last(self, traj_data, length):
        last_index = np.where(length == np.inf, -1, length - 1)
        last_index = last_index.astype(int)
        if isinstance(traj_data, torch.Tensor):
            traj_data = traj_data[np.arange(traj_data.shape[0]), last_index].unsqueeze(
                1
            )
        else:
            traj_data = np.expand_dims(
                traj_data[np.arange(traj_data.shape[0]), last_index], axis=1
            )
        return traj_data

    def _mask_traj(self, data, length):
        """
        Zero out everything after specified indices for each trajectory in the tensor.
        data: tensor
        """
        result = data.clone()  # Clone to preserve the original tensor
        for i in range(data.shape[0]):
            if length[i] != np.inf:
                result[i, int(length[i]) :] = 0
        return result

    def eval_actions(
        self, actions, action_len=None, filename="output", save_video=False
    ):
        """
        actions: detached torch tensors on cuda
        Returns
            metrics, and feedback from env
        """
        n_evals = actions.shape[0]
        if action_len is None:
            action_len = np.full(n_evals, np.inf)
        # rollout in wm
        trans_obs_0 = move_to_device(
            self.preprocessor.transform_obs(self.obs_0), self.device
        )
        trans_obs_g = move_to_device(
            self.preprocessor.transform_obs(self.obs_g), self.device
        )
        with torch.no_grad():
            i_z_obses, _ = self.wm.rollout(
                obs_0=trans_obs_0,
                act=actions,
            )
        i_final_z_obs = self._get_trajdict_last(i_z_obses, action_len + 1)

        # rollout in env
        exec_actions = rearrange(
            actions.cpu(), "b t (f d) -> b (t f) d", f=self.frameskip
        )
        exec_actions = self.preprocessor.denormalize_actions(exec_actions).numpy()
        e_obses, e_states = self.env.rollout(self.seed, self.state_0, exec_actions)
        e_visuals = e_obses["visual"]
        e_final_obs = self._get_trajdict_last(e_obses, action_len * self.frameskip + 1)
        e_final_state = self._get_traj_last(e_states, action_len * self.frameskip + 1)[
            :, 0
        ]  # reduce dim back

        # compute eval metrics
        logs, successes = self._compute_rollout_metrics(
            e_state=e_final_state,
            e_obs=e_final_obs,
            i_z_obs=i_final_z_obs,
        )

        # plot trajs
        if self.wm.decoder is not None:
            i_visuals = self.wm.decode_obs(i_z_obses)[0]["visual"]
            i_visuals = self._mask_traj(
                i_visuals, action_len + 1
            )  # we have action_len + 1 states
            e_visuals = self.preprocessor.transform_obs_visual(e_visuals)
            e_visuals = self._mask_traj(e_visuals, action_len * self.frameskip + 1)
            self._plot_rollout_compare(
                e_visuals=e_visuals,
                i_visuals=i_visuals,
                successes=successes,
                save_video=save_video,
                filename=filename,
                exec_actions=exec_actions,
            )

        return logs, successes, e_obses, e_states

    def _compute_rollout_metrics(self, e_state, e_obs, i_z_obs):
        """
        Args
            e_state
            e_obs
            i_z_obs
        Return
            logs
            successes
        """
        eval_results = self.env.eval_state(self.state_g, e_state)
        successes = eval_results['success']

        logs = {
            f"success_rate" if key == "success" else f"mean_{key}": np.mean(value) if key != "success" else np.mean(value.astype(float))
            for key, value in eval_results.items()
        }

        print("Success rate: ", logs['success_rate'])
        print(eval_results)

        visual_dists = np.linalg.norm(e_obs["visual"] - self.obs_g["visual"], axis=1)
        mean_visual_dist = np.mean(visual_dists)
        proprio_dists = np.linalg.norm(e_obs["proprio"] - self.obs_g["proprio"], axis=1)
        mean_proprio_dist = np.mean(proprio_dists)

        e_obs = move_to_device(self.preprocessor.transform_obs(e_obs), self.device)
        e_z_obs = self.wm.encode_obs(e_obs)
        div_visual_emb = torch.norm(e_z_obs["visual"] - i_z_obs["visual"]).item()
        div_proprio_emb = torch.norm(e_z_obs["proprio"] - i_z_obs["proprio"]).item()

        logs.update({
            "mean_visual_dist": mean_visual_dist,
            "mean_proprio_dist": mean_proprio_dist,
            "mean_div_visual_emb": div_visual_emb,
            "mean_div_proprio_emb": div_proprio_emb,
        })

        return logs, successes

    def _plot_rollout_compare(
        self, e_visuals, i_visuals, successes, save_video=False, filename="", exec_actions=None
    ):
        """
        i_visuals may have less frames than e_visuals due to frameskip, so pad accordingly
        e_visuals: (b, t, h, w, c)
        i_visuals: (b, t, h, w, c)
        goal: (b, h, w, c)
        exec_actions: (b, t-1, action_dim) raw (denormalized) actions actually
            executed, for overlaying start/end pusher pose + direction on the
            executed-frame row. None (or an unsupported env) silently skips it.
        """
        e_visuals = e_visuals[: self.n_plot_samples]
        i_visuals = i_visuals[: self.n_plot_samples]
        if exec_actions is not None:
            exec_actions = exec_actions[: self.n_plot_samples]
        goal_visual = self.obs_g["visual"][: self.n_plot_samples]
        goal_visual = self.preprocessor.transform_obs_visual(goal_visual)

        i_visuals = i_visuals.unsqueeze(2)
        i_visuals = torch.cat(
            [i_visuals] + [i_visuals] * (self.frameskip - 1),
            dim=2,
        )  # pad i_visuals (due to frameskip)
        i_visuals = rearrange(i_visuals, "b t n c h w -> b (t n) c h w")
        i_visuals = i_visuals[:, : i_visuals.shape[1] - (self.frameskip - 1)]

        correction = 0.3  # to distinguish env visuals and imagined visuals

        if save_video:
            for idx in range(e_visuals.shape[0]):
                success_tag = "success" if successes[idx] else "failure"
                frames = []
                for i in range(e_visuals.shape[1]):
                    e_obs = e_visuals[idx, i, ...]
                    i_obs = i_visuals[idx, i, ...]
                    e_obs = torch.cat(
                        [e_obs.cpu(), goal_visual[idx, 0] - correction], dim=2
                    )
                    i_obs = torch.cat(
                        [i_obs.cpu(), goal_visual[idx, 0] - correction], dim=2
                    )
                    frame = torch.cat([e_obs - correction, i_obs], dim=1)
                    frame = rearrange(frame, "c w1 w2 -> w1 w2 c")
                    frame = rearrange(frame, "w1 w2 c -> (w1) w2 c")
                    frame = frame.detach().cpu().numpy()
                    frames.append(frame)
                video_writer = imageio.get_writer(
                    f"{filename}_{idx}_{success_tag}.mp4", fps=12
                )

                for frame in frames:
                    frame = frame * 2 - 1 if frame.min() >= 0 else frame
                    video_writer.append_data(
                        (((np.clip(frame, -1, 1) + 1) / 2) * 255).astype(np.uint8)
                    )
                video_writer.close()

        # pad i_visuals or subsample e_visuals
        if not self.plot_full:
            e_visuals = e_visuals[:, :: self.frameskip]
            i_visuals = i_visuals[:, :: self.frameskip]

        n_columns = e_visuals.shape[1]
        assert (
            i_visuals.shape[1] == n_columns
        ), f"Rollout lengths do not match, {e_visuals.shape[1]} and {i_visuals.shape[1]}"

        n_ctx = self.obs_0["visual"].shape[1]  # frames the world model was given, not predicted
        calib = _get_action_calibration(self.env, e_visuals.shape[-1])
        _plot_rollout_compare_labeled(
            e_visuals=e_visuals.cpu(),
            i_visuals=i_visuals.cpu(),
            goal_visual=goal_visual.cpu(),
            successes=successes,
            n_ctx=n_ctx,
            filename=filename,
            exec_actions=exec_actions,
            calib=calib,
        )
