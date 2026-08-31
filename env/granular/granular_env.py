"""
Interactive (reset/step, one push at a time) wrapper around Genesis's
SandboxManipulation, for closed-loop MPC/CEM planning with a trained
DINO-WM checkpoint. Defaults match the "cube n=30 size=0.012" granular-pile
setup this repo's dataset (data/dino_wm_preview_conv, obs_type=occupancy) was
collected and trained with - see Genesis/data_collection.py's defaults.

Only ONE instance may exist per process: Genesis's gs.init() (inside
SandboxManipulation.__init__) is a process-wide singleton, so plan.py must be
run with n_evals=1 for this env (see conf/plan_granular.yaml).
"""
import sys
from pathlib import Path

import gym
import numpy as np
import torch
import yaml
from gym import spaces

GENESIS_ROOT = Path(__file__).resolve().parents[2].parent / "Genesis"
for _p in (str(GENESIS_ROOT), str(GENESIS_ROOT / "training")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from sandbox_manipulation import SandboxManipulation  # noqa: E402
from export_dino_wm_dataset import make_rasterizer, rasterize_particle_frame  # noqa: E402

DEFAULT_MATERIAL_SETTING = {
    "particle_friction": 0.12,
    "sampled_particle_friction": None,
    "particle_density": 750,
    "sampled_particle_density": None,
    "box_friction": 0.12,
}

_INSTANCE_EXISTS = False


class GranularPushEnv(gym.Env):
    def __init__(
        self,
        shape="cube",
        n_particles=30,
        particle_size=0.012,
        material_setting=None,
        resolution_scale=1.0,
        debug=False,
    ):
        global _INSTANCE_EXISTS
        if _INSTANCE_EXISTS:
            raise RuntimeError(
                "GranularPushEnv already constructed in this process - Genesis's "
                "gs.init() is a process-wide singleton, so only n_evals=1 is "
                "supported when planning against this env (see conf/plan_granular.yaml)."
            )
        _INSTANCE_EXISTS = True

        super().__init__()
        config_path = GENESIS_ROOT / "configs" / "basic.yaml"
        with open(config_path) as f:
            config = yaml.safe_load(f)
        config["material"]["shape"] = shape
        config["material"]["n_particles"] = n_particles
        config["material"]["particle_size"] = particle_size

        self.sm = SandboxManipulation(config=config, n_envs=1, debug=debug)
        # gs.init() (just run above) sets torch's process-wide default device to
        # cuda as a side effect - Genesis's own internals always pass an explicit
        # device=gs.device, so they don't rely on that default, but the rest of
        # dino_wm (Preprocessor, evaluator, ...) implicitly assumes CPU-default
        # bare torch.tensor(...) calls. Reset it so the rest of the pipeline keeps
        # working the way it does for every other (non-Genesis) env.
        torch.set_default_device("cpu")
        self.sm.build()
        self.sm.set_material_properties(material_setting or DEFAULT_MATERIAL_SETTING)

        self.n_particles = n_particles
        self.particle_size = particle_size
        self.state_dim = n_particles * 7
        self.action_dim = 5  # [x_start, y_start, x_end, y_end, angle], meters/radians
        self.proprio_dim = 3  # [x_start, y_start, angle] of the last push taken

        # PileSweepData's rasterizer is built for offline (no-GPU) use - it draws
        # into the grid via cv2 on a numpy view, which requires CPU tensors. But
        # gs.init() (just run inside SandboxManipulation above) sets a process-wide
        # default torch device of cuda, which bare `torch.tensor(...)` calls inside
        # _create_grids() would otherwise pick up. Force CPU for the tensors it
        # allocates so the rasterizer stays entirely off the Genesis device context.
        with torch.device("cpu"):
            self.rasterizer = make_rasterizer(self.sm._config, resolution_scale, soft_particle_occupancy=False)
        h, w = self.rasterizer._output_grid.shape

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.action_dim,), dtype=np.float32)
        self.observation_space = spaces.Box(low=0, high=255, shape=(h, w, 3), dtype=np.uint8)

        self._rng = np.random.default_rng(0)
        self._last_proprio = np.zeros(self.proprio_dim, dtype=np.float32)
        self.reset_to_state = None

    # -- plumbing used by GranularEnvWrapper --

    def seed(self, seed):
        self._rng = np.random.default_rng(seed)

    def set_init_state(self, init_state):
        self.reset_to_state = None if init_state is None else np.asarray(init_state, dtype=np.float32)

    def reset(self):
        if self.reset_to_state is not None:
            state = torch.from_numpy(self.reset_to_state).reshape(self.n_particles, 7)
            self.sm.set_particle_state(state)
        else:
            self.sm.shuffle_particles()
        self.sm.update_material_state()
        self._last_proprio = np.zeros(self.proprio_dim, dtype=np.float32)
        return self._get_obs(), self._get_state()

    def step(self, action):
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        x0, y0, x1, y1, angle = action[:5]
        z = self.sm._operation_height
        device = self.sm._particle_state.device
        p_start = torch.tensor([[x0, y0, z]], dtype=torch.float32, device=device)
        p_stop = torch.tensor([[x1, y1, z]], dtype=torch.float32, device=device)
        angle_t = torch.tensor([angle], dtype=torch.float32, device=device)

        self.sm.execute_action(p_start, p_stop, angle_t)
        self.sm.update_material_state()

        self._last_proprio = np.array([x0, y0, angle], dtype=np.float32)
        obs = self._get_obs()
        state = self._get_state()
        info = {"state": state}
        return obs, 0.0, False, info

    def _get_state(self):
        return self.sm._particle_state[0].detach().cpu().numpy().reshape(-1).astype(np.float32)

    def _get_obs(self):
        particle_state_m = self.sm._particle_state[0].detach().cpu()
        with torch.device("cpu"):
            img = rasterize_particle_frame(self.rasterizer, particle_state_m, self.sm._config)  # (H,W,3) uint8
        visual = img.permute(2, 0, 1).float()  # (C,H,W), 0-255 - resized in the wrapper
        proprio = torch.from_numpy(self._last_proprio.copy())
        return {"visual": visual, "proprio": proprio}
