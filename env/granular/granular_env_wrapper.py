import numpy as np
from torchvision import transforms

from utils import aggregate_dct
from .granular_env import GranularPushEnv

ENV_ACTION_DIM = 5  # [x_start, y_start, x_end, y_end, angle]


class GranularEnvWrapper(GranularPushEnv):
    CONTAINMENT_THRESHOLD = 0.9  # fraction of particles that must be within the target area
    TARGET_CENTER = (0.0, 0.0)   # matches arrange_particles_in_area()'s default center_xy

    def __init__(self, img_size=224, success_radius_tol_factor=0.5, **kwargs):
        super().__init__(**kwargs)
        self.action_dim = ENV_ACTION_DIM
        self.transform = transforms.Resize((img_size, img_size))
        self.success_radius_tol = success_radius_tol_factor * self.particle_size

    def eval_state(self, goal_state, cur_state):
        """
        goal_state, cur_state: (state_dim,) = (n_particles*7,) raw meters.
        "Success" = most of the current particle cloud has gathered inside
        the target area, not that particles occupy any exact positions.
        Target area = TARGET_CENTER, radius = sm.default_area_radius() (the
        same deterministic, box-geometry-derived value
        arrange_particles_in_area() itself defaults to) plus a small
        tolerance.

        Deliberately NOT derived from the goal_state's own realized radial
        spread: a goal cluster placed via arrange_particles_in_area()'s
        overlap-fallback path can settle with a handful of outlier particles
        pushed outside the intended radius, so a goal-derived target would
        be noisy and placement-realization-dependent - this way the target
        area is the same fixed region regardless of how any particular goal
        sample happened to settle.
        """
        cur_xy = np.asarray(cur_state).reshape(self.n_particles, 7)[:, :2]
        goal_xy = np.asarray(goal_state).reshape(self.n_particles, 7)[:, :2]
        center = np.asarray(self.TARGET_CENTER)

        target_radius = self.sm.default_area_radius() + self.success_radius_tol
        cur_radial = np.linalg.norm(cur_xy - center, axis=1)
        contained_fraction = float((cur_radial <= target_radius).mean())

        # permutation-invariant nearest-neighbor (Chamfer) distance to the goal cloud
        dist = np.linalg.norm(cur_xy[:, None, :] - goal_xy[None, :, :], axis=-1)
        chamfer_dist = float((dist.min(axis=1).mean() + dist.min(axis=0).mean()) / 2)

        return {
            "success": contained_fraction >= self.CONTAINMENT_THRESHOLD,
            "contained_fraction": contained_fraction,
            "chamfer_dist": chamfer_dist,
        }

    def sample_random_init_goal_states(self, seed):
        self.seed(seed)
        self.sm.shuffle_particles()
        self.sm.update_material_state()
        init_state = self._get_state()

        self.sm.arrange_particles_in_area()
        self.sm.update_material_state()
        goal_state = self._get_state()

        return init_state, goal_state

    def update_env(self, env_info):
        pass  # GenesisGranularDataset.get_frames() doesn't carry per-episode env info

    def prepare(self, seed, init_state):
        self.seed(seed)
        self.set_init_state(init_state)
        obs, state = self.reset()
        obs["visual"] = self.transform(obs["visual"]).permute(1, 2, 0)
        return obs, state

    def step_multiple(self, actions):
        obses, rewards, dones, infos = [], [], [], []
        for action in actions:
            o, r, d, info = self.step(action)
            o["visual"] = self.transform(o["visual"]).permute(1, 2, 0)
            obses.append(o)
            rewards.append(r)
            dones.append(d)
            infos.append(info)
        obses = aggregate_dct(obses)
        rewards = np.stack(rewards)
        dones = np.stack(dones)
        infos = aggregate_dct(infos)
        return obses, rewards, dones, infos

    def rollout(self, seed, init_state, actions):
        """
        seed: int
        init_state: (state_dim,)
        actions: (T, action_dim)
        obses: dict, values (T+1, H, W, C)
        states: (T+1, D)
        """
        obs, state = self.prepare(seed, init_state)
        obses, rewards, dones, infos = self.step_multiple(actions)
        for k in obses.keys():
            obses[k] = np.vstack([np.expand_dims(obs[k], 0), obses[k]])
        states = np.vstack([np.expand_dims(state, 0), infos["state"]])
        return obses, states
