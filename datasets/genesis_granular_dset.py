import torch
import numpy as np
from einops import rearrange
from pathlib import Path
from typing import Callable, Optional
from .traj_dset import TrajDataset, get_train_val_sliced


class GenesisGranularDataset(TrajDataset):
    """
    Loads Genesis granular-pile rollouts exported by
    Genesis/training/export_dino_wm_dataset.py, i.e.:
        {data_path}/{obs_type}/states.pth      (N, T, n_particles * 7)
        {data_path}/{obs_type}/actions.pth     (N, T, action_dim)
        {data_path}/{obs_type}/proprios.pth    (N, T, proprio_dim)
        {data_path}/{obs_type}/seq_lengths.pth (N,)
        {data_path}/{obs_type}/obses/episode_{idx:06d}.pth  (T, H, W, 3) uint8
    obs_type selects which observation modality to train on:
        "occupancy" - top-down rasterized particle occupancy (no camera needed)
        "rendered"  - real camera frames captured during collection
    """

    def __init__(
        self,
        data_path: str = "data/dino_wm_preview_conv",
        obs_type: str = "occupancy",
        n_rollout: Optional[int] = None,
        transform: Optional[Callable] = None,
        normalize_action: bool = False,
        action_scale: float = 1.0,
    ):
        self.data_path = Path(data_path) / obs_type
        self.transform = transform
        self.normalize_action = normalize_action

        self.states = torch.load(self.data_path / "states.pth").float()
        self.actions = torch.load(self.data_path / "actions.pth").float()
        self.actions = self.actions / action_scale  # scaled back up in env
        self.proprios = torch.load(self.data_path / "proprios.pth").float()
        self.seq_lengths = torch.load(self.data_path / "seq_lengths.pth").long()

        self.n_rollout = n_rollout
        n = self.n_rollout if self.n_rollout else len(self.states)

        self.states = self.states[:n]
        self.actions = self.actions[:n]
        self.proprios = self.proprios[:n]
        self.seq_lengths = self.seq_lengths[:n]

        self.action_dim = self.actions.shape[-1]
        self.state_dim = self.states.shape[-1]
        self.proprio_dim = self.proprios.shape[-1]

        if normalize_action:
            self.action_mean, self.action_std = self.get_data_mean_std(
                self.actions, self.seq_lengths
            )
            self.state_mean, self.state_std = self.get_data_mean_std(
                self.states, self.seq_lengths
            )
            self.proprio_mean, self.proprio_std = self.get_data_mean_std(
                self.proprios, self.seq_lengths
            )
        else:
            self.action_mean = torch.zeros(self.action_dim)
            self.action_std = torch.ones(self.action_dim)
            self.state_mean = torch.zeros(self.state_dim)
            self.state_std = torch.ones(self.state_dim)
            self.proprio_mean = torch.zeros(self.proprio_dim)
            self.proprio_std = torch.ones(self.proprio_dim)

        self.actions = (self.actions - self.action_mean) / self.action_std
        self.proprios = (self.proprios - self.proprio_mean) / self.proprio_std

    def get_data_mean_std(self, data, traj_lengths):
        all_data = []
        for traj in range(len(traj_lengths)):
            traj_len = traj_lengths[traj]
            all_data.append(data[traj, :traj_len])
        all_data = torch.vstack(all_data)
        data_mean = torch.mean(all_data, dim=0)
        data_std = torch.std(all_data, dim=0) + 1e-6
        return data_mean, data_std

    def get_seq_length(self, idx):
        return self.seq_lengths[idx]

    def get_all_actions(self):
        result = []
        for i in range(len(self.seq_lengths)):
            T = self.seq_lengths[i]
            result.append(self.actions[i, :T, :])
        return torch.cat(result, dim=0)

    def get_frames(self, idx, frames):
        obs_file = self.data_path / "obses" / f"episode_{idx:06d}.pth"
        image = torch.load(obs_file)  # (T, H, W, 3) uint8
        image = image[frames]
        image = rearrange(image, "T H W C -> T C H W").float() / 255.0
        if self.transform:
            image = self.transform(image)

        proprio = self.proprios[idx, frames]
        act = self.actions[idx, frames]
        state = self.states[idx, frames]

        obs = {"visual": image, "proprio": proprio}
        return obs, act, state, {}

    def __getitem__(self, idx):
        return self.get_frames(idx, range(self.get_seq_length(idx)))

    def __len__(self):
        return len(self.seq_lengths)

    def preprocess_imgs(self, imgs):
        if isinstance(imgs, np.ndarray):
            raise NotImplementedError
        elif isinstance(imgs, torch.Tensor):
            return rearrange(imgs, "b h w c -> b c h w")


def load_genesis_granular_slice_train_val(
    transform,
    n_rollout=None,
    data_path="data/dino_wm_preview_conv",
    obs_type="occupancy",
    normalize_action=True,
    split_ratio=0.9,
    num_hist=0,
    num_pred=0,
    frameskip=0,
):
    dset = GenesisGranularDataset(
        n_rollout=n_rollout,
        transform=transform,
        data_path=data_path,
        obs_type=obs_type,
        normalize_action=normalize_action,
    )
    dset_train, dset_val, train_slices, val_slices = get_train_val_sliced(
        traj_dataset=dset,
        train_fraction=split_ratio,
        num_frames=num_hist + num_pred,
        frameskip=frameskip,
    )

    datasets = {"train": train_slices, "valid": val_slices}
    traj_dset = {"train": dset_train, "valid": dset_val}
    return datasets, traj_dset
