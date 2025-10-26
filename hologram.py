from itertools import product

import jax
import jax.numpy as jnp
import mne
import numpy as np
from mne.datasets import eegbci
from mne.io import read_raw
from tqdm.auto import tqdm

FMIN = 4
FMAX = 30
NFREQ = 26
SPEC_METHOD = "morlet"  # or "multitaper"
RESOLUTION = 40  # reduce to speed up computation

paths = eegbci.load_data(1, [1, 2])
raw: mne.io.Raw = mne.concatenate_raws([read_raw(p, preload=True) for p in paths])
eegbci.standardize(raw)
raw.set_montage("standard_1005")

tfr = raw.compute_tfr(method=SPEC_METHOD, freqs=np.linspace(FMIN, FMAX, NFREQ), n_jobs=-1)


@jax.jit
def _update_voxels(voxels, dist, tfr_data_ch_freq, freq_scale_val, ch_idx):
    return voxels + jnp.sin(dist[ch_idx, None] * freq_scale_val) * tfr_data_ch_freq


def hologram(tfr, max_steps: int = -1, resolution: int = 40, scale_low: float = 0.5, scale_high: float = 2):
    pos = jnp.array([ch["loc"][:3] for ch in tfr.info["chs"]], dtype=jnp.float32)
    vmin, vmax = pos.min(axis=0), pos.max(axis=0)
    xyz_ch_pos = ((pos - vmin) / (vmax - vmin) * resolution).astype(jnp.int32)

    tfr_data = jnp.array(tfr.get_data(), dtype=jnp.float32)
    tfr_data = tfr_data / (jnp.linalg.norm(tfr_data, axis=(0, 1)) + 1e-6)
    tfr_data = tfr_data.reshape(len(pos), len(tfr.freqs), -1, 1, 1, 1)
    tfr_data = tfr_data[:, :, :max_steps]

    idxs = jnp.arange(resolution, dtype=jnp.int32)
    xs, ys, zs = jnp.meshgrid(idxs, idxs, idxs, indexing="ij")
    xyz_idxs = xyz_ch_pos.reshape(-1, 3, 1, 1, 1)
    dist = jnp.sqrt(
        (xs[None] - xyz_idxs[:, 0]) ** 2 + (ys[None] - xyz_idxs[:, 1]) ** 2 + (zs[None] - xyz_idxs[:, 2]) ** 2
    ).astype(jnp.float32)

    freq_scale = jnp.linspace(scale_low, scale_high, len(tfr.freqs), dtype=jnp.float32)
    voxels = jnp.zeros((tfr_data.shape[2], resolution, resolution, resolution), dtype=jnp.float32)

    # standard Python loop, keeps memory safe
    total = tfr_data.shape[0] * tfr_data.shape[1]
    for ch_idx, freq_idx in tqdm(product(range(tfr_data.shape[0]), range(tfr_data.shape[1])), total=total):
        voxels = _update_voxels(voxels, dist, tfr_data[ch_idx, freq_idx], freq_scale[freq_idx], ch_idx)

    return voxels, xyz_ch_pos


voxels, ch_pos = hologram(tfr, resolution=RESOLUTION)
np.save("voxels.npy", voxels)
np.save("ch_pos.npy", ch_pos)
