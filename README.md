# Brain Hologram

Holographic rendering of spectral EEG activity in 3D voxel space. At a high level, we first apply a channel-wise Fourierr transform over the temporal dimension of EEG. This is followed by a (sort of) inverse Fourier transform in 3D space where the center of the spatial oscillators matches the EEG electrode positions. The hologram is then constructed by computing the interference pattern of all spatial oscillators, resulting in a voxelized representation of neural activity. Note that this is not a reconstruction of the underlying neural sources, but rather a visualization of neural spectral dynamics as measured in the cortex.

<div align="center">
  <video src="https://github.com/user-attachments/assets/6c4d42f7-69d8-4848-967e-22651c432024"></video>
</div>

## Instructions
1. Install dependencies from `requirements.txt`
2. Run `hologram.py` to compute voxel time-series (preferably on an NVIDIA GPU)
3. Run `viz.py` to start the interactive visualization
