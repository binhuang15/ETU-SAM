"""Paper §3.3.1 augmentation only."""
from __future__ import annotations

import numpy as np
from batchgenerators.transforms.abstract_transforms import Compose
from batchgenerators.transforms.color_transforms import BrightnessTransform, GammaTransform
from batchgenerators.transforms.noise_transforms import GaussianNoiseTransform
from batchgenerators.transforms.spatial_transforms import ResizeTransform, SpatialTransform


def paper_transforms(patch_size=(256, 256)):
    rot = 45 / 360.0 * 2 * np.pi
    spatial = Compose(
        [
            ResizeTransform(target_size=patch_size),
            SpatialTransform(
                patch_size,
                [i // 2 for i in patch_size],
                do_elastic_deform=False,
                do_rotation=True,
                angle_x=(-rot, rot),
                angle_y=(-rot, rot),
                do_scale=True,
                scale=(0.75, 1.25),
                border_mode_data="constant",
                border_cval_data=0,
                order_data=3,
                border_mode_seg="constant",
                border_cval_seg=0,
                order_seg=0,
                random_crop=False,
                data_key="data",
                label_key="seg",
                p_el_per_sample=0.0,
                p_scale_per_sample=0.5,
                p_rot_per_sample=0.5,
            ),
        ]
    )
    intensity = Compose(
        [
            BrightnessTransform((-0.1, 0.1), 0, per_channel=False, data_key="data", p_per_sample=0.5),
            GammaTransform(gamma_range=(0.5, 1.5), invert_image=False, per_channel=False, p_per_sample=0.5),
            GaussianNoiseTransform(noise_variance=(0.01, 0.2), p_per_sample=0.5),
        ]
    )
    return spatial, intensity
