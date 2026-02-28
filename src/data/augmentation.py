"""Step 2.1: Physics-aware augmentation pipeline.

Class: ThermalGasAugment
Applies consistently to all three streams (image, mask, overlay) with the
same transform and same seed per sample.

Supported transforms:
- Horizontal flip ONLY (vertical not physically valid for upward-rising gas)
- Random gas intensity scaling on overlay pixels inside mask region: [0.8, 1.2]
- Gaussian noise on mask boundary zone only (erode mask 2px, add sigma=0.02 noise in ring)
- Random brightness/contrast on background region only (mask == 0)
- NO rotation, NO vertical flip, NO elastic distortion
"""

import cv2
import numpy as np

try:
    import torch
except ImportError:
    torch = None


class ThermalGasAugment:
    """Physics-aware augmentation for thermal gas emission data.

    All transforms are applied consistently across the three data streams
    (image, mask, overlay) using a shared random seed per sample.

    Args:
        p_hflip: probability of horizontal flip (default 0.5)
        intensity_range: (min, max) scaling factor for gas region in overlay
        boundary_noise_sigma: std of Gaussian noise added to mask boundary ring
        brightness_range: (min, max) brightness shift for background region
        contrast_range: (min, max) contrast factor for background region
        erosion_px: pixels to erode mask for boundary ring computation
    """

    def __init__(
        self,
        p_hflip: float = 0.5,
        intensity_range: tuple = (0.8, 1.2),
        boundary_noise_sigma: float = 0.02,
        brightness_range: tuple = (-20, 20),
        contrast_range: tuple = (0.9, 1.1),
        erosion_px: int = 2,
    ):
        self.p_hflip = p_hflip
        self.intensity_range = intensity_range
        self.boundary_noise_sigma = boundary_noise_sigma
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.erosion_px = erosion_px

    def __call__(self, image: np.ndarray, mask: np.ndarray, overlay: np.ndarray,
                 seed: int = None) -> tuple:
        """Apply augmentations to a single frame.

        Args:
            image: (H, W, 3) uint8 thermal image
            mask: (H, W) uint8 binary mask (0 or 255)
            overlay: (H, W, 3) uint8 RGB composite overlay
            seed: optional random seed for reproducibility

        Returns:
            (augmented_image, augmented_mask, augmented_overlay)
        """
        rng = np.random.RandomState(seed)

        image = image.copy()
        mask = mask.copy()
        overlay = overlay.copy()

        binary = (mask > 127).astype(np.uint8)

        if rng.random() < self.p_hflip:
            image = np.flip(image, axis=1).copy()
            mask = np.flip(mask, axis=1).copy()
            overlay = np.flip(overlay, axis=1).copy()
            binary = np.flip(binary, axis=1).copy()

        overlay = self._scale_gas_intensity(overlay, binary, rng)
        mask = self._add_boundary_noise(mask, binary, rng)
        image = self._adjust_background(image, binary, rng)
        overlay = self._adjust_background(overlay, binary, rng)

        return image, mask, overlay

    def _scale_gas_intensity(self, overlay: np.ndarray, binary: np.ndarray,
                             rng: np.random.RandomState) -> np.ndarray:
        factor = rng.uniform(*self.intensity_range)
        gas_mask_3c = np.stack([binary] * 3, axis=-1).astype(bool)
        overlay_float = overlay.astype(np.float32)
        overlay_float[gas_mask_3c] *= factor
        return np.clip(overlay_float, 0, 255).astype(np.uint8)

    def _add_boundary_noise(self, mask: np.ndarray, binary: np.ndarray,
                            rng: np.random.RandomState) -> np.ndarray:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (2 * self.erosion_px + 1, 2 * self.erosion_px + 1),
        )
        eroded = cv2.erode(binary, kernel, iterations=1)
        boundary_ring = binary - eroded

        if boundary_ring.sum() == 0:
            return mask

        noise = rng.normal(0, self.boundary_noise_sigma, mask.shape).astype(np.float32)
        mask_float = mask.astype(np.float32) / 255.0
        mask_float += noise * boundary_ring.astype(np.float32)
        mask_out = ((mask_float > 0.5).astype(np.uint8)) * 255
        return mask_out

    def _adjust_background(self, img: np.ndarray, binary: np.ndarray,
                           rng: np.random.RandomState) -> np.ndarray:
        brightness = rng.uniform(*self.brightness_range)
        contrast = rng.uniform(*self.contrast_range)

        bg_mask = (binary == 0)
        if img.ndim == 3:
            bg_mask_nd = np.stack([bg_mask] * img.shape[2], axis=-1)
        else:
            bg_mask_nd = bg_mask

        img_float = img.astype(np.float32)
        img_float[bg_mask_nd] = img_float[bg_mask_nd] * contrast + brightness
        return np.clip(img_float, 0, 255).astype(np.uint8)


if torch is not None:
    class ThermalGasAugmentTorch:
        """Wrapper that accepts and returns torch tensors.

        Converts tensors to numpy, applies ThermalGasAugment, converts back.
        Expects tensors in (C, H, W) format with values in [0, 1].
        """

        def __init__(self, **kwargs):
            self.augmentor = ThermalGasAugment(**kwargs)

        def __call__(self, image: torch.Tensor, mask: torch.Tensor,
                     overlay: torch.Tensor, seed: int = None) -> tuple:
            img_np = (image.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            mask_np = (mask.squeeze(0).numpy() * 255).astype(np.uint8)
            ovl_np = (overlay.permute(1, 2, 0).numpy() * 255).astype(np.uint8)

            img_aug, mask_aug, ovl_aug = self.augmentor(img_np, mask_np, ovl_np, seed)

            img_t = torch.from_numpy(img_aug).permute(2, 0, 1).float() / 255.0
            mask_t = torch.from_numpy(mask_aug).unsqueeze(0).float() / 255.0
            ovl_t = torch.from_numpy(ovl_aug).permute(2, 0, 1).float() / 255.0

            return img_t, mask_t, ovl_t
