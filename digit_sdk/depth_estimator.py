import math
import os

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------------------------------------------------
# Torch DCT helpers (2D DCT-II / DCT-III via FFT)
# Adapted from zh217/torch-dct. Matches scipy.fftpack.dct/idct
# with norm="ortho".
# ------------------------------------------------------------------


def _dct1d(x: torch.Tensor) -> torch.Tensor:
    """1D DCT-II along last dimension via FFT (ortho norm).

    Matches scipy.fftpack.dct(x, type=2, norm='ortho').
    """
    N = x.shape[-1]
    even = x[..., ::2]
    odd = x[..., 1::2].flip(-1)
    v = torch.cat([even, odd], dim=-1)
    Vc = torch.fft.fft(v, dim=-1)
    k = -torch.arange(N, dtype=x.dtype, device=x.device) * math.pi / (2 * N)
    W_r = torch.cos(k)
    W_i = torch.sin(k)
    V = Vc.real * W_r - Vc.imag * W_i
    V[..., 0] /= math.sqrt(N) * 2
    V[..., 1:] /= math.sqrt(N / 2) * 2
    return 2 * V


def _idct1d(x: torch.Tensor) -> torch.Tensor:
    """1D DCT-III (inverse DCT-II) via FFT (ortho norm).

    _idct1d(_dct1d(x)) == x to floating-point precision.
    """
    N = x.shape[-1]
    X_v = x / 2
    X_vc = X_v.clone()
    X_vc[..., 0] *= math.sqrt(N) * 2
    X_vc[..., 1:] *= math.sqrt(N / 2) * 2
    k = torch.arange(N, dtype=x.dtype, device=x.device) * math.pi / (2 * N)
    W_r = torch.cos(k)
    W_i = torch.sin(k)
    V_t_r = X_vc
    zeros = torch.zeros(*x.shape[:-1], 1, dtype=x.dtype, device=x.device)
    flipped = -X_vc.flip(-1)
    V_t_i = torch.cat([zeros, flipped[..., :-1]], dim=-1)
    V_r = V_t_r * W_r - V_t_i * W_i
    V_i = V_t_r * W_i + V_t_i * W_r
    V = torch.complex(V_r, V_i)
    v = torch.fft.irfft(V, n=N, dim=-1)
    result = torch.zeros_like(v)
    half = N - (N // 2)
    result[..., ::2] = v[..., :half]
    result[..., 1::2] = v.flip(-1)[..., :N // 2]
    return result


def _dct2(x: torch.Tensor) -> torch.Tensor:
    """2D DCT-II (separable: rows then cols)."""
    x = _dct1d(x)
    x = _dct1d(x.T).T
    return x


def _idct2(x: torch.Tensor) -> torch.Tensor:
    """2D IDCT-III (separable: cols then rows, exact inverse of _dct2)."""
    x = _idct1d(x.T).T
    x = _idct1d(x)
    return x


# ------------------------------------------------------------------
# Torch binary morphology helpers
# Replaces scipy.ndimage.binary_erosion / binary_closing.
# ------------------------------------------------------------------


def _binary_dilate_torch(x: torch.Tensor, kernel_size: int) -> torch.Tensor:
    """Binary dilation via max_pool2d. x: (H, W) bool. Returns (H, W) bool."""
    if kernel_size <= 1:
        return x
    if kernel_size % 2 == 0:
        kernel_size += 1
    pad = kernel_size // 2
    x_4d = x.float().unsqueeze(0).unsqueeze(0)
    dilated = F.max_pool2d(x_4d, kernel_size, stride=1, padding=pad)
    return dilated.squeeze() > 0.5


def _binary_erosion_torch(x: torch.Tensor, kernel_size: int) -> torch.Tensor:
    """Binary erosion via min-pool = -max_pool(-x). x: (H, W) bool."""
    if kernel_size <= 1:
        return x
    if kernel_size % 2 == 0:
        kernel_size += 1
    pad = kernel_size // 2
    x_4d = x.float().unsqueeze(0).unsqueeze(0)
    eroded = -F.max_pool2d(-x_4d, kernel_size, stride=1, padding=pad)
    return eroded.squeeze() > 0.5


def _binary_closing_torch(x: torch.Tensor, kernel_size: int) -> torch.Tensor:
    """Binary closing = erosion(dilation(x)). x: (H, W) bool."""
    return _binary_erosion_torch(
        _binary_dilate_torch(x, kernel_size), kernel_size)


# ------------------------------------------------------------------
# Depth Neural Network
# ------------------------------------------------------------------


class BGRXYMLPNet(nn.Module):
    """The Neural Network architecture for GelSight calibration."""

    def __init__(self):
        super(BGRXYMLPNet, self).__init__()
        input_size = 5
        self.fc1 = nn.Linear(input_size, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 32)
        self.fc4 = nn.Linear(32, 2)

    def forward(self, x):
        x = F.relu_(self.fc1(x))
        x = F.relu_(self.fc2(x))
        x = F.relu_(self.fc3(x))
        x = self.fc4(x)
        return x


class DepthEstimator:
    """GelSight depth reconstruction class.

    Uses a pre-trained neural network to predict gradients from BGRXY features,
    then integrates them using a torch-based Poisson solver (FFT DCT) to obtain
    depth maps and point clouds.
    """

    def __init__(self, model_path, contact_mode="standard", device="cuda"):
        """Initialize the DepthEstimator.

        Args:
            model_path: str; path to the model file.
            contact_mode: str; the contact mode, can be "standard" or "flat".
            device: str; the device to run the model on, can be "cuda" or "cpu".
        """
        self.model_path = model_path
        self.contact_mode = contact_mode
        self.device = device
        self.bg_image = None
        self.bg_G = None

        # Load the gxy model
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        self.gxy_net = BGRXYMLPNet().to(self.device)
        self.gxy_net.load_state_dict(
            torch.load(model_path, map_location=self.device))
        self.gxy_net.eval()

    def load_bg(self, bg_image):
        """Load the background image.

        Stores background gradient ``self.bg_G`` as a torch tensor
        on the selected device for GPU-accelerated subtraction.

        :param bg_image: np.array (H, W, 3); the background image.
        """
        self.bg_image = bg_image

        # Calculate the gradients of the background
        bgrxys = image2bgrxys(bg_image).reshape(-1, 5)
        features = torch.from_numpy(bgrxys).float().to(self.device)
        with torch.no_grad():
            gxyangles = self.gxy_net(features)
            self.bg_G = torch.tan(
                gxyangles.reshape(
                    bg_image.shape[0], bg_image.shape[1], 2))

    def get_surface_info(
        self, image, ppmm, color_dist_threshold=15, height_threshold=0.2
    ):
        """Get surface info: gradients (G), height map (H), contact mask (C).

        The full pipeline runs on torch tensors (GPU when
        ``self.device == "cuda"``).  Returns numpy arrays
        for downstream compatibility.

        :param image: np.array (H, W, 3); the gelsight image.
        :param ppmm: float; the pixel per mm.
        :param color_dist_threshold: float; color distance
            threshold for contact mask.
        :param height_threshold: float; height threshold for
            contact mask.
        :return: G (H, W, 2), H (H, W), C (H, W) — all numpy.
        """
        dev = self.device

        # ---- Gradients on device ----
        bgrxys = image2bgrxys(image).reshape(-1, 5)
        features = torch.from_numpy(bgrxys).float().to(dev)
        with torch.no_grad():
            gxyangles = self.gxy_net(features)
            G = torch.tan(
                gxyangles.reshape(
                    image.shape[0], image.shape[1], 2))
            if self.bg_G is not None:
                G = G - self.bg_G
            else:
                raise ValueError("Background image is not loaded.")

        # ---- Height map via Poisson solver (torch) ----
        H = poisson_dct_neumaan(G[:, :, 0], G[:, :, 1])

        # ---- Contact mask on device ----
        img_t = torch.from_numpy(image).float().to(dev)
        bg_t = torch.from_numpy(self.bg_image).float().to(dev)
        diff_t = img_t - bg_t
        color_norm = torch.linalg.norm(diff_t, dim=-1)

        if self.contact_mode == "standard":
            color_mask = color_norm > color_dist_threshold
            color_mask = _binary_dilate_torch(color_mask, 7)
            color_mask = _binary_erosion_torch(color_mask, 15)

            # Height-based filter
            cutoff = (torch.quantile(H.flatten().float(), 0.85)
                      - height_threshold * ppmm)
            height_mask = H < cutoff

            C = torch.logical_and(color_mask, height_mask)
        elif self.contact_mode == "flat":
            color_mask = color_norm > 10
            color_mask = _binary_dilate_torch(color_mask, 15)
            C = _binary_erosion_torch(color_mask, 25)
        else:
            C = color_norm > color_dist_threshold

        # ---- Return as numpy (contract with downstream) ----
        G_np = G.cpu().numpy()
        H_np = H.cpu().numpy().astype(np.float32)
        C_np = C.cpu().numpy()
        return G_np, H_np, C_np

    def get_gradient(self, image, ppmm, color_dist_threshold=15,
                     height_threshold=0.2, use_mask=True,
                     refine_mask=True):
        """Get the gradient map from the GelSight image.

        :param image: np.array (H, W, 3); the gelsight image.
        :param ppmm: float; the pixel per mm.
        :param color_dist_threshold: float; color distance
            threshold for contact mask.
        :param height_threshold: float; height threshold for
            contact mask.
        :param use_mask: bool; whether to use the contact mask.
        :return: np.array (H, W, 2); the gradient map (float32).
        """
        G, H, C = self.get_surface_info(
            image, ppmm, color_dist_threshold, height_threshold)
        if refine_mask:
            C = refine_contact_mask(C)

        if use_mask:
            masked_G = np.zeros_like(G)
            masked_G[C != 0] = G[C != 0]
            G = masked_G

        return G

    def get_depth(self, image, ppmm, color_dist_threshold=15,
                  height_threshold=0.2, use_mask=True,
                  refine_mask=True, relative=False,
                  relative_scale=1.0):
        """Get the depth map from the GelSight image.

        :param image: np.array (H, W, 3); the gelsight image.
        :param ppmm: float; the pixel per mm.
        :param color_dist_threshold: float; color distance
            threshold for contact mask.
        :param height_threshold: float; height threshold for
            contact mask.
        :param use_mask: bool; whether to use the contact mask.
        :param relative: bool; normalize depth to [0, 1].
        :param relative_scale: float; scale for relative
            depth normalization.
        :return: np.array (H, W); the depth map in mm (uint8).
        """
        G, H, C = self.get_surface_info(
            image, ppmm, color_dist_threshold, height_threshold)
        if refine_mask:
            C = refine_contact_mask(C)

        # Convert height map to depth in millimeters
        depth = -H / ppmm

        if use_mask:  # Apply contact mask if specified
            masked_depth = np.zeros_like(depth)
            masked_depth[C != 0] = depth[C != 0]
            depth = masked_depth

        if relative:
            # Normalize depth to the range [0, 1]
            depth *= relative_scale
            depth = np.clip(depth, 0, 1)
        depth_map = (depth * 255).astype(np.uint8)

        return depth_map

    def _get_point_sample_stride(self, ppmm, point_sample_mm):
        """Compute stride from point_sample_mm for point cloud
        subsampling.

        Args:
            ppmm: float; pixels per mm.
            point_sample_mm: float; desired point spacing in mm
                (0.0 = no subsampling).

        Returns:
            int; stride value (1 = full resolution).
        """
        if point_sample_mm > 0:
            return max(1, int(point_sample_mm * ppmm))
        return 1

    def _subsample_mask(self, C, stride):
        """Subsample contact mask by stride.

        Args:
            C: np.array (H, W); contact mask.
            stride: int; subsampling stride.

        Returns:
            np.array; subsampled mask.
        """
        if stride > 1:
            return C[::stride, ::stride]
        return C

    def get_point_cloud(self, image, ppmm, color_dist_threshold=15,
                        height_threshold=0.2, use_mask=True,
                        refine_mask=True, return_color=False,
                        mask_only_pointcloud=False,
                        point_sample_mm=0.0):
        """Get the point cloud from the GelSight image.

        :param image: np.array (H, W, 3); the gelsight image.
        :param ppmm: float; the pixel per mm.
        :param color_dist_threshold: float; color distance
            threshold for contact mask.
        :param height_threshold: float; height threshold for
            contact mask.
        :param use_mask: bool; whether to use the contact mask.
        :param return_color: bool; whether to return color info.
        :param mask_only_pointcloud: if True, returns only
            points where mask is True.
        :param point_sample_mm: float; desired point spacing
            in mm. 0.0 = no subsampling (full resolution).
        :return: np.array (N, 3); point cloud in meters.
        """
        G, H, C = self.get_surface_info(
            image, ppmm, color_dist_threshold, height_threshold)
        if refine_mask:
            C = refine_contact_mask(C)

        stride = self._get_point_sample_stride(ppmm, point_sample_mm)
        C = self._subsample_mask(C, stride)

        # Convert height map to point cloud
        pc = height2pointcloud(H, ppmm, stride=stride)

        mask_flat = C.ravel()
        if use_mask:
            if mask_only_pointcloud:
                pc = pc[mask_flat]
            else:
                pc_bg = pc.copy()
                pc_bg[~mask_flat, 2] = 0.0
                pc = pc_bg

        if return_color:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            if stride > 1:
                gray = gray[::stride, ::stride]
            gray = gray.reshape(-1, 1)
            if use_mask:
                gray[~mask_flat] = 0
            return pc, gray
        else:
            return pc

    def estimate(self, image, outputs=None, ppmm=None,
                 color_dist_threshold=15, height_threshold=0.2,
                 use_mask=True, refine_mask=True,
                 relative=False, relative_scale=1.0,
                 mask_only_pointcloud=False, point_sample_mm=0.0):
        """Estimate depth-related outputs from a GelSight image.

        Args:
            image: np.array (H, W, 3); the gelsight image.
            outputs: list of str; outputs to compute. Options:
                'depth', 'gradient', 'pointcloud', 'mask'.
            ppmm: float; the pixel per mm (required).
            color_dist_threshold: float; color distance
                threshold for contact mask.
            height_threshold: float; height threshold for
                contact mask.
            use_mask: bool; whether to use the contact mask.
            refine_mask: bool; whether to refine the mask.
            relative: bool; whether to normalize depth.
            relative_scale: float; scale for relative depth.
            mask_only_pointcloud: bool; if True, use only
                masked area for point cloud.
            point_sample_mm: float; desired point spacing in mm
                for pointcloud subsampling.

        Returns:
            dict: Dictionary with requested outputs.
        """
        if outputs is None:
            outputs = ['depth']
        if ppmm is None:
            raise ValueError("ppmm (pixels per mm) must be provided")

        result = {}

        # Get surface info (shared computation)
        G, H, C = self.get_surface_info(
            image, ppmm, color_dist_threshold, height_threshold)

        if refine_mask:
            C = refine_contact_mask(C)

        # Save full-resolution mask BEFORE any pointcloud subsampling
        if 'mask' in outputs:
            result['mask'] = C.copy()

        # Compute stride for pointcloud subsampling
        stride = self._get_point_sample_stride(ppmm, point_sample_mm)

        # Compute requested outputs
        if 'gradient' in outputs:
            gradient = G.copy()
            if use_mask:
                masked_G = np.zeros_like(gradient)
                masked_G[C != 0] = gradient[C != 0]
                gradient = masked_G
            result['gradient'] = gradient

        if 'depth' in outputs:
            # Convert height map to depth in millimeters
            depth = -H / ppmm

            if use_mask:
                masked_depth = np.zeros_like(depth)
                masked_depth[C != 0] = depth[C != 0]
                depth = masked_depth

            if relative:
                depth *= relative_scale
                depth = np.clip(depth, 0, 1)

            depth_map = (depth * 255).astype(np.uint8)
            result['depth'] = depth_map

        if 'pointcloud' in outputs:
            C_sub = self._subsample_mask(C, stride)
            pc = height2pointcloud(H, ppmm, stride=stride)

            mask_flat = C_sub.ravel()
            if use_mask:
                if mask_only_pointcloud:
                    pc = pc[mask_flat]
                else:
                    pc_bg = pc.copy()
                    pc_bg[~mask_flat, 2] = 0.0
                    pc = pc_bg

            result['pointcloud'] = pc

        return result


def image2bgrxys(image):
    """Convert a bgr image to bgrxy feature.

    :param image: np.array (H, W, 3); the bgr image.
    :return: np.array (H, W, 5); the bgrxy feature.
    """
    ys = np.linspace(0, 1, image.shape[0], endpoint=False,
                     dtype=np.float32)
    xs = np.linspace(0, 1, image.shape[1], endpoint=False,
                     dtype=np.float32)
    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    bgrxys = np.concatenate(
        [image.astype(np.float32) / 255,
         xx[..., np.newaxis], yy[..., np.newaxis]],
        axis=2,
    )
    return bgrxys


def poisson_dct_neumaan(gx, gy):
    """2D integration of depth from gx, gy using Poisson solver.

    Uses torch FFT-based DCT (GPU or CPU).
    Accepts numpy arrays or torch tensors. Returns same type.

    :param gx: np.ndarray or torch.Tensor (H, W); x gradient.
    :param gy: np.ndarray or torch.Tensor (H, W); y gradient.
    :return: same type as input; (H, W) depth map.
    """
    is_numpy = isinstance(gx, np.ndarray)
    if is_numpy:
        gx_t = torch.from_numpy(gx)
        gy_t = torch.from_numpy(gy)
    else:
        gx_t = gx
        gy_t = gy

    H, W = gx_t.shape[-2], gx_t.shape[-1]
    device = gx_t.device

    # Compute Laplacian
    gxx = gx_t[:, torch.cat([
        torch.arange(1, W, device=device),
        torch.tensor([W - 1], device=device)])] \
        - gx_t[:, torch.cat([
            torch.tensor([0], device=device),
            torch.arange(0, W - 1, device=device)])]
    gyy = gy_t[torch.cat([
        torch.arange(1, H, device=device),
        torch.tensor([H - 1], device=device)]), :] \
        - gy_t[torch.cat([
            torch.tensor([0], device=device),
            torch.arange(0, H - 1, device=device)]), :]
    f = gxx + gyy

    # Right hand side of the boundary condition
    b = torch.zeros(H, W, device=device, dtype=gx_t.dtype)
    b[0, 1:-2] = -gy_t[0, 1:-2]
    b[-1, 1:-2] = gy_t[-1, 1:-2]
    b[1:-2, 0] = -gx_t[1:-2, 0]
    b[1:-2, -1] = gx_t[1:-2, -1]
    rsqrt2 = 1.0 / math.sqrt(2)
    b[0, 0] = rsqrt2 * (-gy_t[0, 0] - gx_t[0, 0])
    b[0, -1] = rsqrt2 * (-gy_t[0, -1] + gx_t[0, -1])
    b[-1, -1] = rsqrt2 * (gy_t[-1, -1] + gx_t[-1, -1])
    b[-1, 0] = rsqrt2 * (gy_t[-1, 0] - gx_t[-1, 0])

    # Modification near the boundaries (Eq. 53)
    f[0, 1:-2] -= b[0, 1:-2]
    f[-1, 1:-2] -= b[-1, 1:-2]
    f[1:-2, 0] -= b[1:-2, 0]
    f[1:-2, -1] -= b[1:-2, -1]

    # Modification near the corners (Eq. 54)
    f[0, -1] -= math.sqrt(2) * b[0, -1]
    f[-1, -1] -= math.sqrt(2) * b[-1, -1]
    f[-1, 0] -= math.sqrt(2) * b[-1, 0]
    f[0, 0] -= math.sqrt(2) * b[0, 0]

    # Cosine transform of f
    fcos = _dct2(f)

    # Denominator (Eq. 55)
    x_coord = torch.arange(1, W + 1, device=device, dtype=torch.float32)
    y_coord = torch.arange(1, H + 1, device=device, dtype=torch.float32)
    Xg, Yg = torch.meshgrid(x_coord, y_coord, indexing='xy')
    denom = 4 * (
        (torch.sin(0.5 * math.pi * Xg / W)) ** 2
        + (torch.sin(0.5 * math.pi * Yg / H)) ** 2
    )

    # Inverse cosine transform
    img_tt = _idct2(-fcos / denom)
    img_tt = img_tt.mean() + img_tt

    if is_numpy:
        return img_tt.cpu().numpy()
    return img_tt


def height2pointcloud(H, ppmm, stride=1):
    """Convert height map to point cloud (normalflow style).

    Args:
        H: np.array (H, W); height map.
        ppmm: float; pixels per mm.
        stride: int; subsample stride on each axis
            (1 = full resolution).

    Returns:
        np.array (N, 3); point cloud in meters.
    """
    h, w = H.shape
    if stride > 1:
        yy = (np.arange(0, h, stride) - h / 2 + 0.5)
        xx = (np.arange(0, w, stride) - w / 2 + 0.5)
        X, Y = np.meshgrid(xx, yy)
        H_sub = H[::stride, ::stride]
    else:
        yy = (np.arange(h) - h / 2 + 0.5)
        xx = (np.arange(w) - w / 2 + 0.5)
        X, Y = np.meshgrid(xx, yy)
        H_sub = H
    # Stack and scale to mm, then to meters
    pc = np.stack((X, Y, H_sub), axis=-1) / ppmm / 1000.0
    return pc.reshape(-1, 3)


def refine_contact_mask(C):
    """Erode and close the contact mask to obtain a robust mask.

    Accepts numpy or torch tensor. Returns same type.
    GPU acceleration via torch CUDA when available.

    :param C: np.ndarray (H, W) or torch.Tensor (H, W) bool;
        the contact mask.
    :return: same type as input; the refined contact mask.
    """
    is_numpy = isinstance(C, np.ndarray)
    if is_numpy:
        C_t = torch.from_numpy(C)
        if torch.cuda.is_available():
            C_t = C_t.cuda()
    else:
        C_t = C
    H_img = C_t.shape[0]
    erode_size = max(H_img // 24, 1)
    close_size = max(H_img // 12, 1)
    eroded = _binary_erosion_torch(C_t, erode_size)
    closed = _binary_closing_torch(eroded, close_size)
    if is_numpy:
        return closed.cpu().numpy()
    return closed
