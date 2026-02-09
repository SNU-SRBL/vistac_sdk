import math
import os
import warnings

import cv2
import numpy as np
from scipy import fftpack
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import binary_erosion, binary_closing

class BGRXYMLPNet(nn.Module):
    """
    The Neural Network architecture for GelSight calibration.
    """

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
    """
    The GelSight depth reconstruction class.

    This class handles 3D reconstruction from calibrated GelSight images.
    It uses a pre-trained neural network to predict gradients from BGRXY features,
    then integrates them using Poisson solver to obtain depth maps and point clouds.
    """

    def __init__(self, model_path, contact_mode="standard", device="cuda"):
        """
        Initialize the DepthEstimator.
        
        Args:
            model_path: str; path to the model file.
            contact_mode: str; the contact mode, can be "standard" or "flat".
            device: str; the device to run the model on, can be "cuda" or "cpu".
        """
        self.model_path = model_path
        self.contact_mode = contact_mode
        self.device = device
        self.bg_image = None
        
        # Load the gxy model
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        self.gxy_net = BGRXYMLPNet().to(self.device)
        self.gxy_net.load_state_dict(torch.load(model_path, map_location=self.device))
        self.gxy_net.eval()

    def load_bg(self, bg_image):
        """
        Load the background image.

        :param bg_image: np.array (H, W, 3); the background image.
        """
        self.bg_image = bg_image

        # Calculate the gradients of the background
        bgrxys = image2bgrxys(bg_image).reshape(-1, 5)
        features = torch.from_numpy(bgrxys).float().to(self.device)
        with torch.no_grad():
            gxyangles = self.gxy_net(features)
            gxyangles = gxyangles.cpu().detach().numpy()
            self.bg_G = np.tan(
                gxyangles.reshape(bg_image.shape[0], bg_image.shape[1], 2)
            )

    def get_surface_info(
        self, image, ppmm, color_dist_threshold=15, height_threshold=0.2
    ):
        """
        Get the surface information including gradients (G), height map (H), and contact mask (C).

        :param image: np.array (H, W, 3); the gelsight image.
        :param ppmm: float; the pixel per mm.
        :param color_dist_threshold: float; the color distance threshold for contact mask.
        :param height_threshold: float; the height threshold for contact mask.
        :return G: np.array (H, W, 2); the gradients.
                H: np.array (H, W); the height map.
                C: np.array (H, W); the contact mask.
        """
        # Calculate the gradients
        bgrxys = image2bgrxys(image).reshape(-1, 5)
        features = torch.from_numpy(bgrxys).float().to(self.device)
        with torch.no_grad():
            gxyangles = self.gxy_net(features)
            gxyangles = gxyangles.cpu().detach().numpy()
            G = np.tan(gxyangles.reshape(image.shape[0], image.shape[1], 2))
            if self.bg_image is not None:
                G = G - self.bg_G
            else:
                raise ValueError("Background image is not loaded.")

        # Calculate the height map
        H = poisson_dct_neumaan(G[:, :, 0], G[:, :, 1]).astype(np.float32)

        # Calculate the contact mask
        if self.contact_mode == "standard":
            # Filter by color difference
            diff_image = image.astype(np.float32) - self.bg_image.astype(np.float32)
            color_mask = np.linalg.norm(diff_image, axis=-1) > color_dist_threshold
            color_mask = cv2.dilate(
                color_mask.astype(np.uint8), np.ones((7, 7), np.uint8)
            )
            color_mask = cv2.erode(
                color_mask.astype(np.uint8), np.ones((15, 15), np.uint8)
            )

            # Filter by height
            # cutoff = np.percentile(H, 85) - height_threshold / ppmm
            cutoff = np.percentile(H, 85) - height_threshold * ppmm     #fixed ppmm 
            height_mask = H < cutoff

            # Combine the masks
            C = np.logical_and(color_mask, height_mask)
        elif self.contact_mode == "flat":
            # Find the contact mask based on color difference
            diff_image = image.astype(np.float32) - self.bg_image.astype(np.float32)
            color_mask = np.linalg.norm(diff_image, axis=-1) > 10
            color_mask = cv2.dilate(
                color_mask.astype(np.uint8), np.ones((15, 15), np.uint8)
            )
            C = cv2.erode(
                color_mask.astype(np.uint8), np.ones((25, 25), np.uint8)
            ).astype(np.bool_)
        return G, H, C
    
    def get_gradient(self, image, ppmm, color_dist_threshold=15,
                    height_threshold=0.2, use_mask=True, refine_mask=True):
        """
        Get the gradient map from the GelSight image.
        :param image: np.array (H, W, 3); the gelsight image.
        :param ppmm: float; the pixel per mm.
        :param color_dist_threshold: float; the color distance threshold for contact mask.
        :param height_threshold: float; the height threshold for contact mask.
        :param use_mask: bool; whether to use the contact mask.
        :return: np.array (H, W, 2); the gradient map (float32).
        """
        G, H, C = self.get_surface_info(
            image, ppmm, color_dist_threshold, height_threshold
        )
        if refine_mask:
            C = refine_contact_mask(C)

        if use_mask:
            masked_G = np.zeros_like(G)
            masked_G[C != 0] = G[C != 0]
            G = masked_G

        return G
    
    def get_depth(self, image, ppmm, color_dist_threshold=15,
                height_threshold=0.2, use_mask=True, refine_mask=True, relative=False, relative_scale=1.0):
        """
        Get the depth map from the GelSight image.
        :param image: np.array (H, W, 3); the gelsight image.
        :param ppmm: float; the pixel per mm.
        :param color_dist_threshold: float; the color distance threshold for contact mask.
        :param height_threshold: float; the height threshold for contact mask.
        :param use_mask: bool; whether to use the contact mask.
        :param length_scale_in_mm: float; normalization scale for depth.
        :return: np.array (H, W); the depth map in mm (uint8).
        """
        G, H, C = self.get_surface_info(
            image, ppmm, color_dist_threshold, height_threshold
        )
        if refine_mask:
            # Erode the contact mask to remove noise
            C =refine_contact_mask(C)
        
        depth = -H / ppmm # Convert height map to depth in millimeters

        if use_mask:  # Apply contact mask if specified
            masked_depth = np.zeros_like(depth)
            masked_depth[C != 0] = depth[C != 0]
            depth = masked_depth

        # print("Depth min:", depth.min(), "max:", depth.max(), "mean:", depth.mean())

        if relative:
            # Normalize depth to the range [0, 1] based on the relative scale
            depth *= relative_scale
            depth = np.clip(depth, 0, 1)
        depth_map = (depth * 255).astype(np.uint8)

        return depth_map
    
    def get_point_cloud(self, image, ppmm, color_dist_threshold=15,
                        height_threshold=0.2, use_mask=True, refine_mask=True, return_color=False,
                        mask_only_pointcloud=False):
        """
        Get the point cloud from the GelSight image.
        :param image: np.array (H, W, 3); the gelsight image.
        :param ppmm: float; the pixel per mm.
        :param color_dist_threshold: float; the color distance threshold for contact mask.
        :param height_threshold: float; the height threshold for contact mask.
        :param use_mask: bool; whether to use the contact mask.
        :param return_color: bool; whether to return the color information.
        :param mask_only_pointcloud: if True, returns only points where mask is True.
        :return: np.array (N, 3); the point cloud (scaled in meters).
        """
        G, H, C = self.get_surface_info(
            image, ppmm, color_dist_threshold, height_threshold
        )
        if refine_mask:
            C = refine_contact_mask(C)

        pc = height2pointcloud(H, ppmm) # Convert height map to point cloud

        mask_flat = C.ravel()
        if use_mask:
            if mask_only_pointcloud:
                pc = pc[mask_flat]
            else:
                pc_bg = pc.copy()
                pc_bg[~mask_flat, 2] = 0.0
                pc = pc_bg
        
        if return_color:
            # Convert image to grayscale and flatten
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).reshape(-1, 1)
            if use_mask:
                gray[~mask_flat] = 0
            return pc, gray
        else:
            return pc

    def estimate(self, image, outputs=['depth'], ppmm=None, 
                 color_dist_threshold=15, height_threshold=0.2, 
                 use_mask=True, refine_mask=True, 
                 relative=False, relative_scale=1.0, 
                 mask_only_pointcloud=False):
        """
        Estimate depth-related outputs from a GelSight image.
        
        Args:
            image: np.array (H, W, 3); the gelsight image.
            outputs: list of str; outputs to compute. Options: 'depth', 'gradient', 'pointcloud', 'mask'
            ppmm: float; the pixel per mm (required).
            color_dist_threshold: float; the color distance threshold for contact mask.
            height_threshold: float; the height threshold for contact mask.
            use_mask: bool; whether to use the contact mask.
            refine_mask: bool; whether to refine the contact mask.
            relative: bool; whether to normalize depth to the range [0, 1].
            relative_scale: float; the scale for relative depth normalization.
            mask_only_pointcloud: bool; if True, use only masked area for point cloud.
            
        Returns:
            dict: Dictionary with requested outputs:
                - 'depth': np.array (H, W) uint8, depth map in mm scaled to [0, 255]
                - 'gradient': np.array (H, W, 2) float32, gradient angles
                - 'pointcloud': np.array (N, 3) float32, XYZ in meters
                - 'mask': np.array (H, W) bool, contact mask
        """
        if ppmm is None:
            raise ValueError("ppmm (pixels per mm) must be provided")
        
        result = {}
        
        # Get surface info (shared computation)
        G, H, C = self.get_surface_info(image, ppmm, color_dist_threshold, height_threshold)
        
        if refine_mask:
            C = refine_contact_mask(C)
        
        # Compute requested outputs
        if 'gradient' in outputs:
            gradient = G.copy()
            if use_mask:
                masked_G = np.zeros_like(gradient)
                masked_G[C != 0] = gradient[C != 0]
                gradient = masked_G
            result['gradient'] = gradient
        
        if 'depth' in outputs:
            depth = -H / ppmm  # Convert height map to depth in millimeters
            
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
            pc = height2pointcloud(H, ppmm)
            
            mask_flat = C.ravel()
            if use_mask:
                if mask_only_pointcloud:
                    pc = pc[mask_flat]
                else:
                    pc_bg = pc.copy()
                    pc_bg[~mask_flat, 2] = 0.0
                    pc = pc_bg
            
            result['pointcloud'] = pc
        
        if 'mask' in outputs:
            result['mask'] = C
        
        return result


def image2bgrxys(image):
    """
    Convert a bgr image to bgrxy feature.

    :param image: np.array (H, W, 3); the bgr image.
    :return: np.array (H, W, 5); the bgrxy feature.
    """
    ys = np.linspace(0, 1, image.shape[0], endpoint=False, dtype=np.float32)
    xs = np.linspace(0, 1, image.shape[1], endpoint=False, dtype=np.float32)
    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    bgrxys = np.concatenate(
        [image.astype(np.float32) / 255, xx[..., np.newaxis], yy[..., np.newaxis]],
        axis=2,
    )
    return bgrxys


def poisson_dct_neumaan(gx, gy):
    """
    2D integration of depth from gx, gy using Poisson solver.

    :param gx: np.array (H, W); the x gradient.
    :param gy: np.array (H, W); the y gradient.
    :return: np.array (H, W); the depth map.
    """
    # Compute Laplacian
    gxx = 1 * (
        gx[:, (list(range(1, gx.shape[1])) + [gx.shape[1] - 1])]
        - gx[:, ([0] + list(range(gx.shape[1] - 1)))]
    )
    gyy = 1 * (
        gy[(list(range(1, gx.shape[0])) + [gx.shape[0] - 1]), :]
        - gy[([0] + list(range(gx.shape[0] - 1))), :]
    )
    f = gxx + gyy

    # Right hand side of the boundary condition
    b = np.zeros(gx.shape)
    b[0, 1:-2] = -gy[0, 1:-2]
    b[-1, 1:-2] = gy[-1, 1:-2]
    b[1:-2, 0] = -gx[1:-2, 0]
    b[1:-2, -1] = gx[1:-2, -1]
    b[0, 0] = (1 / np.sqrt(2)) * (-gy[0, 0] - gx[0, 0])
    b[0, -1] = (1 / np.sqrt(2)) * (-gy[0, -1] + gx[0, -1])
    b[-1, -1] = (1 / np.sqrt(2)) * (gy[-1, -1] + gx[-1, -1])
    b[-1, 0] = (1 / np.sqrt(2)) * (gy[-1, 0] - gx[-1, 0])

    # Modification near the boundaries to enforce the non-homogeneous Neumann BC (Eq. 53 in [1])
    f[0, 1:-2] = f[0, 1:-2] - b[0, 1:-2]
    f[-1, 1:-2] = f[-1, 1:-2] - b[-1, 1:-2]
    f[1:-2, 0] = f[1:-2, 0] - b[1:-2, 0]
    f[1:-2, -1] = f[1:-2, -1] - b[1:-2, -1]

    # Modification near the corners (Eq. 54 in [1])
    f[0, -1] = f[0, -1] - np.sqrt(2) * b[0, -1]
    f[-1, -1] = f[-1, -1] - np.sqrt(2) * b[-1, -1]
    f[-1, 0] = f[-1, 0] - np.sqrt(2) * b[-1, 0]
    f[0, 0] = f[0, 0] - np.sqrt(2) * b[0, 0]

    # Cosine transform of f
    tt = fftpack.dct(f, norm="ortho")
    fcos = fftpack.dct(tt.T, norm="ortho").T

    # Cosine transform of z (Eq. 55 in [1])
    (x, y) = np.meshgrid(range(1, f.shape[1] + 1), range(1, f.shape[0] + 1), copy=True)
    denom = 4 * (
        (np.sin(0.5 * math.pi * x / (f.shape[1]))) ** 2
        + (np.sin(0.5 * math.pi * y / (f.shape[0]))) ** 2
    ).astype(np.float32)

    # Inverse Discrete cosine Transform
    f = -fcos / denom
    tt = fftpack.idct(f, norm="ortho")
    img_tt = fftpack.idct(tt.T, norm="ortho").T
    img_tt = img_tt.mean() + img_tt

    return img_tt

def height2pointcloud(H, ppmm):
    """Convert height map to point cloud (normalflow style)."""
    h, w = H.shape
    xx, yy = np.meshgrid(np.arange(w), np.arange(h))
    xx = xx - w / 2 + 0.5
    yy = yy - h / 2 + 0.5
    # Stack and scale to mm, then to meters
    pc = np.stack((xx, yy, H), axis=-1) / ppmm / 1000.0
    return pc.reshape(-1, 3)

def refine_contact_mask(C):
    """
    Erode and close the contact mask to obtain a robust contact mask.

    :param C: np.ndarray (H, W); the contact mask.
    :return refined_C: np.ndarray (H, W); the refined contact mask.
    """
    erode_size = max(C.shape[0] // 24, 1)
    close_size = max(C.shape[0] // 12, 1)
    eroded_C = binary_erosion(C, structure=np.ones((erode_size, erode_size)))
    closed_C = binary_closing(eroded_C, structure=np.ones((close_size, close_size)))

    return closed_C