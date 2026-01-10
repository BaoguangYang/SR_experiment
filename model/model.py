import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = "1"
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2 as cv
import numpy as np
import json
from .space_to_depth import DepthToSpace, SpaceToDepth


def retrieve_elements_from_indices(tensor: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    _, iC, _, _ = indices.shape
    B, C, H, W = tensor.shape
    indices = indices.flatten(start_dim=1).view(B, 1, -1)
    tensor = tensor.flatten(start_dim=2)

    indices = indices.repeat(1, C, 1) 

    tensor = tensor.gather(dim=2, index=indices).view(B, C, H, W)
    return tensor 


def warp(image: torch.Tensor, motion: torch.Tensor) -> torch.Tensor:
    """
    image: (B, C, H, W) - High resolution image
    motion: (B, 2, h, w) - Low resolution motion vectors
    """

    B, C, H, W = image.shape
    _, _, h, w = motion.shape
    device = image.device
    # Step 1: Bilinear upsample motion vector to match image resolution
    # motion_up = F.interpolate(motion, size=(H, W), mode='bilinear', align_corners=True)
    motion_up = F.interpolate(motion, size=(H, W), mode='nearest')

    # Step 2: Build pixel coordinate grid
    yy, xx = torch.meshgrid(
        torch.arange(H, device=device),
        torch.arange(W, device=device),
        indexing='ij'
    )  # (H, W)

    xx = xx.unsqueeze(0).expand(B, -1, -1).float()  # (B, H, W)
    yy = yy.unsqueeze(0).expand(B, -1, -1).float()  # (B, H, W)

    # Step 3: Apply motion to pixel positions
    # Note: HLSL uses (-mv.x, mv.y) convention
    new_x = xx + (-motion_up[:, 0] * W)
    new_y = yy + ( motion_up[:, 1] * H)

    # Step 4: Normalize coordinates to [-1, 1] for grid_sample
    new_x = (2.0 * new_x / (W - 1)) - 1.0
    new_y = (2.0 * new_y / (H - 1)) - 1.0

    grid = torch.stack((new_x, new_y), dim=-1)  # (B, H, W, 2)

    # Step 5: Bilinear sample image at warped positions
    warped = F.grid_sample(
        image,
        grid,
        mode='bilinear',
        padding_mode='border',  # border padding like SampleLevel usually does
        align_corners=True
    )
    return warped 



class Warping(nn.Module):
    def __init__(self, scale_factor: int = 2, jitter: bool = False, depth_dilation: bool = False, depth_block_size: int = 7) -> None:
        assert depth_block_size % 2 == 1
        super().__init__()
        self.space_to_depth = SpaceToDepth(scale_factor=scale_factor)
        self.max_pool = nn.MaxPool2d(kernel_size=depth_block_size, stride=1, padding=depth_block_size // 2, return_indices=True)
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=True)

        self.jitter_on = jitter
        self.depth_dilation = depth_dilation

    def forward(self, 
                depth: torch.Tensor, 
                jitter: torch.Tensor, 
                prev_jitter: torch.Tensor,
                motion: torch.Tensor, 
                prev_features: torch.Tensor, 
                prev_color: torch.Tensor,
                device,
                high_res = False) -> torch.Tensor:
        """
        motion: B, 2, h, w
        depth: B, 1, h, w
        jitter: B, 2, 1, 1
        prev_jitter: B, 2, 1, 1
            assumed that jitter is in relative coordinates 
            (i.e. -1 to 1, -1 is the whole image left, 1 is the whole image right)
        prev_features: B, 1, H, W
        prev_color: B, 3, H, W
        """

        _, _, h, w = motion.shape

        if self.jitter_on:
            motion[:, 0] = motion[:, 0] + (-prev_jitter[:, 0] + jitter[:, 0]) / w
            motion[:, 1] = motion[:, 1] + (-prev_jitter[:, 1] + jitter[:, 1]) / h
        
        if self.depth_dilation:
            _, indices = self.max_pool(depth)
            motion = retrieve_elements_from_indices(motion, indices)
        
        # Warp previous features and color
        prev_features = warp(prev_features, motion)
        prev_color = warp(prev_color, motion) # [batch, 3, H, W]

        if high_res:
            return prev_features, prev_color # B, 3, H, W

        prev_features = self.space_to_depth(prev_features)
        prev_color = self.space_to_depth(prev_color) # B, 12, h, w

        return prev_features, prev_color


class Reconstruction(nn.Module):
    """
    reconstruction network for neural network module
    """
    def __init__(self, f: int, m: int, jitter: bool, scale_factor: int = 2):
        super().__init__()

        if jitter:
            self.encoder = nn.Sequential(nn.Conv2d(22, f, 3, 1, 1), nn.ReLU())
            self.decoder = nn.Conv2d(f, 17, 3, 1, 1)
        else:
            self.encoder = nn.Sequential(nn.Conv2d(20, f, 3, 1, 1), nn.ReLU())
            self.decoder = nn.Conv2d(f, 17, 3, 1, 1)
        
        self.net = nn.Sequential(
            *[nn.Sequential(nn.Conv2d(f, f, 3, 1, 1), nn.ReLU()) for _ in range(m)],
        )

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.depth_to_space = DepthToSpace(scale_factor=scale_factor)

        self.jitter_on = jitter


    def forward(self, 
                color: torch.Tensor, # [B, 3, h, w]
                depth: torch.Tensor, # [B, 1, h, w]
                jitter: torch.Tensor, # [B, 2, 1, 1]
                prev_features: torch.Tensor, # [B, 4, h, w]
                prev_color: torch.Tensor, # [B, 12, h, w]
                ):
        B, _, h, w = color.shape
        jitter_ = torch.zeros_like(jitter)

        if self.jitter_on:
            jitter_[:, 0, :, :] = jitter[:, 0, :, :] / w
            jitter_[:, 1, :, :] = jitter[:, 1, :, :] / h
            jitter_ = jitter_.repeat(1, 1, h, w) # (B, 2, h, w)

            x = torch.cat([color, depth, prev_features, prev_color, jitter_], dim=1)
        else:
            x = torch.cat([color, depth, prev_features, prev_color], dim=1)

        x = self.encoder(x)
        x = self.net(x)
        x = self.decoder(x)

        features = x[:, :4]
        blending_mask = self.sigmoid(x[:, 4:5])
        current_frame = self.relu(x[:, 5:])

        features = self.depth_to_space(features)
        color = self.depth_to_space(blending_mask * current_frame + (1 - blending_mask) * prev_color)
        
        return features, color, blending_mask


class Model(nn.Module):
    def __init__(self, f: int, m: int, jitter: bool, depth_dilation: bool, scale_factor: int=2, depth_block_size: int=7):
    super().__init__()

    self.warping = Warping(scale_factor, jitter, depth_dilation, depth_block_size)
    self.network = Reconstruction(f, m, jitter)

    def forward(self, 
                color: torch.Tensor, # [B, 3, h, w]
                depth: torch.Tensor, # [B, 1, h, w]
                jitter: torch.Tensor, # [B, 2, 1, 1]
                prev_jitter: torch.Tensor, # [B, 2, 1, 1]
                motion: torch.Tensor, # [B, 2, h, w]
                prev_features: torch.Tensor, # [B, 1, H, W]
                prev_color: torch.Tensor, # [B, 3, H, W]
                device
    ):

        warped_features, warped_color = self.warping(depth, jitter, prev_jitter, motion, prev_features, prev_color, device)
        features, color, blending_mask = self.network(color, depth, jitter, warped_features, warped_color)
        return color, features, blending_mask