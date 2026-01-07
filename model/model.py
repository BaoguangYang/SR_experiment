import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = "1"
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2 as cv
import numpy as np
import json
from .base_model import BaseModel
from .space_to_depth import DepthToSpace, SpaceToDepth
from .kernel_prediction import KernelPrediction
from .utils import warp, retrieve_elements_from_indices, get_unique_filename


class Warping(BaseModel):
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

class Reconstruction(BaseModel):
    """
    reconstruction network for neural network module
    """
    def __init__(self, f: int, m: int, jitter: bool, jitter_conv: bool, jitter_conv_channels: int, scale_factor: int = 2):
        super().__init__()

        if jitter_conv:
            self.enc_kernel_predictor = KernelPrediction(layers=7, hidden_features=2048, kernel_size=3, num_kernel=1)
            self.dec_kernel_predictor = KernelPrediction(layers=7, hidden_features=2048, kernel_size=3, num_kernel=3)

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
        self.jitter_conv_on = jitter_conv
        self.jitter_conv_channels = 4 # only color LR and depth LR will do jitter conv


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


        if self.jitter_conv_on:
            enc_kernel = self.enc_kernel_predictor(jitter) # [B, 1, 3, 3]
            dec_kernel = self.dec_kernel_predictor(jitter) # [B, 3, 3, 3]
            enc_kernel = enc_kernel.repeat(1, self.jitter_conv_channels, 1, 1).view(B*self.jitter_conv_channels, 1, 3, 3)

            x_input = x[:, :self.jitter_conv_channels].reshape(1, B*self.jitter_conv_channels, h, w)
            x_output = F.conv2d(x_input, enc_kernel, padding=1, groups=B*self.jitter_conv_channels)
            x = torch.cat([x_output.reshape(B, self.jitter_conv_channels, h, w), x[:, self.jitter_conv_channels:]], dim=1)

        x = self.encoder(x)
        x = self.net(x)
        x = self.decoder(x)

        if self.jitter_conv_on:
            features = F.conv2d(
                x[:, :4].reshape(1, B*4, h, w), 
                dec_kernel[:, 0:1].repeat(1, 4, 1, 1).reshape(B*4, 1, 3, 3), 
                padding=1, groups=B*4,
            ).reshape(B, 4, h, w)
            blending_mask = self.sigmoid(F.conv2d(
                x[:, 4:5].reshape(1, B*1, h, w),
                dec_kernel[:, 1:2], 
                padding=1, groups=B*1,
            )).reshape(B, 1, h, w)
            current_frame = self.relu(F.conv2d(
                x[:, 5:].reshape(1, B*12, h, w), 
                dec_kernel[:, 2:3].repeat(1, 12, 1, 1).reshape(B*12, 1, 3, 3), 
                padding=1, groups=B*12,
            )).reshape(B, 12, h, w)
        else:
            features = x[:, :4]
            blending_mask = self.sigmoid(x[:, 4:5])
            current_frame = self.relu(x[:, 5:])

        features = self.depth_to_space(features)
        color = self.depth_to_space(blending_mask * current_frame + (1 - blending_mask) * prev_color)
        
        return features, color, blending_mask


class Model(BaseModel):
    def __init__(self, f: int, m: int, jitter: bool, jitter_conv: bool, jitter_conv_channels: int, depth_dilation: bool, scale_factor: int=2, depth_block_size: int=7):
    super().__init__()

    self.warping = Warping(scale_factor, jitter, depth_dilation, depth_block_size)
    self.network = Reconstruction(f, m, jitter, jitter_conv, jitter_conv_channels)

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