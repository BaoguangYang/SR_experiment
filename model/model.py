import torch
import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel
from model.modules import DepthToSpace, SpaceToDepth
from utils import warp, retrieve_elements_from_indices


def conv_block(in_ch, out_ch, k=3, s=1, p=1):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, kernel_size=k, stride=1, padding=p),
        nn.ReLU(inplace=True),
    )


class ENSS(BaseModel):
    """
    Approximate implementation of the ENSS network from:
      'Efficient Neural Supersampling on a Novel Gaming Dataset' (Mercier et al., ICCV 2023)

    Design choices:
      - Work in a space-to-depth (subpixel) domain via SpaceToDepth/DepthToSpace.
      - Keep a recurrent hidden state in subpixel (low-res) space.
      - Use utils.warp(hidden, motion) for temporal reprojection:
          * motion is assumed to be in NORMALIZED grid coordinates
            (same convention as F.grid_sample: [-1, 1] range).
          * This matches utils.warp, which constructs a base grid and adds motion.
      - Reconstruct high-res RGB via DepthToSpace.
    """

    def __init__(
        self,
        in_channels_rgb=3,
        in_channels_depth=1,
        in_channels_motion=2,
        upscale_factor=2,
        base_channels=32,
        use_depth=True,
        use_motion=True,
        use_prev_state=True,
    ):
        super().__init__()

        self.upscale_factor = upscale_factor
        self.use_depth = use_depth
        self.use_motion = use_motion
        self.use_prev_state = use_prev_state

        # Space<->Depth modules (wrapper around depth_to_space / space_to_depth)
        self.s2d = SpaceToDepth(block_size=upscale_factor)
        self.d2s = DepthToSpace(block_size=upscale_factor)

        # ------------------------------------------------------------------
        # Channels in space-to-depth domain
        # ------------------------------------------------------------------
        sf = upscale_factor

        # RGB in S2D domain
        in_ch = in_channels_rgb * (sf ** 2)

        # Depth in S2D domain
        if self.use_depth:
            self.depth_channels_s2d = in_channels_depth * (sf ** 2)
            in_ch += self.depth_channels_s2d
        else:
            self.depth_channels_s2d = 0

        # Motion in S2D domain (if you choose to concatenate it)
        if self.use_motion:
            self.motion_channels_s2d = in_channels_motion * (sf ** 2)
            in_ch += self.motion_channels_s2d
        else:
            self.motion_channels_s2d = 0

        # Recurrent hidden state (in S2D resolution)
        self.hidden_channels = base_channels
        if self.use_prev_state:
            in_ch += self.hidden_channels

        # ------------------------------------------------------------------
        # Encoder
        # ------------------------------------------------------------------
        self.enc1 = conv_block(in_ch, base_channels)             # H/sf,     W/sf
        self.enc2 = conv_block(base_channels, base_channels * 2) # H/(2sf),  W/(2sf)
        self.enc3 = conv_block(base_channels * 2, base_channels * 4)

        # ------------------------------------------------------------------
        # Bottleneck
        # ------------------------------------------------------------------
        self.bottleneck = conv_block(base_channels * 4, base_channels * 4)

        # ------------------------------------------------------------------
        # Decoder (UNet-like)
        # ------------------------------------------------------------------
        self.dec3 = conv_block(base_channels * 4 + base_channels * 4,
                               base_channels * 2)
        self.dec2 = conv_block(base_channels * 2 + base_channels * 2,
                               base_channels)
        self.dec1 = conv_block(base_channels + base_channels,
                               base_channels)

        # ------------------------------------------------------------------
        # Output projection in space-to-depth domain
        # ------------------------------------------------------------------
        # Output is (B, 3*sf^2, H/sf, W/sf) -> DepthToSpace -> (B, 3, H*sf, W*sf)
        self.to_out = nn.Conv2d(
            base_channels,
            in_channels_rgb * (sf ** 2),
            kernel_size=3,
            padding=1,
        )

        # Bottleneck -> recurrent hidden state (S2D resolution)
        self.to_hidden = nn.Conv2d(
            base_channels * 4,
            self.hidden_channels,
            kernel_size=1,
        )

        # Internal recurrent state (space-to-depth resolution)
        self._hidden_state = None

    # ----------------------------------------------------------------------
    # State management
    # ----------------------------------------------------------------------
    def reset_state(self):
        """Reset recurrent hidden state; call at the start of a new sequence."""
        self._hidden_state = None

    # ----------------------------------------------------------------------
    # Temporal warping in space-to-depth domain
    # ----------------------------------------------------------------------
    def _warp_hidden(self, motion: torch.Tensor) -> torch.Tensor | None:
        """
        Warp previous hidden state using utils.warp(hidden, motion).

        Assumptions:
          - self._hidden_state has shape (B, C_h, H_s2d, W_s2d)
          - motion is (B, 2, H, W) in NORMALIZED grid coordinates
            (the same convention as F.grid_sample: [-1, 1] range)
          - utils.warp constructs a base [-1, 1] grid and adds 'motion',
            so motion is *displacement* in normalized space.

        Implementation details:
          - We resize motion from (H, W) -> (H_s2d, W_s2d) via bilinear
            interpolation, which is standard for optical-flow-like fields.
          - We do NOT clamp motion; out-of-range coords follow utils.warp
            behaviour (grid_sample padding_mode='zeros').
        """
        if self._hidden_state is None:
            return None

        B, C_h, H_s2d, W_s2d = self._hidden_state.shape

        # Resize motion to match hidden resolution (S2D)
        motion_s2d = F.interpolate(
            motion,
            size=(H_s2d, W_s2d),
            mode="bilinear",
            align_corners=True,  # IMPORTANT: match utils.warp / grid_sample
        )

        # warp(hidden, motion_s2d) -> (B, C_h, H_s2d, W_s2d)
        warped = warp(self._hidden_state, motion_s2d)
        return warped

    # ----------------------------------------------------------------------
    # Forward
    # ----------------------------------------------------------------------
    def forward(self, x, depth=None, motion=None, jitter=None):
        """
        x      : (B, 3, H, W)       low-res RGB
        depth  : (B, 1, H, W)       optional depth
        motion : (B, 2, H, W)       optional motion vectors in NORMALIZED coords
        jitter : (B, 2) or (B, 2, 1, 1), currently unused here directly but
                 may have been applied when creating 'motion' in the dataloader.

        Returns:
          y_hr : (B, 3, H*sf, W*sf) high-res RGB
        """

        B, C, H, W = x.shape
        sf = self.upscale_factor

        # --------------------------------------------------------------
        # Move inputs to space-to-depth domain
        # --------------------------------------------------------------
        # (B, 3, H, W) -> (B, 3*sf^2, H/sf, W/sf)
        x_s2d = self.s2d(x)

        depth_s2d = None
        motion_s2d_for_concat = None

        if self.use_depth and depth is not None:
            depth_s2d = self.s2d(depth)

        if self.use_motion and motion is not None:
            # For concatenation, we also put motion into S2D domain.
            # Note: this is separate from the version resized in _warp_hidden,
            # which uses bilinear resize to match hidden resolution precisely.
            motion_s2d_for_concat = self.s2d(motion)

        # --------------------------------------------------------------
        # Build encoder input
        # --------------------------------------------------------------
        inputs = [x_s2d]

        if depth_s2d is not None:
            inputs.append(depth_s2d)
        if motion_s2d_for_concat is not None:
            inputs.append(motion_s2d_for_concat)

        # Warp recurrent hidden state (S2D resolution) using normalized motion
        if self.use_prev_state and motion is not None:
            warped_hidden = self._warp_hidden(motion)
        else:
            warped_hidden = None

        if self.use_prev_state:
            if warped_hidden is None:
                warped_hidden = torch.zeros(
                    (B, self.hidden_channels, x_s2d.shape[2], x_s2d.shape[3]),
                    device=x.device,
                    dtype=x.dtype,
                )
            inputs.append(warped_hidden)

        x_in = torch.cat(inputs, dim=1)

        # --------------------------------------------------------------
        # Encoder
        # --------------------------------------------------------------
        e1 = self.enc1(x_in)                           # H/sf,     W/sf
        e2 = self.enc2(F.avg_pool2d(e1, 2))            # H/(2sf),  W/(2sf)
        e3 = self.enc3(F.avg_pool2d(e2, 2))            # H/(4sf),  W/(4sf)

        # --------------------------------------------------------------
        # Bottleneck
        # --------------------------------------------------------------
        b = self.bottleneck(e3)

        # Update recurrent hidden state (S2D resolution)
        if self.use_prev_state:
            # Upsample b to encoder stage-1 resolution (H/sf, W/sf)
            h = F.interpolate(
                b,
                size=e1.shape[-2:],
                mode="bilinear",
                align_corners=True,  # keep consistent with warp/grid_sample usage
            )
            self._hidden_state = self.to_hidden(h)

        # --------------------------------------------------------------
        # Decoder (UNet-style)
        # --------------------------------------------------------------
        d3 = F.interpolate(b, scale_factor=2, mode="bilinear", align_corners=True)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))

        d2 = F.interpolate(d3, scale_factor=2, mode="bilinear", align_corners=True)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = F.interpolate(d2, scale_factor=2, mode="bilinear", align_corners=True)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        # --------------------------------------------------------------
        # Project back to RGB via DepthToSpace
        # --------------------------------------------------------------
        out_s2d = self.to_out(d1)          # (B, 3*sf^2, H/sf, W/sf)
        y_hr = self.d2s(out_s2d)           # (B, 3, H*sf, W*sf)

        return y_hr


class Warping(BaseModel):
    def __init__(self, scale_factor: int, depth_block_size: int = 3) -> None:
        assert depth_block_size % 2 == 1 # They used 8x8 but I can't use even kernels yet
        super().__init__()
        self.space_to_depth = SpaceToDepth(block_size=scale_factor)
        self.max_pool = nn.MaxPool2d(kernel_size=depth_block_size, stride=1, padding=depth_block_size // 2, return_indices=True)

    def forward(self, 
                depth: torch.Tensor, 
                jitter: torch.Tensor, 
                prev_jitter: torch.Tensor,
                motion: torch.Tensor, 
                prev_features: torch.Tensor, 
                prev_color: torch.Tensor) -> torch.Tensor:
        """
        motion: B, 2, H, W
        depth: B, 1, H, W
        jitter: B, 2, 1, 1
        prev_jitter: B, 2, 1, 1
            assumed that jitter is in relative coordinates 
            (i.e. -1 to 1, -1 is the whole image left, 1 is the whole image right)
        prev_features: B, 1, H, W
        prev_color: B, 3, H, W

        Note:
            H, W are target resolutions
            it is assumed that depth and motion are upsampled prior
        """

        # Jitter compensation for motion
        motion[:, 0] = motion[:, 0] + prev_jitter[:, 0] - jitter[:, 0] # x
        motion[:, 1] = motion[:, 1] + prev_jitter[:, 1] - jitter[:, 1] # y

        # Depth informed dilation
        # Get indices of closest pixels and use those motion vectors
        _, indices = self.max_pool(depth)
        motion = retrieve_elements_from_indices(motion, indices)

        # Warp previous features and color
        prev_features = warp(prev_features, motion)
        prev_color = warp(prev_color, motion)

        # Transform to input resolution
        prev_features = self.space_to_depth(prev_features)
        prev_color = self.space_to_depth(prev_color)

        return prev_features, prev_color