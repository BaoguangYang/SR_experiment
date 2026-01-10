import torch.nn as nn

class SpaceToDepth(nn.Module):
    #[R1, G1, B1, R2, G2, B2, R3, G3, B3, R4, G4, B4]
    def __init__(self, scale_factor: int):
        super().__init__()
        self.bs = scale_factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N, C, H, W = x.size()
        x = x.view(N, C, H // self.bs, self.bs, W // self.bs, self.bs)  # (N, C, H//bs, bs, W//bs, bs)
        x = x.permute(0, 3, 5, 1, 2, 4).contiguous()  # (N, bs, bs, C, H//bs, W//bs)
        x = x.view(N, C * (self.bs ** 2), H // self.bs, W // self.bs)  # (N, C*bs^2, H//bs, W//bs)
        return x 


class DepthToSpace(nn.Module):
    def __init__(self, scale_factor: int):
        super().__init__()
        self.bs = scale_factor


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N, C, H, W = x.size()
        x = x.view(N, self.bs, self.bs, C // (self.bs ** 2), H, W)  # (N, bs, bs, C//bs^2, H, W)
        x = x.permute(0, 3, 4, 1, 5, 2).contiguous()  # (N, C//bs^2, H, bs, W, bs)
        x = x.view(N, C // (self.bs ** 2), H * self.bs, W * self.bs)  # (N, C//bs^2, H * bs, W * bs)
        return x 