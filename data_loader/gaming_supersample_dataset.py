import os
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import json # Need for loading jitter data from JSON
import cv2 # Import OpenCV for EXR loading

# -------------------------------------------------------------------------
# Helper functions
# -------------------------------------------------------------------------

def load_image(path):
    """Load RGB or depth images (L or RGB). Output: float32 tensor [0,1]."""
    img = Image.open(path)
    arr = np.array(img).astype(np.float32)

    # Normalize 0–255 to 0–1 if needed
    if arr.dtype != np.float32 or arr.max() > 1.0:
        arr = arr / 255.0

    if arr.ndim == 2:  # grayscale
        arr = arr[..., None]  # H, W, 1

    # Convert to tensor C,H,W
    arr = np.transpose(arr, (2, 0, 1))
    return torch.from_numpy(arr)


def load_flow(path):
    """
    Load motion vectors (flow) as float32 HxWx2 array.
    Supports EXR (via OpenCV) or numpy binary formats.
    """
    ext = Path(path).suffix.lower()
    print(f"[load_flow] Attempting to load: {path} with extension {ext}") # Diagnostic print

    if ext == ".npy":
        flow = np.load(path).astype(np.float32)  # H, W, 2

    elif ext in [".png", ".jpg"]:
        # Optional: some datasets store encoded flow in RGB channels
        raise NotImplementedError(
            f"Flow from {ext} not implemented. Convert motion to .npy or .exr"
        )

    elif ext == ".exr":
        print("[load_flow] Attempting to load EXR with OpenCV.") # Diagnostic print
        try:
            # Use OpenCV to load EXR files
            # IMREAD_UNCHANGED ensures all channels are loaded as is, typically float32
            flow = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if flow is None:
                raise RuntimeError(f"OpenCV failed to load EXR file: {path}")
            # OpenCV loads as H, W, C. We need C, H, W for PyTorch
            flow = np.transpose(flow, (2, 0, 1)).astype(np.float32)
            # Ensure it has 2 channels (X, Y components of flow)
            if flow.shape[0] != 2:
                raise RuntimeError(f"Expected 2 channels for flow, but got {flow.shape[0]} from {path}")

        except Exception as e:
            raise RuntimeError(f"Could not load flow file: {path}\n{e}")

    else:
        raise NotImplementedError(
            f"Flow from {ext} not implemented. Supports .npy and .exr"
        )

    # Return torch tensor C,H,W
    return torch.from_numpy(flow)


# -------------------------------------------------------------------------
# Dataset
# -------------------------------------------------------------------------

class GamingSupersampleDataset(Dataset):
    """
    Dataset for Efficient Neural Supersampling project.

    Expected directory structure (updated based on analysis):

    root/ (e.g., /content/drive/MyDrive/ENSS/data/QRISP/FloodedGrounds)
      1080p/
        CameraData/
        Enhanced/ (HR target)
        Native/
      270p/
        CameraData/
        DepthMipBiasMinus2/
        MotionVectorsMipBiasMinus2/
        MipBiasMinus2/ (LR input)
      ... other resolutions (360p, 540p) with similar subfolders ...

    Each folder contains frame-specific subfolders (e.g., '0000/', '0001/')
    which then contain the actual image/json/exr files.

    Returns a dict:
      {
        "lr_rgb":  (3, H, W)
        "depth":   (1, H, W)
        "motion":  (2, H, W) normalized to [-1,1]
        "jitter":  (2,) or (2,1,1)
        "hr_rgb":  (3, H*sf, W*sf)
      }
    """

    def __init__(
        self,
        root, # This should already be /content/drive/MyDrive/ENSS/data/QRISP/FloodedGrounds
        split="540p", # This now represents the LR resolution folder
        upscale_factor=2,
        use_jitter=True,
        jitter_channels=2,
        transform=None, # Added for augmentations
        target_transform=None, # Added for augmentations
    ):
        self.root = Path(root)
        self.split = split # e.g., '270p', '360p', '540p'
        self.upscale_factor = upscale_factor
        self.use_jitter = use_jitter
        self.jitter_channels = jitter_channels
        self.transform = transform
        self.target_transform = target_transform

        # Define dynamic folder mappings for modalities with mipmap biasing
        # These maps are used to construct paths like 'DepthMipBiasMinusX'
        self.depth_folder_map = {
            "540p": "DepthMipBiasMinus1",
            "360p": "DepthMipBiasMinus1.58",
            "270p": "DepthMipBiasMinus2",
        }
        self.motion_folder_map = {
            "540p": "MotionVectorsMipBiasMinus1",
            "360p": "MotionVectorsMipBiasMinus1.58",
            "270p": "MotionVectorsMipBiasMinus2",
        }
        # Assuming LR RGB is always 'MipBiasMinusX' (or Native in current context) at the `split` resolution
        # Let's adjust for 'Native' for LR RGB based on file listing.
        self.lr_rgb_folder_map = {
            "540p": "Native", # Or 'Native' if that's the base LR input. Let's start with Native.
            "360p": "Native", # Let's assume Native is the primary LR source
            "270p": "Native", # Let's assume Native is the primary LR source
        }

        # Base directories for modalities within the current split resolution
        # Use 'Native' for LR RGB as it's the simplest representation of LR.
        self.lr_base_dir = self.root / self.split / self.lr_rgb_folder_map.get(self.split, "Native")
        
        # Hardcoded depth directory path
        self.depth_dir = self.root / "270p" / "DepthMipBiasMinus2"

        self.motion_dir = self.root / self.split / self.motion_folder_map.get(self.split, f"MotionVectorsMipBiasMinus{self.split.replace('p', '')}") # Fallback logic
        # Use the chosen LR RGB path
        self.lr_rgb_dir = self.lr_base_dir

        self.jitter_base_dir = self.root / self.split / "CameraData" # Jitter data is in CameraData as JSON
        self.hr_base_dir = self.root / "1080p" / "Enhanced" # High-res target is 1080p/Enhanced

        # Sample list inferred from lr_base_dir, now iterating through camera_id folders
        self.samples = []
        # Correctly iterate through camera_id subfolders to get frame_ids
        for camera_folder_path in sorted(self.lr_base_dir.iterdir()):
            if camera_folder_path.is_dir():
                camera_id = camera_folder_path.name
                # Assuming .png for lr_rgb. Adjust if other formats are present.
                for frame_file_path in sorted(camera_folder_path.glob("*.png")):
                    frame_id = frame_file_path.stem
                    self.samples.append((camera_id, frame_id))

    def __len__(self):
        return len(self.samples)

    def _load_jitter(self, path):
        """Load jitter vector from JSON as torch tensor shape (2,)"""
        if not path.exists():
            # If jitter file doesn't exist, return zero vector (e.g., for non-jittered data)
            return torch.zeros(self.jitter_channels, dtype=torch.float32)
        
        with open(path, 'r') as f:
            camera_data = json.load(f)
        
        # Extract jitter_x and jitter_y. These keys are observed in the CameraData JSON.
        # The paper states jitter offset Jt is a 2D vector, so we expect two values.
        jitter_x = camera_data.get('jitter_offset_x', 0.0) # Using more specific key if available
        jitter_y = camera_data.get('jitter_offset_y', 0.0) # Using more specific key if available

        # Based on file listing, the `CameraData` json files have `jitter_offset_x` and `jitter_offset_y` keys.
        # Fallback to 0.0 if not found, though they should be present for jittered frames.
        return torch.tensor([jitter_x, jitter_y], dtype=torch.float32)

    def __getitem__(self, idx):
        camera_id, frame_id = self.samples[idx]

        # Construct paths based on the new directory structure
        # LR RGB is from Native folder within the selected split resolution
        lr_path = self.lr_rgb_dir / camera_id / f"{frame_id}.png"
        lr = load_image(lr_path)  # (3, H, W)

        # Depth uses dynamically determined folder name (e.g., DepthMipBiasMinus2)
        depth_path = self.depth_dir / camera_id / f"{frame_id}.png"
        depth = load_image(depth_path)  # (1, H, W)

        # Motion vectors use dynamically determined folder name (e.g., MotionVectorsMipBiasMinus2)
        motion_path = self.motion_dir / camera_id / f"{frame_id}.exr" # Confirmed .exr extension
        motion = load_flow(motion_path)  # (2, H, W)

        # HR RGB is from 1080p/Enhanced folder
        hr_path = self.hr_base_dir / camera_id / f"{frame_id}.png"
        hr = load_image(hr_path)  # (3, H*sf, W*sf)

        # Jitter is from CameraData folder, .json file
        jitter = torch.zeros(self.jitter_channels, dtype=torch.float32) # Default non-jittered
        if self.use_jitter:
            jitter_json_path = self.jitter_base_dir / camera_id / f"{frame_id}.json"
            jitter = self._load_jitter(jitter_json_path)

        # ------------------------------------------------------------------
        # Normalize motion to grid_sample coordinates [-1,1]
        # ------------------------------------------------------------------
        _, H, W = lr.shape

        # Convert pixel motion to normalized displacements
        # Horizontal range [-1,1] corresponds to [-W/2, W/2] pixels
        motion[0, ...] = motion[0, ...] / ((W - 1) / 2)
        motion[1, ...] = motion[1, ...] / ((H - 1) / 2)

        # Apply transforms
        if self.transform:
            lr = self.transform(lr)
        if self.target_transform:
            hr = self.target_transform(hr)

        return {
            "lr_rgb": lr,
            "depth": depth,
            "motion": motion,
            "jitter": jitter,
            "hr_rgb": hr
        }
