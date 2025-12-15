import os
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms
import torchvision.transforms.functional as TF
import json # Ensure json is imported for _load_jitter
from base import BaseDataLoader


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
    Supports EXR (via OpenEXR + Imath) or numpy binary formats.
    Modify this for your dataset's actual flow format.
    """
    ext = Path(path).suffix.lower()

    if ext == ".npy":
        flow = np.load(path).astype(np.float32)  # H, W, 2

    elif ext in [".png", ".jpg"]:
        # Optional: some datasets store encoded flow in RGB channels
        raise NotImplementedError(
            f"Flow from {ext} not implemented. Convert motion to .npy or .exr"
        )

    else:
        # EXR likely:
        try:
            import OpenEXR, Imath
            pt = Imath.PixelType(Imath.PixelType.FLOAT)
            file = OpenEXR.InputFile(path)
            dw = file.header()['dataWindow']
            W = dw.max.x - dw.min.x + 1
            H = dw.max.y - dw.min.y + 1

            # Expect channels "X" and "Y"
            flow_x = np.frombuffer(file.channel("X", pt), dtype=np.float32).reshape(H, W)
            flow_y = np.frombuffer(file.channel("Y", pt), dtype=np.float32).reshape(H, W)
            flow = np.stack([flow_x, flow_y], axis=-1)

        except Exception as e:
            raise RuntimeError(f"Could not load flow file: {path}\n{e}")

    # Return torch tensor C,H,W
    return torch.from_numpy(flow).permute(2, 0, 1)


# -------------------------------------------------------------------------
# Dataset
# -------------------------------------------------------------------------

class GamingSupersampleDataset(Dataset):
    """
    Dataset for Efficient Neural Supersampling project.

    Expected directory structure:

    root/
      train/ or val/ (e.g., 540p)
        Native/ (LR RGB)
          [camera_id_folder]/
            [frame_id].png
        DepthMipBiasMinusX/ (Depth, where X varies by resolution)
          [camera_id_folder]/
            [frame_id].png
        MotionVectorsMipBiasMinusX/ (Motion, where X varies by resolution)
          [camera_id_folder]/
            [frame_id].exr # Corrected extension to .exr
        CameraData/ (Jitter - contains JSON files in camera_id_folders)
          [camera_id_folder]/
            [frame_id].json
      1080p/ (HR RGB - fixed resolution, usually higher than split)
          [camera_id_folder]/
            [frame_id].png

    Each folder contains frames named the same:
      000001.png, 000002.png, ...

    Returns a dict:
      {{
        "lr_rgb":  (3, H, W)
        "depth":   (1, H, W)
        "motion":  (2, H, W) normalized to [-1,1]
        "jitter":  (2,) or (2,1,1)
        "hr_rgb":  (3, H*sf, W*sf)
      }}
    """

    def __init__(
        self,
        root,
        split= "540p", #"train",
        upscale_factor=2,
        use_jitter=True,
        jitter_channels=2,
        transform=None, # Added for augmentations
        target_transform=None, # Added for augmentations
    ):
        self.root = Path(root) # Changed to Path(root) for consistency and Path methods
        self.split = split
        self.upscale_factor = upscale_factor
        self.use_jitter = use_jitter
        self.jitter_channels = jitter_channels

        # Folders - Note: These now point to the parent directories of camera_id_folders
        self.lr_base_dir = self.root / split / "Native"
        
        # Determine correct depth folder name based on split
        depth_folder_map = {
            "540p": "DepthMipBiasMinus1",
            "360p": "DepthMipBiasMinus1.58",
            "270p": "DepthMipBiasMinus2",
        }
        self.depth_base_dir = self.root / split / depth_folder_map.get(split, "DepthMipBiasMinus1") # Default to Minus1

        # Determine correct motion folder name based on split
        motion_folder_map = {
            "540p": "MotionVectorsMipBiasMinus1",
            "360p": "MotionVectorsMipBiasMinus1.58",
            "270p": "MotionVectorsMipBiasMinus2",
        }
        self.motion_base_dir = self.root / split / motion_folder_map.get(split, "MotionVectorsMipBiasMinus1") # Default to Minus1

        self.jitter_base_dir = self.root / split / "CameraData" # Jitter data is in CameraData as JSON
        self.hr_base_dir = self.root / "1080p" / "Enhanced" # High-res target is 1080p/Enhanced

        # Sample list inferred from lr_rgb folder, now enumerating (camera_id, frame_id)
        self.samples = []
        for camera_folder_path in sorted(self.lr_base_dir.iterdir()):
            if camera_folder_path.is_dir():
                camera_id = camera_folder_path.name
                for frame_file_path in sorted(camera_folder_path.glob("*.png")):
                    frame_id = frame_file_path.stem
                    self.samples.append((camera_id, frame_id))

        self.transform = transform
        self.target_transform = target_transform

    def __len__(
            self):
        return len(self.samples)

    def _load_jitter(self, path): # Modified to load from JSON
        """Load jitter vector from JSON as torch tensor shape (2,)"""
        if not path.exists():
            return torch.zeros(2)
        
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

        # ---- Load low-resolution RGB ----
        lr_path = self.lr_base_dir / camera_id / f"{frame_id}.png"
        lr = load_image(lr_path)  # (3, H, W)

        # ---- Depth ----
        depth_path = self.depth_base_dir / camera_id / f"{frame_id}.png"
        depth = load_image(depth_path)  # (1, H, W)

        # ---- Motion (flow) ----
        motion_path = self.motion_base_dir / camera_id / f"{frame_id}.exr"
        motion = load_flow(motion_path)  # (2, H, W)

        # ---- High-resolution RGB ----
        hr_path = self.hr_base_dir / camera_id / f"{frame_id}.png"
        hr = load_image(hr_path)  # (3, H*sf, W*sf)

        # ---- Jitter (optional) ----
        if self.use_jitter:
            # The Jitter folder appears to be inside CameraData as JSON
            jitter_path = self.jitter_base_dir / camera_id / f"{frame_id}.json"
            jitter = self._load_jitter(jitter_path)
        else:
            jitter = torch.zeros(2)

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
class MnistDataLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        trsfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.data_dir = data_dir
        self.dataset = datasets.MNIST(self.data_dir, train=training, download=True, transform=trsfm)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

class GamingSupersampleDataLoader(BaseDataLoader):
    """
    DataLoader wrapper for the GamingSupersampleDataset.
    Only the BaseDataLoader parameters go into super().__init__():
        dataset, batch_size, shuffle, validation_split, num_workers
    """
    def __init__(self, data_dir, split, upscale_factor, use_jitter, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True,
                 mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], scale=256, crop_size=224,
                 random_crop_size=None,
                 random_horizontal_flip=False,
                 random_vertical_flip=False,
                 color_jitter=0.0,
                 random_rotation=0.0):
        # These are passed directly to the dataset
        self.data_dir = data_dir
        self.training = training
        
        # New dynamic transform logic
        normalize = transforms.Normalize(mean=mean, std=std)

        # Create lists for transforms
        transforms_list = []
        target_transforms_list = [] # Assuming target also needs transformations

        # Initial common transforms: Scaling
        transforms_list.append(transforms.Resize(scale))
        target_transforms_list.append(transforms.Resize(scale))

        # Training-specific augmentations (applied conditionally)
        if self.training:
            if random_horizontal_flip:
                transforms_list.append(transforms.RandomHorizontalFlip())
            if random_vertical_flip:
                transforms_list.append(transforms.RandomVerticalFlip())
            if color_jitter > 0:
                transforms_list.append(transforms.ColorJitter(brightness=color_jitter, contrast=color_jitter, saturation=color_jitter, hue=color_jitter))
            if random_rotation > 0:
                transforms_list.append(transforms.RandomRotation(random_rotation))

            # Random cropping (applied conditionally during training)
            if random_crop_size:
                # To apply the same random crop to both HR and LR, we need to generate parameters once.
                # This is typically done in the Dataset's __getitem__ or with a custom paired transform.
                # For simplicity here, apply independent random crops. Further refinement might be needed.
                transforms_list.append(transforms.RandomCrop(random_crop_size))
                target_transforms_list.append(transforms.RandomCrop(random_crop_size))
            elif crop_size: # If no random crop, apply center crop if crop_size is defined
                transforms_list.append(transforms.CenterCrop(crop_size))
                target_transforms_list.append(transforms.CenterCrop(crop_size))
        else: # For validation/testing, if no random crop, apply center crop
            if crop_size:
                transforms_list.append(transforms.CenterCrop(crop_size))
                target_transforms_list.append(transforms.CenterCrop(crop_size))

        # Final common transforms
        transforms_list.extend([
            transforms.ToTensor(),
            normalize
        ])
        target_transforms_list.extend([
            transforms.ToTensor(),
            normalize
        ])

        transform = transforms.Compose(transforms_list)
        target_transform = transforms.Compose(target_transforms_list)

        self.dataset = GamingSupersampleDataset(data_dir, split, upscale_factor=upscale_factor, use_jitter=use_jitter, transform=transform, target_transform=target_transform)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)
