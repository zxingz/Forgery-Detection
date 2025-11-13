
# %%
import sys
import os
from glob import glob
import pickle
import os
import urllib
from functools import reduce
from collections import deque
import gc

from joblib import Parallel, delayed

from tqdm import tqdm

import numpy as np

from scipy.ndimage import convolve

import pandas as pd

from PIL import Image
from PIL import ImageEnhance

import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision.transforms.functional as TF

sys.path.insert(0, os.getcwd())

# import all utils
from utils import data_directory, \
                dinov3_repo_dir, \
                dinov3_vitb16_weight_raw, \
                dinov3_vith16_weight_raw, \
                dinov3_vit7B16_weight_raw, \
                dinov3_vits16_weight_raw
                
sys.path.insert(0, os.path.join(dinov3_repo_dir))

#  %%

forged_folder = os.path.join(data_directory, 'train_images', 'forged')
authentic_folder = os.path.join(data_directory, 'train_images', 'authentic')
mask_folder = os.path.join(data_directory, 'train_masks')

# Image Setting
PATCH_SIZE = 16
DEFAULT_IMAGE_SIZE = 512 # Should be multiple of PATCH_SIZE
MASK_THRESHOLD = 0

# Batch Settings
BATCH_SIZE = 8
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Available DINOv3 models:
MODEL_DINOV3_VITS = "dinov3_vits16"
MODEL_DINOV3_VITSP = "dinov3_vits16plus"
MODEL_DINOV3_VITB = "dinov3_vitb16"
MODEL_DINOV3_VITL = "dinov3_vitl16"
MODEL_DINOV3_VITHP = "dinov3_vith16plus"
MODEL_DINOV3_VIT7B = "dinov3_vit7b16"

MODEL_TO_NUM_LAYERS = {
    MODEL_DINOV3_VITS: 12,
    MODEL_DINOV3_VITSP: 12,
    MODEL_DINOV3_VITB: 12,
    MODEL_DINOV3_VITL: 24,
    MODEL_DINOV3_VITHP: 32,
    MODEL_DINOV3_VIT7B: 40,
}

MODEL_TO_EMBED_DIM = {
    MODEL_DINOV3_VITS: 384,
    MODEL_DINOV3_VITSP: 384,  # ViT-Small+
    MODEL_DINOV3_VITB: 768,
    MODEL_DINOV3_VITL: 1024,
    MODEL_DINOV3_VITHP: 1536, # ViT-Huge+
    MODEL_DINOV3_VIT7B: 4096,
}

MODEL_TO_WEIGHT_FILE = {
    MODEL_DINOV3_VITS: dinov3_vits16_weight_raw,
    MODEL_DINOV3_VITB: dinov3_vitb16_weight_raw,
    MODEL_DINOV3_VITHP: dinov3_vith16_weight_raw, # ViT-Huge+
}

def count_components_floodfill(row:dict, connectivity: int = 8):
    """
    Count connected True components in a 2D boolean numpy mask using flood-fill (iterative).
    Args:
        mask: 2D boolean numpy array
        connectivity: 4 or 8 (neighbour connectivity)
    Returns:
        n_components: int
        sizes: list of int (size of each component)
        labels: 2D int32 array same shape as mask where 0 = background, 1..n = component ids
    """
    mask = row['bool_mask']
    
    assert mask.ndim == 2, "mask must be 2D"
    assert connectivity in (4, 8)
    H, W = mask.shape
    visited = np.zeros((H, W), dtype=bool)
    # labels = np.zeros((H, W), dtype=np.int32)
    neighbors4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    neighbors8 = neighbors4 + [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    neighbors = neighbors8 if connectivity == 8 else neighbors4

    comp_id = 0
    # sizes = []

    for y in range(H):
        for x in range(W):
            if mask[y, x] and not visited[y, x]:
                comp_id += 1
                q = deque()
                q.append((y, x))
                visited[y, x] = True
                # labels[y, x] = comp_id
                # size = 0
                while q:
                    cy, cx = q.popleft()
                    # size += 1
                    for dy, dx in neighbors:
                        ny, nx = cy + dy, cx + dx
                        if 0 <= ny < H and 0 <= nx < W and not visited[ny, nx] and mask[ny, nx]:
                            visited[ny, nx] = True
                            # labels[ny, nx] = comp_id
                            q.append((ny, nx))
                # sizes.append(size)

    return {"name":row["name"], "index":row["index"], "bool_mask":mask, "n_comps":comp_id} #, sizes, labels

def resize_image_to_fit_patch(
    image: Image,
    image_size: int = DEFAULT_IMAGE_SIZE,
    patch_size: int = PATCH_SIZE,
) -> Image:
    w, h = image.size
    w_hat = h_hat = PATCH_SIZE * (image_size // PATCH_SIZE)
    resized_img = image.resize((w_hat, h_hat), \
                    resample=Image.Resampling.LANCZOS)
    return resized_img

def resized_image_to_mask(image_resized, mask_threshold: int = MASK_THRESHOLD):
    image_array = np.array(image_resized)
    mask = image_array > mask_threshold
    return mask

def mask_to_resized_image(mask):
    image = Image.fromarray((mask * 255).astype(np.uint8), 'L')
    image_resized = resize_image_to_fit_patch(image)
    return image_resized

def resize_mask_to_fit_patch(
    mask: np.ndarray,
    mask_threshold: int = MASK_THRESHOLD,
) -> np.ndarray:
    resized_image = mask_to_resized_image(mask)
    return resized_image_to_mask(resized_image)

def pixel_to_patch_coords(y_pixel, x_pixel, patch_size=PATCH_SIZE):
    y_coord = y_pixel // patch_size
    x_coord = x_pixel // patch_size
    return y_coord, x_coord

# Convert auth_mask to patch coordinates and mask the similarity map
def pixel_mask_to_patch_float(pixel_mask, patch_size=PATCH_SIZE):
    """Convert pixel-level mask to patch-level mask"""
    H_pixels, W_pixels = pixel_mask.shape
    H_patches = H_pixels // patch_size
    W_patches = W_pixels // patch_size
    
    patch_mask = np.zeros((H_patches, W_patches), dtype=float)
    
    for i in range(H_patches):
        for j in range(W_patches):
            # Check if any pixel in this patch is True
            patch_region = pixel_mask[i*patch_size:(i+1)*patch_size, 
                                     j*patch_size:(j+1)*patch_size]
            patch_mask[i, j] = np.sum(patch_region) / (patch_size * patch_size)
    
    return patch_mask

def load_10_mask_entities(name=None, idxs=None):
    all_mask_entities = []
    if name is not None and idxs is not None:
        for idx in idxs:
            file = os.path.join("mask_entities", f"{name}_{idx}.pkl")
            all_mask_entities.append(pickle.load(open(file, 'rb')))
        return all_mask_entities
    files = glob(os.path.join("mask_entities", '*.pkl'))[:10]
    all_mask_entities = []
    for file in files:
        mask_entities = pickle.load(open(file, 'rb'))
        all_mask_entities.append(mask_entities)
    return all_mask_entities

def get_batches(batch_size:int = BATCH_SIZE):
    
    # Files
    files = glob(os.path.join("mask_entities", '*.pkl'))
    
    # Names
    names = sorted(list({x.split(os.sep)[-1].split('_')[0] for x in files}))
    
    i = 0
    while i < len(names):
        forged_imgs = []
        auth_masks = []
        forged_masks = []
        ids = []
        
        batch_files = names[i:i+batch_size]
        
        for name in batch_files:
            
            ids.append(name)
            
            name_files = glob(os.path.join("mask_entities", f'{name}_*.pkl'))
            
            forged_image_path = os.path.join(forged_folder, name + '.png')
            forged_img = Image.open(forged_image_path).convert('RGB')
            forged_img_resized = resize_image_to_fit_patch(forged_img)
            forged_img_np = np.array(forged_img_resized)
            forged_imgs.append(forged_img_np)
            
            combined_auth_mask = []
            combined_forged_mask = []
            
            for file in name_files:
                
                # Masked entities
                me = pickle.load(open(file, 'rb'))
                
                auth_mask = resize_mask_to_fit_patch(me["auth_mask"])
                # auth_mask = pixel_mask_to_patch_float(auth_mask)
                combined_auth_mask.append(auth_mask)
                
                forged_mask = resize_mask_to_fit_patch(me["forged_mask"])
                # forged_mask = pixel_mask_to_patch_float(forged_mask)
                combined_forged_mask.append(forged_mask)
            
            # Combine masks by taking logical OR across entities
            combined_auth_mask = np.logical_or.reduce(combined_auth_mask)
            combined_auth_mask = pixel_mask_to_patch_float(combined_auth_mask)
            auth_masks.append(combined_auth_mask)
            
            combined_forged_mask = np.logical_or.reduce(combined_forged_mask)
            combined_forged_mask = pixel_mask_to_patch_float(combined_forged_mask)
            forged_masks.append(combined_forged_mask)
            
        forged_imgs = np.stack(forged_imgs, axis=0)
        auth_masks = np.stack(auth_masks, axis=0)
        forged_masks = np.stack(forged_masks, axis=0)
        
        yield forged_imgs, auth_masks, forged_masks, ids
        
        i += batch_size


# %%
# Model

MODEL_NAME = MODEL_DINOV3_VITB # MODEL_DINOV3_VITHP # MODEL_DINOV3_VITS
N_LAYERS = MODEL_TO_NUM_LAYERS[MODEL_NAME]
EMBED_DIM = MODEL_TO_EMBED_DIM[MODEL_NAME]
WEIGHT_FILE = MODEL_TO_WEIGHT_FILE[MODEL_NAME]

#  %%
class SegmentationHead(nn.Module):
    def __init__(self, in_channels, hidden_dim=EMBED_DIM):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, hidden_dim, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(hidden_dim)
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(hidden_dim)
        self.conv3 = nn.Conv2d(hidden_dim, 1, 1)  # Output 1 channel for mask
        
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = torch.sigmoid(self.conv3(x))  # Sigmoid for probability mask
        return x

class DINOv3Segmentation(nn.Module):
    def __init__(self, backbone, in_channels=EMBED_DIM):
        super().__init__()
        self.backbone = backbone
        self.seg_head = SegmentationHead(in_channels)
        
        # Freeze backbone
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        # Initialize segmentation head
        for m in self.seg_head.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Get features from backbone
        features = self.backbone.get_intermediate_layers(
            x,
            n=range(N_LAYERS),
            reshape=True,
            norm=True
        )
        last_layer_features = features[-1]  # Shape: [B, C, H, W]
        
        # Pass through segmentation head
        mask = self.seg_head(last_layer_features)  # Shape: [B, 1, H, W]
        return mask

# %%

# Loss Function

def auth_and_forged_sim_mat(auth_mask, forged_mask):
        """
        Calculate similarity matrix between authentic and forged patches using matrix multiplication
            
        Returns:
            similarity_matrix: Tensor of shape [H1, W1, H2, W2]
        """
        
        # Get dimensions
        F1, W1 = auth_mask.shape
        F2, W2 = forged_mask.shape
            
        similarity_matrix = torch.matmul(auth_mask, forged_mask.transpose(-1, -2))
        
        return similarity_matrix

class GramLoss(nn.Module):
    """Implementation of the gram loss"""

    def __init__(
        self,
        apply_norm=False,
        remove_neg=True,
        remove_only_teacher_neg=False,
    ):
        super().__init__()

        # Loss
        self.mse_loss = torch.nn.MSELoss()

        # Parameters
        self.apply_norm = apply_norm
        self.remove_neg = remove_neg
        self.remove_only_teacher_neg = remove_only_teacher_neg

        if self.remove_neg or self.remove_only_teacher_neg:
            assert self.remove_neg != self.remove_only_teacher_neg


    def forward(self, output_feats, target_sim, img_level=True):
        """Compute the MSE loss between the gram matrix of the input and target features.
        Returns:
            loss: scalar
        """

        # Dimensions of the tensor should be (B, N, dim)
        if img_level:
            assert len(output_feats.shape) == 3
        
        # print('target shape', target_sim.shape)

        # Patch correlation
        if self.apply_norm:
            output_feats = F.normalize(output_feats, dim=-1)

        if not img_level and len(output_feats.shape) == 3:
            # Flatten (B, N, D) into  (B*N, D)
            output_feats = output_feats.flatten(0, 1)
            
        # print(output_feats.shape)

        # Compute similarities
        student_sim = torch.matmul(output_feats, output_feats.transpose(-1, -2))
        
        # print(student_sim.shape)

        if self.remove_neg:
            target_sim[target_sim < 0] = 0.0
            student_sim[student_sim < 0] = 0.0

        elif self.remove_only_teacher_neg:
            # Remove only the negative sim values of the teacher
            target_sim[target_sim < 0] = 0.0
            student_sim[(student_sim < 0) & (target_sim < 0)] = 0.0

        return self.mse_loss(student_sim, target_sim)
    
class TverskyLoss(nn.Module):
    """Tversky Loss for segmentation, generalization of Dice Loss."""
    
    def __init__(self, alpha=0.5, beta=0.5):
        super().__init__()
        self.alpha = alpha  # Penalizes false positives
        self.beta = beta    # Penalizes false negatives
    
    def forward(self, pred, target):
        """
        Compute Tversky Loss.
        pred: (B, 1, H, W) or (B, H, W), sigmoid probabilities
        target: (B, H, W) or (B, 1, H, W), ground truth mask (0 or 1, or float)
        """
        if pred.dim() == 4 and pred.shape[1] == 1:
            pred = pred.squeeze(1)
        if target.dim() == 4 and target.shape[1] == 1:
            target = target.squeeze(1)
        
        target = target.float()
        
        tp = (pred * target).sum(dim=(1, 2))
        fp = (pred * (1 - target)).sum(dim=(1, 2))
        fn = ((1 - pred) * target).sum(dim=(1, 2))
        
        tversky_index = tp / (tp + self.alpha * fp + self.beta * fn + 1e-6)
        loss = 1 - tversky_index.mean()
        return loss
    
    
    
#  %%

# Training Loop

BATCH_SIZE = 128

tversky_loss_fn = TverskyLoss(alpha=0.7, beta=0.5)

# Create segmentation model
model = torch.hub.load(
    repo_or_dir=dinov3_repo_dir,
    model=MODEL_NAME,
    source="local",
    weights=WEIGHT_FILE,
)
seg_model = DINOv3Segmentation(model, in_channels=EMBED_DIM).to(DEVICE)
seg_model.train()

# # Create optimizer for segmentation head only
optimizer = torch.optim.AdamW(
    seg_model.seg_head.parameters(),  # Only optimize segmentation head
    lr=1e-4,
    weight_decay=0.05
)

for epoch in tqdm(range(100)):
    total_loss = 0.0
    
    # Zero gradients
    optimizer.zero_grad()
    
    for img_batch, auth_masks, forged_masks, _ in get_batches(batch_size=BATCH_SIZE):
        
        img_batch = torch.from_numpy(img_batch).permute(0, 3, 1, 2).float().div(255.0).to(DEVICE)  # (B, 3, H, W)
        img_batch = TF.normalize(img_batch, mean=IMAGENET_MEAN, std=IMAGENET_STD)
        forged_masks_tensor = torch.from_numpy(forged_masks).to(DEVICE)  # (B, H, W)
        
        # Get model output
        with torch.amp.autocast_mode.autocast(device_type=DEVICE.type, dtype=torch.float):
            
            seg_output = seg_model(img_batch)  # (B, 1, H, W)
            
            print(seg_output.shape, forged_masks_tensor.shape)
            
            # Calculate loss
            loss = tversky_loss_fn(seg_output, forged_masks_tensor)
        
            # Backward pass
            loss.backward()
            optimizer.step()
            
            print(f'Epoch {epoch}, Loss: {loss.item()}')
        
        # Update progress
        total_loss += loss.item()
        
        # Save model checkpoint with training info
        checkpoint = {
            'model_state_dict': seg_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'model_name': MODEL_NAME,
            'num_layers': N_LAYERS,
            'embed_dim': EMBED_DIM,
            'epoch': epoch,
            'total_loss': total_loss,
        }

        # Create checkpoints directory if it doesn't exist 
        os.makedirs('checkpoints', exist_ok=True)

        # Save checkpoint
        checkpoint_path = os.path.join('weights', f'{MODEL_NAME}_dinov3_peft.pth')
        torch.save(checkpoint, checkpoint_path)

        print(f"Model checkpoint saved to {checkpoint_path}")
# %%
