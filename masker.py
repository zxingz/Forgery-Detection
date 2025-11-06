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
import torchvision.transforms.functional as TF

sys.path.insert(0, os.getcwd())

# import all utils
from utils import data_directory, \
                dinov3_repo_dir, \
                dinov3_vitb16_weight_raw, \
                dinov3_vith16_weight_raw, \
                dinov3_vit7B16_weight_raw
                
sys.path.insert(0, os.path.join(dinov3_repo_dir))

from dinov3.loss.gram_loss import GramLoss
    
    
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


forged_folder = os.path.join(data_directory, 'train_images', 'forged')
authentic_folder = os.path.join(data_directory, 'train_images', 'authentic')
mask_folder = os.path.join(data_directory, 'train_masks')

PATCH_SIZE = 16
DEFAULT_IMAGE_SIZE = 512 # Should be multiple of PATCH_SIZE
MASK_THRESHOLD = 0

def resize_image_to_fit_patch(
    image: Image,
    image_size: int = DEFAULT_IMAGE_SIZE,
    patch_size: int = PATCH_SIZE,
) -> Image:
    w, h = image.size
    h_hat = image_size
    w_hat = PATCH_SIZE * int(((w/h)*image_size)// PATCH_SIZE)
    resized_img = image.resize((w_hat, h_hat), \
                    resample=Image.Resampling.LANCZOS)
    return resized_img

def mask_to_resized_image(mask):
    image = Image.fromarray((mask * 255).astype(np.uint8), mode='L')
    image_resized = resize_image_to_fit_patch(image)
    return image_resized

def resize_mask_to_patch_dimensions(
    mask: np.ndarray,
    mask_threshold: int = MASK_THRESHOLD,
) -> np.ndarray:
    resized_image = mask_to_resized_image(mask)
    w_hat, h_hat = resized_image.size
    patched_image = resized_image.resize(
        (w_hat // PATCH_SIZE, h_hat // PATCH_SIZE),
        resample=Image.Resampling.NEAREST)
    patched_array = np.array(patched_image)
    patched_mask = patched_array > mask_threshold
    return patched_mask

def resized_image_to_mask(image_resized, mask_threshold: int = MASK_THRESHOLD):
    image_array = np.array(image_resized)
    mask = image_array > mask_threshold
    return mask

def pixel_to_patch_coords(y_pixel, x_pixel, patch_size=PATCH_SIZE):
    y_coord = y_pixel // patch_size
    x_coord = x_pixel // patch_size
    return y_coord, x_coord

# Convert auth_mask to patch coordinates and mask the similarity map
def pixel_mask_to_patch_mask(pixel_mask, patch_size=PATCH_SIZE):
    """Convert pixel-level mask to patch-level mask"""
    H_pixels, W_pixels = pixel_mask.shape
    H_patches = H_pixels // patch_size
    W_patches = W_pixels // patch_size
    
    patch_mask = np.zeros((H_patches, W_patches), dtype=bool)
    
    for i in range(H_patches):
        for j in range(W_patches):
            # Check if any pixel in this patch is True
            patch_region = pixel_mask[i*patch_size:(i+1)*patch_size, 
                                     j*patch_size:(j+1)*patch_size]
            patch_mask[i, j] = np.any(patch_region)
    
    return patch_mask


def softmax(x):
    """Compute softmax values for each set of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def smooth_boolean_mask(mask, kernel_size=3):
    """
    Smooths a boolean pixel mask based on a majority vote of neighboring pixels.
    Kernel/accumulators use int32 to avoid overflow for large kernels.
    """
    if kernel_size % 2 == 0 or kernel_size < 3:
        raise ValueError("kernel_size must be an odd integer >= 3")

    # Use int32 to avoid overflow when summing large neighborhoods
    int_mask = mask.astype(np.int32)

    kernel = np.ones((kernel_size, kernel_size), dtype=np.int32)

    neighbor_sum = convolve(int_mask, kernel, mode='constant', cval=0)

    # majority threshold excluding center? here we keep original behavior:
    majority_threshold = (kernel_size * kernel_size) / 2

    new_mask = neighbor_sum > majority_threshold

    return new_mask.astype(bool)

def smooth_boolean_mask2(mask, kernel_size=3):
    """
    Smooth boolean mask: a pixel becomes False only if a strict majority of its neighbors
    (excluding itself) are False; otherwise it keeps its original value.

    Uses int32 accumulator to prevent overflow when kernel_size grows.
    """
    if kernel_size % 2 == 0 or kernel_size < 3:
        raise ValueError("kernel_size must be an odd integer >= 3")

    # Use int32 to avoid overflow
    int_mask = mask.astype(np.int32)

    kernel = np.ones((kernel_size, kernel_size), dtype=np.int32)

    # Convolve to get sum including center
    neighbor_sum_incl = convolve(int_mask, kernel, mode='constant', cval=0)

    # Sum of neighbors only (exclude the center pixel)
    neighbors_sum = neighbor_sum_incl - int_mask

    # Number of neighbors (excluding center)
    n_neighbors = kernel_size * kernel_size - 1

    # Count of False neighbors
    false_neighbors = n_neighbors - neighbors_sum

    # Majority threshold: strict majority of neighbors being False
    majority_threshold = n_neighbors // 2

    # New mask: keep original value unless majority of neighbors are False -> set False
    new_mask = mask.copy().astype(bool)
    majority_false = false_neighbors > majority_threshold
    new_mask[majority_false] = False

    return new_mask

def process_mask(row):
    
    mask_name = row["name"]
    
    # print(mask_name, row["index"])
    
    forged_image_path = os.path.join(forged_folder, mask_name+'.png')
    forged_img = Image.open(forged_image_path).convert('RGB')

    auth_image_path = os.path.join(authentic_folder, mask_name+'.png')
    auth_img = Image.open(auth_image_path).convert('RGB')
    
    # Subtract auth_img from forged_img and convert to boolean mask
    forged_array = np.sum(np.array(forged_img), axis=2)
    forged_array = (forged_array - np.min(forged_array))/(np.max(forged_array) - np.min(forged_array))
    
    # # visualize the masks
    # plt.figure(figsize=(12, 4))
    # plt.subplot(1, 2, 1)
    # plt.imshow(forged_array, cmap='gray')
    # plt.title(f'Forged')
    # plt.axis('off')
    
    auth_array = np.sum(np.array(auth_img), axis=2)
    auth_array = (auth_array - np.min(auth_array))/(np.max(auth_array) - np.min(auth_array))
    
    # # visualize the masks
    # plt.figure(figsize=(12, 4))
    # plt.subplot(1, 2, 1)
    # plt.imshow(auth_array, cmap='gray')
    # plt.title(f'Auth')
    # plt.axis('off')

    # Calculate absolute difference
    diff_array = np.abs(forged_array.astype(np.float32) - auth_array.astype(np.float32))

    # Convert to grayscale by taking mean across channels
    # diff_gray = np.mean(diff_array, axis=2)

    # Create boolean mask (threshold can be adjusted)
    DIFF_THRESHOLD = 0  # Adjust this value as needed
    diff_boolean_mask = diff_array > DIFF_THRESHOLD
    diff_boolean_mask = diff_boolean_mask.astype(bool)
    
    # # visualize the masks
    # plt.figure(figsize=(12, 4))
    # plt.subplot(1, 2, 1)
    # plt.imshow(diff_boolean_mask, cmap='gray')
    # plt.title(f'Diff')
    # plt.axis('off')
    
    if not np.any(diff_boolean_mask):
        return None
    
    # Smooth the diff boolean mask
    kernel_size = 3
    last_diff_boolean_mask = diff_boolean_mask.copy()
    it = 0
    while 1:
        while 1:
            it += 1
            prev_mask = diff_boolean_mask.copy()
            diff_boolean_mask = smooth_boolean_mask2(diff_boolean_mask, kernel_size=kernel_size)
            if np.array_equal(prev_mask, diff_boolean_mask):
                del prev_mask
                gc.collect()
                break
        
        n_comps = count_components_floodfill({"name":mask_name, \
                                    "index":-1, \
                                    "bool_mask":diff_boolean_mask})["n_comps"]
        kernel_size += 2
        if not np.any(diff_boolean_mask):
            break
        if np.array_equal(last_diff_boolean_mask, diff_boolean_mask):
            continue
        last_diff_boolean_mask = diff_boolean_mask
        if n_comps <= row["n_comps"]-1:
            break
        
    diff_boolean_mask = last_diff_boolean_mask
    
    # # visualize the masks
    # plt.figure(figsize=(12, 4))
    # plt.subplot(1, 2, 1)
    # plt.imshow(diff_boolean_mask, cmap='gray')
    # plt.title(f'Diff')
    # plt.axis('off')
    
    
    # diff_boolean_mask = resize_mask_to_patch_dimensions(diff_boolean_mask)
    
    # # visualize the masks
    # plt.figure(figsize=(12, 4))
    # plt.subplot(1, 2, 1)
    # plt.imshow(diff_boolean_mask, cmap='gray')
    # plt.title(f'Diff')
    # plt.axis('off')
    
    masked_img_boolean = row["bool_mask"]
    # masked_img_boolean = resize_mask_to_patch_dimensions(masked_img_boolean)
    
    
    # find intersections between diff_boolean_mask and masked_img_boolean
    forged_mask = np.logical_and(diff_boolean_mask, masked_img_boolean)
    if not np.any(diff_boolean_mask):
        return None
    
    # Smooth the boolean mask
    kernel_size = 3
    last_forged_mask = forged_mask.copy()
    it = 0
    while 1:
        while 1:
            it += 1
            prev_mask = forged_mask.copy()
            forged_mask = smooth_boolean_mask2(forged_mask, kernel_size=kernel_size)
            if np.array_equal(prev_mask, forged_mask):
                del prev_mask
                gc.collect()
                break
        n_comps = count_components_floodfill({"name":mask_name, \
                                    "index":-1, \
                                    "bool_mask":forged_mask})["n_comps"]
        kernel_size += 2
        if not np.any(forged_mask):
            break
        if np.array_equal(last_forged_mask, forged_mask):
            continue
        last_forged_mask = forged_mask
        if n_comps <= row["n_comps"]-1:
            break
    forged_mask = last_forged_mask
    
    # # visualize the masks
    # plt.figure(figsize=(12, 4))
    # plt.subplot(1, 2, 1)
    # plt.imshow(resize_mask_to_patch_dimensions(forged_mask), cmap='gray')
    # plt.title(f'Forged')
    # plt.axis('off')
    
    # get auth mask
    auth_mask = masked_img_boolean & ~forged_mask
    if not np.any(auth_mask):
        return None
    
    # Smooth the boolean mask
    kernel_size = 3
    last_auth_mask = auth_mask.copy()
    it = 0
    while 1:
        while 1:
            it += 1
            prev_mask = auth_mask.copy()
            auth_mask = smooth_boolean_mask(auth_mask, kernel_size=kernel_size)
            if np.array_equal(prev_mask, auth_mask):
                del prev_mask
                gc.collect()
                break
        n_comps = count_components_floodfill({"name":mask_name, \
                                    "index":-1, \
                                    "bool_mask":auth_mask})["n_comps"]
        kernel_size += 2
        if not np.any(auth_mask):
            break
        if np.array_equal(last_auth_mask, auth_mask):
            continue
        last_auth_mask = auth_mask
        if n_comps <= 1:
            break
    auth_mask = last_auth_mask
    
    # # visualize the masks
    # plt.figure(figsize=(12, 4))
    # plt.subplot(1, 2, 1)
    # plt.imshow(resize_mask_to_patch_dimensions(auth_mask), cmap='gray')
    # plt.title(f'Auth')
    # plt.axis('off')


    me = {"mask_name": mask_name, \
                "index": row["index"], \
                "auth_mask": resize_mask_to_patch_dimensions(auth_mask), \
                "forged_mask": resize_mask_to_patch_dimensions(forged_mask),}
    return me
    # pickle.dump(me, open(os.path.join('mask_entities', f'{mask_name}.pkl'), 'wb'))

component_counts = pickle.load(open('component_counts.pkl', 'rb'))

# chunk_size = 24  # adjust as needed
# total = len(component_counts)
# n_chunks = (total + chunk_size - 1) // chunk_size
# for chunk_idx, start in enumerate(range(0, total, chunk_size), 1):
#     end = min(start + chunk_size, total)
#     chunk = component_counts[start:end]
#     print(f"Processing chunk {chunk_idx}/{n_chunks} (items {start}:{end})")
#     # process this chunk in parallel
#     Parallel(n_jobs=-1)(delayed(process_mask)(row) for row in tqdm(chunk))
#     # masked_entities = [item for sublist in masked_entities for item in sublist]
#     # pickle.dump(masked_entities, open(os.path.join('mask_entities', f'mask_components{chunk_idx}.pkl'), 'wb'))
    


# for row in tqdm(component_counts):
#     process_mask(row)

masked_entities = Parallel(n_jobs=-1)(delayed(process_mask)(row) for row in tqdm(component_counts))
# # masked_entities = [process_mask(component_counts[0])] # 19684  13511
# masked_entities = [process_mask(x) for x in component_counts if x["name"]=="19684"] # 19684  13511
masked_entities = [item for item in masked_entities if item is not None]
pickle.dump(masked_entities, open('mask_components.pkl', 'wb'))