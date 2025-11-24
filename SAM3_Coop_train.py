# # %%
# !pip intall pip install git+https://github.com/huggingface/transformers

# %%
import os
import sys
import requests
from typing import Optional
from glob import glob
import pickle

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from tqdm import tqdm

import torch
import torch.nn as nn
from torch.optim import AdamW
import torchvision.transforms.functional as TVF

from PIL import Image

from transformers import Sam3Processor, Sam3Model
from transformers import CLIPTextConfig

sys.path.insert(0, os.getcwd())

from config import Config
config = Config(data_directory=r"D:\RecodAI\recodai-luc-scientific-image-forgery-detection")
config.DINOV3_DEVICE = torch.device('cpu')


from DinoV3_Train import resize_image_to_fit_patch, resize_mask_to_fit_patch, pixel_mask_to_patch_float


#  %%
sam3_model = Sam3Model.from_pretrained("facebook/sam3").to(config.DINOV3_DEVICE)
sam3_processor = Sam3Processor.from_pretrained("facebook/sam3")

# %%

class CLIPTextEmbeddings(nn.Module):
    def __init__(self, config: CLIPTextConfig, n_ctx=16, ctx_init=None):
        super().__init__()
        self.n_ctx = n_ctx
        self.embed_dim = config.hidden_size
        self.token_embedding = nn.Embedding(config.vocab_size, self.embed_dim)
        self.position_embedding = nn.Embedding(config.max_position_embeddings, self.embed_dim)

        if ctx_init:
            ctx_init = ctx_init.replace("_", " ")
            init_tokens = ctx_init.split()
            self.n_ctx = min(len(init_tokens), n_ctx)
        # Learned context vectors
        ctx_vectors = torch.empty(self.n_ctx, self.embed_dim)
        nn.init.normal_(ctx_vectors, std=0.02)
        self.ctx = nn.Parameter(ctx_vectors)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
    ) -> torch.Tensor:
        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError("Either input_ids or inputs_embeds must be provided.")
            inputs_embeds = self.token_embedding(input_ids)  # (B, L, D)

        B, L, D = inputs_embeds.shape
        # Replace first n_ctx token embeddings (prefix injection)
        prefix_len = min(self.n_ctx, L)
        if prefix_len > 0:
            # Broadcast learned context to batch and overwrite
            inputs_embeds = inputs_embeds.clone()
            inputs_embeds[:, :prefix_len, :] = self.ctx[:prefix_len].unsqueeze(0).expand(B, -1, -1)

        # Position embeddings (standard)
        if position_ids is None:
            position_ids = torch.arange(L, device=inputs_embeds.device).unsqueeze(0)  # (1, L)
        pos_embeds = self.position_embedding(position_ids)
        return inputs_embeds + pos_embeds

# %%
# Inject CoOp trainable prompts into SAM3 text encoder
def inject_coop_into_sam3(model, n_ctx=16, ctx_init=None):
    text_model = model.text_encoder.text_model
    clip_cfg = text_model.config
    coop_embeddings = CLIPTextEmbeddings(clip_cfg, n_ctx=n_ctx, ctx_init=ctx_init)

    # Copy original weights
    coop_embeddings.token_embedding.weight.data.copy_(
        text_model.embeddings.token_embedding.weight.data
    )
    coop_embeddings.position_embedding.weight.data.copy_(
        text_model.embeddings.position_embedding.weight.data
    )

    # Replace
    text_model.embeddings = coop_embeddings.to(model.device)

    # Freeze everything then unfreeze context
    for p in model.parameters():
        p.requires_grad = False
    text_model.embeddings.ctx.requires_grad_(True)

    print(f"Injected CoOp prefix (n_ctx={coop_embeddings.n_ctx}) without changing sequence length.")
    return model

# Inject CoOp into the model
sam3_model = inject_coop_into_sam3(sam3_model, n_ctx=16)

#  %%

BATCH_SIZE = 2
num_epochs=5
lr=1e-4
weight_decay=0.01
text_prompt=""
save_every=1

class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.7, beta=0.5):
        super().__init__()
        self.alpha = alpha  # false positives
        self.beta = beta    # false negatives

    def forward(self, pred, target):
        # pred: (B, H, W) probs in [0,1]
        # target: (B, H, W) binary/float
        pred = pred.float()
        target = target.float()
        tp = (pred * target).sum(dim=(1, 2))
        fp = (pred * (1 - target)).sum(dim=(1, 2))
        fn = ((1 - pred) * target).sum(dim=(1, 2))
        tversky_index = tp / (tp + self.alpha * fp + self.beta * fn + 1e-6)
        return 1 - tversky_index.mean()

forged_folder = os.path.join(config.data_directory, 'train_images', 'forged')
authentic_folder = os.path.join(config.data_directory, 'train_images', 'authentic')
mask_folder = os.path.join(config.data_directory, 'train_masks')

def get_batches(batch_size:int=config.DINOV3_BATCH_SIZE):
    
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


checkpoint_dir = os.path.join('weights')

tversky_loss_fn = TverskyLoss(alpha=0.7, beta=0.5)
    
# Only train parameters that require grad (CoOp context tokens)
ctx_param = sam3_model.text_encoder.text_model.embeddings.ctx
ctx_param.requires_grad_(True)

# Use the Parameter itself (leaf) for optimization
trainable_params = [ctx_param]


if len(trainable_params) == 0:
    raise RuntimeError("No trainable parameters found (context tokens may not be marked requires_grad).")

optimizer = AdamW(trainable_params, lr=lr, weight_decay=weight_decay)
loss_fn = TverskyLoss(alpha=0.7, beta=0.5)

sam3_model.train()
for epoch in range(num_epochs):
    epoch_loss = 0.0
    n_batches = 0

    for img_batch, _, forged_masks, ids in get_batches(batch_size=BATCH_SIZE):
        # Prepare inputs
        inputs = sam3_processor(
            images=img_batch,
            text=[text_prompt] * len(img_batch),
            return_tensors="pt"
        ).to(config.DINOV3_DEVICE)

        # Forward (keep gradients)
        outputs = sam3_model(**inputs, return_dict=True)

        # Extract first mask probs and apply sigmoid
        raw_pm = outputs.pred_masks[:, 0, :, :]          # (B, Hp, Wp)
        pred_masks = torch.sigmoid(raw_pm)               # ensure in [0,1]

        # Resize predicted masks to target image size
        target_size = config.DINOV3_DEFAULT_IMAGE_SIZE
        pred_masks_resized = torch.nn.functional.interpolate(
            pred_masks.unsqueeze(1),                     # (B,1,Hp,Wp)
            size=(target_size, target_size),
            mode='bilinear',
            align_corners=False
        ).squeeze(1)

        gt_tensor = torch.from_numpy(forged_masks).float().to(config.DINOV3_DEVICE)
        gt_tensor_resized = torch.nn.functional.interpolate(
            gt_tensor.unsqueeze(1),
            size=(target_size, target_size),
            mode='nearest'
        ).squeeze(1)
        loss = loss_fn(pred_masks_resized, gt_tensor_resized)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        n_batches += 1

    avg_loss = epoch_loss / max(n_batches, 1)
    print(f"[Epoch {epoch+1}/{num_epochs}] Train Loss: {avg_loss:.4f}")

    # Save checkpoint
    if (epoch + 1) % save_every == 0:
        ckpt_path = os.path.join(checkpoint_dir, f'sam3_FT.pth')
        torch.save({
            "epoch": epoch + 1,
            "model_state_dict": sam3_model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": avg_loss,
        }, ckpt_path)
        print(f"Saved checkpoint to {ckpt_path}")

print("Training complete.")




#  %%
# Load image
image_url = "http://images.cocodataset.org/val2017/000000077595.jpg"
image = Image.open(requests.get(image_url, stream=True).raw).convert("RGB")

#  %%
image.size

#  %%
# Segment using text prompt
inputs = sam3_processor(images=image, text="ear", return_tensors="pt").to(config.DINOV3_DEVICE)

print(list(inputs.keys()))
print(inputs["pixel_values"].shape)

#  %%
with torch.no_grad():
    outputs = sam3_model(**inputs, return_dict=True)

# Post-process results
results = sam3_processor.post_process_instance_segmentation(
    outputs,
    threshold=0.5,
    mask_threshold=0.5,
    target_sizes=inputs.get("original_sizes").tolist()
)[0]

print(f"Found {len(results['masks'])} objects")
# Results contain:
# - masks: Binary masks resized to original image size
# - boxes: Bounding boxes in absolute pixel coordinates (xyxy format)
# - scores: Confidence scores

# %%
outputs.pred_masks[:, 0, :, :].shape


# %%
plt.imshow(image)
plt.title(f'OG')
plt.axis('off')
plt.show()

outputs.pred_masks.squeeze(1).shape


#  %%
results["masks"][1, :, :].detach().cpu().numpy()

# %%
# Training setup



import numpy as np



# %%

fr, aum, frm, ids = next(get_batches())


#  %%

fr[0].size, aum[0].shape, frm[0].shape
