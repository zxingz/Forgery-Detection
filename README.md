# RecodAI: Scientific Image Forgery Detection (Recod.ai / LUC Challenge)

This repository contains an end‑to‑end experimental pipeline for detecting and segmenting copy–move forgeries in biomedical research images for the Kaggle competition "Recod.ai/LUC – Scientific Image Forgery Detection". The project focuses on two complementary approaches:

1. Patch‑level segmentation using a frozen DINOv3 Vision Transformer backbone with a lightweight trainable segmentation head (Tversky loss).
2. Prompt‑tuned SAM3 (Segment Anything Model v3) via CoOp‑style learnable context tokens injected into the text encoder for forgery region localization.

The current state represents a research prototype; metrics are preliminary (mAP 0.314 across IoU thresholds 0.30–0.95) and several improvement avenues are outlined below.

---
## 1. Competition Overview (Paraphrased)
The challenge asks participants to automatically identify and segment regions in scientific/biomedical images that have been manipulated through copy–move forgeries. Rather than simple classification, the task demands spatial localization (pixel masks) of duplicated or transplanted structures. High‑quality detection helps safeguard research integrity by flagging suspicious image alterations. The organizer (Recod.AI) provides a dataset of paired authentic and forged images plus corresponding ground‑truth masks for training. A custom evaluation metric centered on averaged precision over multiple IoU thresholds determines leaderboard ranking. Submissions require robust generalization: images vary in modality, texture density, and scale of manipulated entities.

Key elements:
- Objective: Segment forged (copy–moved) regions in biomedical images.
- Data: Authentic image, forged counterpart, and forgery mask for training; test set without masks for prediction.
- Metric: Mean Average Precision (mAP) computed across a wide IoU threshold span (0.30–0.95).
- Difficulty: Small/highly similar structures, variable contrast, potential class imbalance (small forged area vs large background), and need for precise localization.

---
## 2. Repository Layout
Top‑level files:
- `config.py` central configuration (paths, model constants).
- `DinoV3_Train.py` patch‑level segmentation training (ViT backbone freeze + segmentation head).
- `DinoV3_Train_Larger.py` variant with deeper / regularized head (extra conv + dropout).
- `SAM3_Coop_train.py` prompt‑tuning of SAM3 using learnable context (CoOp) for text guidance.
- `DinoV3_Infer.ipynb` inference & analysis notebook (sample visualization, object extraction, metric computation).
- `masker.py` (entity/mask utilities; not expanded here).
- `mask_entities/` serialized per‑image entity masks (`*.pkl`) produced during preprocessing.
- `weights/` and `weights_backup/` pretrained DINOv3 weights and fine‑tuned checkpoints. URL[https://drive.google.com/drive/folders/1oinGdSZY97qKhNXp04ywMw4jkUrdGsta?usp=drive_link]
- `repos/` embedded source copies of `dinov3`, `sam3`, and `transformers` (local hub loading & potential modifications).

Dataset (expected external path defined in `Config`):
```
train_images/
	forged/      # Forged variant PNGs
	authentic/   # Authentic originals
train_masks/   # Ground‑truth forgery masks
```

Generated artifacts:
- `mask_entities/` per entity dicts with `auth_mask` and `forged_mask` (pixel arrays) used to combine multiple forgery components.
- Model checkpoints: `dinov3_*_dinov3_S_FT*.pth`, `sam3.pth`, etc.

---
## 3. Preprocessing & Patch Mask Construction
To align image resolution with transformer patch grids:
1. Images resized to a square whose side is a multiple of patch size (default DINOv3: 16; image size 512 or 1024 depending script).
2. Pixel masks thresholded (`MASK_THRESHOLD = 0`) then converted to boolean arrays.
3. Multiple entity masks aggregated via logical OR, then converted to patch‑level float masks where each patch stores the fraction of forged pixels (`pixel_mask_to_patch_float`).
4. Flood‑fill utility (`count_components_floodfill`) counts connected components for post‑hoc analysis (e.g., small artifact filtering, future weighting schemes).

This yields dense patch probability targets enabling segmentation head learning independent of feature extractor fine‑tuning.

---
## 4. DINOv3 Segmentation Architecture
### Backbone
Frozen DINOv3 variants (default small `dinov3_vits16`, alternative `dinov3_vitb16`, `dinov3_vith16plus`) loaded locally via `torch.hub.load` pointing to `repos/dinov3` with raw weight files.

### Feature Extraction
`get_intermediate_layers(..., reshape=True, norm=True)` returns a stack of layer feature maps. Last map `[B, C, H, W]` drives segmentation.

### Segmentation Head (Two Variants)
1. Basic (`DinoV3_Train.py`): `Conv2d -> BN -> ReLU -> Conv2d -> Sigmoid` producing a probability mask.
2. Larger (`DinoV3_Train_Larger.py`): Adds an extra conv + BN and spatial dropout for regularization.

### Loss Function
Tversky Loss (alpha=0.7, beta=0.5) balances false positives vs false negatives, suitable for small target regions. Dice Loss is its symmetric special case (alpha=beta=0.5).

### Optimization & Early Stopping
- Optimizer: `AdamW` on segmentation head parameters only (backbone frozen) LR=1e‑4, weight decay=0.05.
- Validation: First produced batch reserved.
- Early stopping: Patience of 5 epochs monitoring validation Tversky loss.
- Checkpointing: Saves full state dict + optimizer + metrics each epoch; best model separately.

### Mixed Precision
`torch.amp.autocast_mode.autocast` used for efficiency (device‑type conditional).

---
## 5. SAM3 Prompt Tuning (CoOp Injection)
Instead of training the entire SAM3 model, we inject learnable context tokens into the CLIP text encoder:
1. Replace `text_model.embeddings` with a custom `CLIPTextEmbeddings` wrapper holding `n_ctx` learned vectors.
2. Freeze all original weights; only context parameters (`ctx`) are optimized.
3. Use a descriptive text prompt: "detect forged regions in the image" repeated per batch.
4. Forward pass yields `pred_masks` (select first predicted mask channel) which is resized to the canonical image size and supervised with Tversky loss against forged masks.

This approach drastically reduces trainable parameters and enables rapid experimentation with textual guidance. It is analogous to parameter‑efficient prefix tuning.

---
## 6. Inference & Analysis (Notebook Highlights)
`DinoV3_Infer.ipynb` demonstrates:
- Loading fine‑tuned checkpoint: `dinov3_vits16_dinov3_S_FT.pth`.
- Generating predicted probability masks and binarizing to highlight suspected forged entities.
- Visual overlays: bounding boxes around candidate original vs found duplicates (SIFT feature matching demonstration) and mask visualization.
- Component extraction counts (e.g., "Extracted 3 objects ... Found 2 objects using SIFT").
- Example predicted masks for sample IDs (e.g., `10030`).
- Evaluation: Computed mAP = 0.0314. Per IoU threshold AP values:
	- 0.30–0.75: 0.10
	- 0.80–0.95: 0.00

Interpretation: The model currently achieves moderate detection at lower IoU thresholds but struggles with precise boundary alignment at higher thresholds, indicating a need for improved mask refinement (e.g., multi‑scale decoding, CRF post‑processing, or training with boundary‑aware losses).

---
## 7. Setup & Environment (Windows / Conda Suggested)
Prerequisites: Python 3.10+, CUDA GPU recommended for training.

Example environment creation (adjust paths):
```cmd
conda create -n recodai python=3.10 -y
conda activate recodai
pip install --upgrade pip
rem Install core libraries
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install numpy scipy pandas pillow tqdm joblib scikit-image opencv-python matplotlib
rem Install transformers (for SAM3) and any local editable repos if needed
pip install transformers
pip install git+https://github.com/facebookresearch/dinov3.git  --no-deps
pip install git+https://github.com/facebookresearch/segment-anything-3.git --no-deps
```

Place required pretrained weight files inside `weights/` (already present for DINOv3). Ensure `Config` points `data_directory` to the competition dataset root.

---
## 8. Reproducible Training
### DINOv3 Head Training
```cmd
conda activate recodai
python DinoV3_Train.py
```
Adjust inside script:
- `BATCH_SIZE`, image size, model variant (`DINOV3_MODEL_NAME` in `config.py`).

### Larger Head Variant
```cmd
python DinoV3_Train_Larger.py
```

### SAM3 Prompt Tuning
```cmd
python SAM3_Coop_train.py
```
Tune parameters: `n_ctx`, `num_epochs`, `lr`, and prompt text.

---
## 9. Inference Workflow
1. Generate/refresh `mask_entities/` (entity extraction script or notebook cell – ensure pickles exist).
2. Run `DinoV3_Infer.ipynb` cells sequentially to:
	 - Load model & weights.
	 - Produce segmentation predictions.
	 - Visualize forged vs authentic image pairs.
	 - Compute mAP across IoU thresholds.
3. Export masks for submission formatting (future utility TBD).

Potential CLI (future):
```cmd
python -m run.infer --images <path> --checkpoint weights/dinov3_vits16_dinov3_best.pth --output ./pred_masks
```

---
## 10. Evaluation Metric Notes
The custom mAP spans IoU thresholds from 0.30 to 0.95 at granularity 0.05. Low AP at high IoUs indicates coarse boundaries; improvements could include:
- Focal Tversky / Boundary loss hybrid.
- Feature fusion of multiple intermediate layers (multi‑scale context).
- Post‑processing: morphological closing, conditional random fields, or diffusion‑based refinement.
- Semi‑supervised augmentation via consistency between SAM3 and DINOv3 predictions.

---
## 11. Current Results Snapshot
| Metric | Value |
|--------|-------|
| mAP (0.30–0.95) | 0.314 |
| AP (0.30–0.75)  | 0.10   |
| AP (0.80–0.95)  | 0.00   |

Sample qualitative findings:
- Model identifies major forged corn ear duplicates in example image but produces blocky patches.
- SIFT feature matching corroborates duplicate regions (2–3 objects found vs predicted placement).
- Missed fine boundary details and minor artifacts.

---
## 12. Roadmap / Future Work
Short term:
- Integrate SAM3 mask proposals to refine DINOv3 patch predictions (ensemble).
- Add boundary‑aware loss and test Dice vs Focal Tversky variations.
- Improve preprocessing: adaptive image resizing preserving aspect ratio + padding.
Medium term:
- Light fine‑tuning of selective DINOv3 layers (e.g., last 2 blocks) with low LR.
- Add test‑time augmentation (horizontal flips, slight scaling) for mask averaging.
- Implement prediction confidence filtering for submission packaging.
Long term:
- Multi‑task learning: joint detection + segmentation (object proposals + masks).
- Explore diffusion or transformer decoder heads for sharper masks.
- Deploy lightweight ONNX or TorchScript model for faster batch inference.

---
## 13. Known Limitations
- Preliminary mAP; not yet competitive with top leaderboard entries.
- No robust post‑processing of masks (pure sigmoid threshold currently).
- Limited hyperparameter sweep; learning rate and batch size not optimized.
- Potential overfitting risks due to small number of trainable parameters vs complex distribution shift.

---
## 14. Citation & Attribution
If using underlying models, please cite original DINOv3 and SAM papers and follow license files in `repos/dinov3/` and `repos/sam3/`. Competition assets belong to Recod.AI / Kaggle. This README paraphrases public competition goals without reproducing proprietary text verbatim.

---
## 15. Getting Help
Open an issue describing:
- Environment (Python version, GPU model).
- Exact command run.
- Stack trace (if any).
- Sample image ID producing unexpected mask.

---
## 16. Disclaimer
This codebase is a research prototype; performance numbers are subject to change with further experimentation. Use results cautiously for any production or integrity decision workflows.

---
## 17. Quick Start Summary
```cmd
rem 1. Create environment (see section 7)
rem 2. Configure dataset path in Config(data_directory=...)
rem 3. Train segmentation head
python DinoV3_Train.py
rem 4. (Optional) Prompt-tune SAM3
python SAM3_Coop_train.py
rem 5. Run notebook DinoV3_Infer.ipynb for evaluation & visualization
```

---
## 18. Acknowledgments
Thanks to the Recod.AI community, Kaggle participants, and open‑source contributors of DINOv3, SAM, and Transformers projects for enabling rapid experimentation.

