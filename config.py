import os

import torch

class Config:
        
    def __init__(self, data_directory, code_directory=None):
        
        if code_directory is None:
            self.code_directory = os.path.dirname(os.path.abspath(__file__))
        else:
            self.code_directory = code_directory

        self.data_directory = data_directory

        self.dinov3_repo_dir = os.path.join(self.code_directory, "repos", "dinov3")
        self.transformers_repo_dir = os.path.join(self.code_directory, "repos", "transformers")
        self.sam3_repo_dir = os.path.join(self.code_directory, "repos", "sam3")

        self.dinov3_vitb16_weight_raw = os.path.join(self.code_directory, "weights", "dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth")
        self.dinov3_vith16_weight_raw = os.path.join(self.code_directory, "weights", "dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth")
        self.dinov3_vits16_weight_raw = os.path.join(self.code_directory, "weights", "dinov3_vits16_pretrain_lvd1689m-08c60483.pth")
        self.dinov3_vit7B16_weight_raw = os.path.join(self.code_directory, "weights", "dinov3_vit7b16_pretrain_lvd1689m-a955f4ea.pth")
        
        # DINOV3 Image Setting
        self.DINOV3_PATCH_SIZE = 16
        self.DINOV3_DEFAULT_IMAGE_SIZE = 512 # Should be multiple of PATCH_SIZE
        self.DINOV3_MASK_THRESHOLD = 0

        # DINOV3 Batch Settings
        self.DINOV3_BATCH_SIZE = 8
        self.DINOV3_IMAGENET_MEAN = (0.485, 0.456, 0.406)
        self.DINOV3_IMAGENET_STD = (0.229, 0.224, 0.225)

        # DINOV3 Device
        self.DINOV3_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Available DINOv3 models:
        MODEL_DINOV3_VITS = "dinov3_vits16"
        MODEL_DINOV3_VITSP = "dinov3_vits16plus"
        MODEL_DINOV3_VITB = "dinov3_vitb16"
        MODEL_DINOV3_VITL = "dinov3_vitl16"
        MODEL_DINOV3_VITHP = "dinov3_vith16plus"
        MODEL_DINOV3_VIT7B = "dinov3_vit7b16"

        DINOV3_MODEL_TO_NUM_LAYERS = {
            MODEL_DINOV3_VITS: 12,
            MODEL_DINOV3_VITSP: 12,
            MODEL_DINOV3_VITB: 12,
            MODEL_DINOV3_VITL: 24,
            MODEL_DINOV3_VITHP: 32,
            MODEL_DINOV3_VIT7B: 40,
        }

        DINOV3_MODEL_TO_EMBED_DIM = {
            MODEL_DINOV3_VITS: 384,
            MODEL_DINOV3_VITSP: 384,  # ViT-Small+
            MODEL_DINOV3_VITB: 768,
            MODEL_DINOV3_VITL: 1024,
            MODEL_DINOV3_VITHP: 1536, # ViT-Huge+
            MODEL_DINOV3_VIT7B: 4096,
        }

        DINOV3_MODEL_TO_WEIGHT_FILE = {
            MODEL_DINOV3_VITS: self.dinov3_vits16_weight_raw,
            MODEL_DINOV3_VITB: self.dinov3_vitb16_weight_raw,
            MODEL_DINOV3_VITHP: self.dinov3_vith16_weight_raw, # ViT-Huge+
        }

        # Model
        self.DINOV3_MODEL_NAME = MODEL_DINOV3_VITS # MODEL_DINOV3_VITB # MODEL_DINOV3_VITHP # MODEL_DINOV3_VITS
        self.DINOV3_N_LAYERS = DINOV3_MODEL_TO_NUM_LAYERS[self.DINOV3_MODEL_NAME]
        self.DINOV3_EMBED_DIM = DINOV3_MODEL_TO_EMBED_DIM[self.DINOV3_MODEL_NAME]
        self.DINOV3_WEIGHT_FILE = DINOV3_MODEL_TO_WEIGHT_FILE[self.DINOV3_MODEL_NAME]

