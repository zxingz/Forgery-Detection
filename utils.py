import os



data_directory = r"D:\RecodAI\recodai-luc-scientific-image-forgery-detection"
current_dir = os.path.dirname(os.path.abspath(__file__))

dinov3_repo_dir = os.path.join(current_dir, "repos", "dinov3")
dinov3_vitb16_weight_raw = os.path.join(current_dir, "weights", "dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth")
dinov3_vith16_weight_raw = os.path.join(current_dir, "weights", "dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth")
dinov3_vits16_weight_raw = os.path.join(current_dir, "weights", "dinov3_vits16_pretrain_lvd1689m-08c60483.pth")
dinov3_vit7B16_weight_raw = os.path.join(current_dir, "weights", "dinov3_vit7b16_pretrain_lvd1689m-a955f4ea.pth")