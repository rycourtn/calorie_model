# Updated training script for Nutrition5K with GroupNorm + Checkpointing
import os
import gc
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.model_selection import KFold
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision.transforms as transforms
import timm
from tqdm import tqdm
from safetensors.torch import load_file
import timm.models._hub as timm_hub
import torch.utils.checkpoint as cp

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32"
timm_hub._has_hf_hub = True
gc.collect()
torch.cuda.empty_cache()
torch.cuda.ipc_collect()

def is_blurry(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return True
    return cv2.Laplacian(img, cv2.CV_64F).var() < 100

class Nutrition5KDataset(Dataset):
    def __init__(self, csv_path, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, hue=0.1),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        raw_data = pd.read_csv(csv_path)
        valid_rows = []
        for _, row in raw_data.iterrows():
            image_path = os.path.join(self.image_dir, 'realsense_overhead', row['id'], 'rgb.png')
            if os.path.exists(image_path) and not is_blurry(image_path):
                valid_rows.append(row)

        self.data = pd.DataFrame(valid_rows)
        print(f"✅ Found {len(self.data)} clean samples with usable images.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image_path = os.path.join(self.image_dir, 'realsense_overhead', row['id'], 'rgb.png')
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)

        label = torch.tensor([
            row['calories'],
            row['protein_g'],
            row['fat_g'],
            row['carbs_g']
        ], dtype=torch.float32)
        return image, label

class MultiTaskLoss(nn.Module):
    def __init__(self, num_tasks=4):
        super().__init__()
        self.log_vars = nn.Parameter(torch.zeros(num_tasks))

    def forward(self, preds, targets):
        per_task_mae = torch.abs(preds - targets).mean(dim=0)
        total_loss = 0
        for i in range(targets.shape[1]):
            precision = torch.exp(-self.log_vars[i])
            loss = precision * per_task_mae[i] + self.log_vars[i]
            total_loss += loss
        return total_loss, per_task_mae

# Updated MultiEffSwin model: EfficientNet -> FPN -> Swin Transformer -> Head
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import torch.utils.checkpoint as cp
from safetensors.torch import load_file


def convert_batchnorm_to_groupnorm(module):
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            setattr(module, name, nn.GroupNorm(num_groups=32, num_channels=child.num_features))
        else:
            convert_batchnorm_to_groupnorm(child)


class MultiEffSwin(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device

        # EfficientNet as feature extractor
        self.eff = timm.create_model('tf_efficientnetv2_xl.in21k', pretrained=False, features_only=True).to(device)
        eff_weights = load_file('/notebooks/eff_apr_21.safetensors')
        self.eff.load_state_dict(eff_weights, strict=False)
        convert_batchnorm_to_groupnorm(self.eff)

        # FPN to fuse EfficientNet features
        self.fpn = nn.ModuleList([
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.Conv2d(96, 64, kernel_size=3, padding=1),
            nn.Conv2d(256, 64, kernel_size=3, padding=1)
        ]).to(device)

        # Project FPN features to uniform channels for Swin input
        self.proj = nn.ModuleList([
            nn.Conv2d(64, 3, 1) for _ in range(4)
        ]).to(device)

        # Swin Transformer per FPN feature level
        def load_swin():
            swin = timm.create_model('swin_large_patch4_window7_224.ms_in22k_ft_in1k', pretrained=False, features_only=True).to(device)
            weights = load_file('/notebooks/swin_apr_21.safetensors')
            swin.load_state_dict(weights, strict=False)
            convert_batchnorm_to_groupnorm(swin)
            return swin

        self.swin = nn.ModuleList([load_swin() for _ in range(4)])
        self.swin_proj = nn.ModuleList([
            nn.Conv2d(1536, 64, 1) for _ in range(4)
        ]).to(device)

        # Final head
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 4)
        ).to(device)

    def forward(self, x):
        eff_feats = cp.checkpoint(lambda y: self.eff(y), x)
        fpn_feats = []
        for i in range(4):
            f = self.fpn[i](eff_feats[i])
            up = F.interpolate(f, size=(224, 224), mode='bilinear', align_corners=False)
            fpn_feats.append(up)

        swin_feats = []
        for i in range(4):
            swin_in = self.proj[i](fpn_feats[i])
            swin_out = cp.checkpoint(lambda y: self.swin[i](y)[-1], swin_in).permute(0, 3, 1, 2)
            fused = self.swin_proj[i](swin_out)
            swin_feats.append(F.interpolate(fused, size=(224, 224), mode='bilinear', align_corners=False))

        final_feat = torch.cat(swin_feats, dim=1)  # shape: [B, 256, 224, 224]
        return self.head(final_feat)

def train_model_cv(csv_path, image_dir, device, k_folds=5):
    dataset = Nutrition5KDataset(csv_path, image_dir)
    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    results = []

    for fold, (train_ids, val_ids) in enumerate(kfold.split(dataset)):
        print(f"\n--- Fold {fold+1}/{k_folds} ---")
        train_loader = DataLoader(Subset(dataset, train_ids), batch_size=1, shuffle=True)
        val_loader = DataLoader(Subset(dataset, val_ids), batch_size=1)

        model = MultiEffSwin(device).to(device)
        criterion = MultiTaskLoss().to(device)
        optimizer = torch.optim.Adam(list(model.parameters()) + list(criterion.parameters()), lr=1e-3, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
        best_val_loss = float('inf')
        accum_steps = 32

        for epoch in range(100):
            model.train()
            total_loss = 0
            optimizer.zero_grad()

            for i, (images, targets) in enumerate(tqdm(train_loader, desc=f"Fold {fold+1} Epoch {epoch+1} [Train]")):
                images, targets = images.to(device), targets.to(device)
                outputs = model(images)
                loss, _ = criterion(outputs, targets)
                (loss / accum_steps).backward()

                if (i + 1) % accum_steps == 0 or (i + 1) == len(train_loader):
                    optimizer.step()
                    optimizer.zero_grad()

                total_loss += loss.item()

            scheduler.step()

            model.eval()
            val_loss, all_preds, all_targets = 0, [], []
            with torch.no_grad():
                for images, targets in tqdm(val_loader, desc=f"Fold {fold+1} Epoch {epoch+1} [Val]"):
                    images, targets = images.to(device), targets.to(device)
                    outputs = model(images)
                    loss, _ = criterion(outputs, targets)
                    val_loss += loss.item()
                    all_preds.append(outputs.cpu())
                    all_targets.append(targets.cpu())

            avg_val_loss = val_loss / len(val_loader)
            preds = torch.cat(all_preds, dim=0)
            targets = torch.cat(all_targets, dim=0)
            mae = torch.abs(preds - targets).mean(dim=0)
            safe_targets = targets.clone()
            safe_targets[safe_targets.abs() < 1.0] = 1.0
            mape = (torch.abs(preds - targets) / safe_targets).mean(dim=0)

            mae_np = mae.numpy()
            mape_np = (mape * 100).numpy()

            print(f"Fold {fold+1} Epoch {epoch+1} | Val Loss: {avg_val_loss:.2f} | MAE: {mae_np} | MAPE: {mape_np}")
            results.append((avg_val_loss, mae_np, mape_np))

            with open("/notebooks/results.txt", "a") as f:
                f.write(f"Fold {fold+1} Epoch {epoch+1}:\n")
                f.write(f"  Val Loss: {avg_val_loss:.4f}\n")
                f.write(f"  MAE:      {mae_np}\n")
                f.write(f"  MAPE:     {mape_np}\n\n")

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model, f"/notebooks/best_model_fold{fold+1}.pth")
                print("✅ Best model saved for fold", fold+1)

    return results

if __name__ == '__main__':
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = train_model_cv("combined.csv", "imagery", device)