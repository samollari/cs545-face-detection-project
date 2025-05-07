import torch
from torch.utils.data import DataLoader, WeightedRandomSampler, DistributedSampler, ConcatDataset
import torch.optim as optim
from torchvision import transforms
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn.functional as F
import random
import numpy as np
from tqdm import tqdm
import sys
import torch.distributed as dist
import os

from arcface_model import ArcFaceModel
from dataloaders import ZipImageDataset, LFWPairDataset
from util import parse_lfw_pairs_from_zip, lfw_eval_fn

# Initialize distributed environment
def init_distributed_mode():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])  # Get local rank from SLURM
    torch.cuda.set_device(local_rank)  # Set the device to the local GPU
    return local_rank

## Switches and Variable Params
# 1: Real Data
# 2: Unprocessed Synthetic
# 3: Contrast+Histogram Processed Synthetic
# 4: Edgemap Processed Synthetic
training_scenario = 1 
use_distributed = True # Enable this if using multiple gpus
b_size = 128 # batch size
workers = 4 # Number of workers
num_gpu=2 # Number of GPUs

# Hardcoded Params (Do not change)
num_epochs = 15
eval_fn = lfw_eval_fn
num_classes = 0
transform_train = []
scenario_type = ""

match training_scenario:
    case 1:
        num_classes = 10572 
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(112, scale=(0.8, 1.0)),  # Random crop and slight scale jitter
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomRotation(10),  # small rotations
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        scenario_type = "Real.pth"
        
        # Load full training dataset
        full_dataset = ZipImageDataset("./../datasets/archive.zip", root_in_zip="casia-webface",transform=transform_train)
        
        zip_path = "./../datasets/lfw.zip"
        image_dir = "lfw-deepfunneled/lfw-deepfunneled"
        s = 25
        m = 0.5            
    case 2:
        num_classes = 4000
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(112, scale=(0.8, 1.0)),  # Random crop and slight scale jitter
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomRotation(10),  # small rotations
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ]) 
        scenario_type = "Raw_Synthetic.pth" 

        dataset1 = ZipImageDataset("./../datasets/raw_1.zip",transform=transform_train)
        dataset2 = ZipImageDataset("./../datasets/raw_2.zip",transform=transform_train)
        
        full_dataset = ConcatDataset([dataset1, dataset2]) 
        
        zip_path = "./../datasets/lfw.zip"
        image_dir = "lfw-deepfunneled/lfw-deepfunneled"
        s = 18
        m = 0.2                   
    case 3:
        num_classes = 4000
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(112, scale=(0.8, 1.0)),  # Random crop and slight scale jitter
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomRotation(10),  # small rotations
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        scenario_type = "RGB_Synthetic.pth"
        
        # Load full training dataset
        dataset1 = ZipImageDataset("./../datasets/processed.zip", root_in_zip="processed",transform=transform_train)
        dataset2 = ZipImageDataset("./../datasets/2000-3999-processed.zip", root_in_zip="processed",transform=transform_train)

        full_dataset = ConcatDataset([dataset1, dataset2])
        
        zip_path = "./../datasets/lfw.zip"
        image_dir = "lfw-deepfunneled/lfw-deepfunneled" 
        s = 18
        m = 0.2 
    case 4:
        num_classes = 4000
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(112, scale=(0.95, 1.0), ratio=(0.95, 1.05)),
            transforms.RandomAffine(degrees=8, translate=(0.02, 0.02), scale=(0.97, 1.03), shear=3),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.3)),
            transforms.ToTensor()
        ])
        scenario_type = "Edge_Synthetic.pth"
        
        # Load full training dataset
        dataset1 = ZipImageDataset("./../datasets/edges.zip", root_in_zip="edges",transform=transform_train)
        dataset2 = ZipImageDataset("./../datasets/2000-3999-edges.zip", root_in_zip="edges",transform=transform_train)

        full_dataset = ConcatDataset([dataset1, dataset2])
        
        zip_path = "./../datasets/lfw_edges.zip" 
        image_dir = "lfw_edges" 
        s = 18
        m = 0.2       
    case _:
        sys.exit("Invalid case number")
        
save_path = "./../trained_models/"+scenario_type

# Set random seed for reproducibility
random.seed(42)

# Set up distributed training
if use_distributed:
    local_rank = init_distributed_mode()
    device = torch.device(f"cuda:{local_rank}")
else: device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

targets = []
if not (training_scenario == 1):
    for dataset in full_dataset.datasets:
        targets.extend([sample[1] for sample in dataset.samples])
else:
    targets.extend([sample[1] for sample in full_dataset.samples])
    
print(f"Number of targets found: {len(targets)}")
print(f"Number of classes found: {len(np.unique(targets))}")

# Step 2: Create sampler for distributed training
if use_distributed:
    train_sampler = DistributedSampler(full_dataset, num_replicas=dist.get_world_size(), rank=dist.get_rank())
    
    # Step 3: Create DataLoader with DistributedSampler
    train_loader = DataLoader(
        full_dataset,
        batch_size=b_size,
        sampler=train_sampler,  # Use DistributedSampler
        num_workers=workers,
        pin_memory=True
    )
else: 
    # Step 3: Create DataLoader with DistributedSampler
    train_loader = DataLoader(
        full_dataset,
        batch_size=b_size,
        num_workers=workers,
        pin_memory=True
    )    

# Get Evaluation Data
csv_name = "pairs.csv"
lfw_pairs = parse_lfw_pairs_from_zip(zip_path, csv_name, image_dir)

# Define your image preprocessing
if training_scenario == 4:
    transform = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor()
    ])
else:
    transform = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])    

# Initialize dataset
lfw_dataset = LFWPairDataset(lfw_pairs, zip_path=zip_path, transform=transform)
lfw_loader = DataLoader(lfw_dataset, batch_size=b_size, shuffle=False, num_workers=workers)

# Initialize model, optimizer
model = ArcFaceModel(num_classes).to(device)
model.arc_margin.s = s
model.arc_margin.m = m
if use_distributed:
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])  # Wrap model in DDP

optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
scheduler = CosineAnnealingLR(optimizer, T_max=25, eta_min=1e-5)  # Suppose 25 epochs

# Initialize scaler for mixed precision
scaler = torch.cuda.amp.GradScaler(init_scale=2.**10, growth_interval=2000)

# Training and evaluation loop
model.train()

best_acc = 0.0

for epoch in range(num_epochs):
    if use_distributed:
        train_sampler.set_epoch(epoch)  # Set epoch for sampler to shuffle data correctly
    
    model.train()

    total_loss = 0.0
    total_samples = 0
    if use_distributed:
        is_main_process = (dist.get_rank() == 0)
    else: is_main_process = True

    pbar = tqdm(train_loader, desc=f"Epoch {epoch}", dynamic_ncols=False, file=sys.stdout, mininterval=102) if is_main_process else train_loader
    
    i = 0
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        # Enable mixed precision for forward pass
        with torch.cuda.amp.autocast():
            outputs = model(images, labels)  # model returns ArcFace logits
            loss = F.cross_entropy(outputs, labels)

        # Backward pass with scaling
        scaler.scale(loss).backward()        
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * images.size(0)
        total_samples += images.size(0)
        if is_main_process and (i%100 == 0):
            pbar.set_postfix(loss=loss.item())
        i += 1
      
  
    if is_main_process:
        avg_loss = total_loss / total_samples
        print(f"Epoch {epoch} Average Loss: {avg_loss:.4f}")
    
    scheduler.step()
    
    # Optional evaluation
    if eval_fn is not None:
        val_acc = eval_fn(model, device, lfw_loader, world_size=num_gpu, use_dist=use_distributed, rank=dist.get_rank())

        # Save best model from rank 0
        if is_main_process and val_acc > best_acc:
            best_acc = val_acc
            if save_path:
                torch.save(model.state_dict(), save_path)
                print(f"Saved best model to {save_path}")
        if is_main_process:
            print(f"Epoch {epoch} Validation Accuracy: {val_acc:.2f}%")

print(f"Best Validation Accuracy: {best_acc:.2f}%")
