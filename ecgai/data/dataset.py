"""
Dataset otimizado para sinais de ECG.
"""
from pathlib import Path
from typing import Tuple, List, Optional
import os

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler

from ecgai.config import Config


class ECGDataset(Dataset):
    """Dataset para carregar sinais de ECG de arquivos .npz"""
    
    def __init__(
        self, 
        data_folder: str,
        use_mmap: bool = True,
        use_fp16_io: bool = False,
        transform: Optional[callable] = None
    ):
        self.data_folder = Path(data_folder)
        self.use_mmap = use_mmap
        self.use_fp16_io = use_fp16_io
        self.transform = transform
        
        self.files = sorted([
            f for f in self.data_folder.glob("*.npz")
        ])
        
        if len(self.files) == 0:
            raise ValueError(f"Nenhum arquivo .npz encontrado em {data_folder}")
    
    def __len__(self) -> int:
        return len(self.files)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        file_path = self.files[idx]
        
        mmap_mode = "r" if self.use_mmap else None
        data = np.load(file_path, mmap_mode=mmap_mode)
        
        dtype = np.float16 if self.use_fp16_io else np.float32
        signal = np.asarray(data["signal"], dtype=dtype)
        label = np.asarray(data["label"], dtype=np.float32)
        
        signal = torch.from_numpy(signal)
        label = torch.from_numpy(label)
        
        if self.transform is not None:
            signal = self.transform(signal)
        
        return signal, label


class DataAugmentation:
    """Transformações de data augmentation para sinais de ECG"""
    
    def __init__(
        self, 
        noise_std: float = 0.01,
        scale_range: Tuple[float, float] = (0.9, 1.1),
        noise_prob: float = 0.5,
        scale_prob: float = 0.5
    ):
        self.noise_std = noise_std
        self.scale_range = scale_range
        self.noise_prob = noise_prob
        self.scale_prob = scale_prob
    
    def __call__(self, signal: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() < self.noise_prob:
            noise = torch.randn_like(signal) * self.noise_std
            signal = signal + noise
        
        if torch.rand(1).item() < self.scale_prob:
            scale = torch.empty(1).uniform_(*self.scale_range).item()
            signal = signal * scale
        
        return signal


def create_dataloaders(
    config: Config,
    rank: int = 0,
    world_size: int = 1
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    data_dir = Path(config.data.data_dir)
    num_workers = config.dataloader.get_num_workers(world_size)
    
    train_transform = DataAugmentation() if world_size > 0 else None
    train_dataset = ECGDataset(
        data_dir / "train",
        use_mmap=config.data.use_mmap,
        use_fp16_io=config.data.use_fp16_io,
        transform=train_transform
    )
    
    val_dataset = ECGDataset(
        data_dir / "val",
        use_mmap=config.data.use_mmap,
        use_fp16_io=config.data.use_fp16_io
    )
    
    test_dataset = ECGDataset(
        data_dir / "test",
        use_mmap=config.data.use_mmap,
        use_fp16_io=config.data.use_fp16_io
    )
    
    train_sampler = DistributedSampler(
        train_dataset, 
        num_replicas=world_size, 
        rank=rank, 
        shuffle=True
    ) if world_size > 1 else None
    
    val_sampler = DistributedSampler(
        val_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False
    ) if world_size > 1 else None
    
    test_sampler = DistributedSampler(
        test_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False
    ) if world_size > 1 else None
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=num_workers,
        pin_memory=config.dataloader.pin_memory,
        persistent_workers=config.dataloader.persistent_workers and num_workers > 0,
        prefetch_factor=config.dataloader.prefetch_factor if num_workers > 0 else None
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        sampler=val_sampler,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=config.dataloader.pin_memory,
        persistent_workers=config.dataloader.persistent_workers and num_workers > 0,
        prefetch_factor=config.dataloader.prefetch_factor if num_workers > 0 else None
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.training.batch_size,
        sampler=test_sampler,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=config.dataloader.pin_memory,
        persistent_workers=config.dataloader.persistent_workers and num_workers > 0,
        prefetch_factor=config.dataloader.prefetch_factor if num_workers > 0 else None
    )
    
    return train_loader, val_loader, test_loader


def compute_class_weights(
    data_dir: str,
    labels: List[str],
    splits: List[str] = ["train", "val"]
) -> torch.Tensor:
    data_path = Path(data_dir)
    all_labels = []
    
    for split in splits:
        split_path = data_path / split
        if not split_path.exists():
            continue
            
        for npz_file in split_path.glob("*.npz"):
            data = np.load(npz_file)
            all_labels.append(data["label"])
    
    y_all = np.vstack(all_labels)
    freq = np.mean(y_all, axis=0)
    weights = 1.0 / (freq + 1e-6)
    weights = weights / weights.sum() * len(labels)
    
    return torch.tensor(weights, dtype=torch.float32)
