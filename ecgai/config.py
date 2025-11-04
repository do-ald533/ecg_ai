"""
Configurações centralizadas para o projeto ECG.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional
import yaml
import os


@dataclass
class DataConfig:
    """Configurações de dados."""
    data_dir: str = "processed_npz"
    labels: List[str] = field(default_factory=lambda: [
        "1dAVb", "RBBB", "LBBB", "SB", "AF", "ST", "normal_ecg"
    ])
    num_leads: int = 12
    signal_length: int = 4096
    use_mmap: bool = True
    use_fp16_io: bool = True


@dataclass
class TrainingConfig:
    """Configurações de treinamento."""
    batch_size: int = 32
    epochs: int = 50
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    patience: int = 7
    eval_every: int = 5
    gradient_clip: Optional[float] = None
    warmup_epochs: int = 0
    use_amp: bool = False
    compile_model: bool = False
    
    
@dataclass
class DataLoaderConfig:
    """Configurações do DataLoader."""
    num_workers: Optional[int] = None
    pin_memory: bool = False
    prefetch_factor: int = 2
    persistent_workers: bool = True
    
    def get_num_workers(self, world_size: int = 1) -> int:
        if self.num_workers is not None:
            return self.num_workers
        return max(1, os.cpu_count() // (2 * world_size))


@dataclass
class DistributedConfig:
    """Configurações de treinamento distribuído."""
    backend: str = "gloo"
    timeout_hours: float = 4.0
    find_unused_parameters: bool = False


@dataclass
class ModelConfig:
    """Configurações do modelo."""
    dropout_rate: float = 0.3
    

@dataclass
class CheckpointConfig:
    """Configurações de checkpoints."""
    save_dir: str = "checkpoints"
    best_model_name: str = "best_model.pt"
    last_model_name: str = "last_model.pt"
    weights_name: str = "class_weights.pt"
    metrics_csv: str = "training_metrics.csv"
    save_last: bool = True


@dataclass
class LoggingConfig:
    """Configurações de logging."""
    log_level: str = "INFO"
    log_file: Optional[str] = "train.log"
    use_tensorboard: bool = False
    tensorboard_dir: str = "runs"


@dataclass
class Config:
    """Configuração principal que agrega todas as sub-configurações."""
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    dataloader: DataLoaderConfig = field(default_factory=DataLoaderConfig)
    distributed: DistributedConfig = field(default_factory=DistributedConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> "Config":
        """Carrega configuração de arquivo YAML."""
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        return cls(
            data=DataConfig(**config_dict.get('data', {})),
            training=TrainingConfig(**config_dict.get('training', {})),
            dataloader=DataLoaderConfig(**config_dict.get('dataloader', {})),
            distributed=DistributedConfig(**config_dict.get('distributed', {})),
            model=ModelConfig(**config_dict.get('model', {})),
            checkpoint=CheckpointConfig(**config_dict.get('checkpoint', {})),
            logging=LoggingConfig(**config_dict.get('logging', {}))
        )
    
    def to_yaml(self, yaml_path: str) -> None:
        """Salva configuração em arquivo YAML."""
        config_dict = {
            'data': self.data.__dict__,
            'training': self.training.__dict__,
            'dataloader': self.dataloader.__dict__,
            'distributed': self.distributed.__dict__,
            'model': self.model.__dict__,
            'checkpoint': self.checkpoint.__dict__,
            'logging': self.logging.__dict__
        }
        with open(yaml_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)
    
    @classmethod
    def get_default(cls) -> "Config":
        """Retorna configuração padrão."""
        return cls()


def load_config(config_path: Optional[str] = None) -> Config:
    """
    Carrega configuração de arquivo YAML ou retorna padrão.
    
    Args:
        config_path: Caminho para arquivo YAML. Se None, usa padrão.
        
    Returns:
        Objeto Config.
    """
    if config_path and Path(config_path).exists():
        return Config.from_yaml(config_path)
    return Config.get_default()
