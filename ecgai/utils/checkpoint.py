"""
Funções para salvar e carregar checkpoints.
"""
from pathlib import Path
from typing import Optional, Dict, Any
import torch
import pandas as pd
from ecgai.config import Config


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[Any],
    epoch: int,
    metrics: Dict[str, float],
    config: Config,
    filename: str = "checkpoint.pt"
) -> None:
    """
    Salva checkpoint do modelo.
    
    Args:
        model: Modelo PyTorch
        optimizer: Otimizador
        scheduler: Scheduler de LR (opcional)
        epoch: Época atual
        metrics: Métricas da época
        config: Configuração
        filename: Nome do arquivo
    """
    checkpoint_dir = Path(config.checkpoint.save_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint_path = checkpoint_dir / filename
    

    if hasattr(model, 'module'):
        model_state = model.module.state_dict()
    else:
        model_state = model.state_dict()
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model_state,
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
    }
    
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
    
    torch.save(checkpoint, checkpoint_path)


def load_checkpoint(
    checkpoint_path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None
) -> Dict[str, Any]:
    """
    Carrega checkpoint.
    
    Args:
        checkpoint_path: Caminho para checkpoint
        model: Modelo para carregar pesos
        optimizer: Otimizador para carregar estado (opcional)
        scheduler: Scheduler para carregar estado (opcional)
        
    Returns:
        Dicionário com informações do checkpoint
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    

    if hasattr(model, 'module'):
        model.module.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint['model_state_dict'])
    

    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    

    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    return {
        'epoch': checkpoint.get('epoch', 0),
        'metrics': checkpoint.get('metrics', {})
    }


def save_metrics_to_csv(
    metrics_history: list,
    config: Config
) -> None:
    """
    Salva histórico de métricas em CSV.
    
    Args:
        metrics_history: Lista de dicionários com métricas
        config: Configuração
    """
    if not metrics_history:
        return
    
    checkpoint_dir = Path(config.checkpoint.save_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    csv_path = checkpoint_dir / config.checkpoint.metrics_csv
    df = pd.DataFrame(metrics_history)
    df.to_csv(csv_path, index=False)


class EarlyStopping:
    """
    Early stopping para interromper treinamento quando métrica não melhora.
    """
    
    def __init__(
        self, 
        patience: int = 7, 
        min_delta: float = 0.0,
        mode: str = "max"
    ):
        """
        Args:
            patience: Número de épocas sem melhoria antes de parar
            min_delta: Melhoria mínima considerada significativa
            mode: "max" para maximizar métrica, "min" para minimizar
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.should_stop = False
        
    def __call__(self, metric: float) -> bool:
        """
        Verifica se deve parar o treinamento.
        
        Args:
            metric: Valor da métrica atual
            
        Returns:
            True se deve parar, False caso contrário
        """
        score = metric if self.mode == "max" else -metric
        
        if self.best_score is None:
            self.best_score = score
            return False
        
        if score > self.best_score + self.min_delta:

            self.best_score = score
            self.counter = 0
            return False
        else:

            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                return True
            return False
    
    def reset(self) -> None:
        """Reseta o contador."""
        self.counter = 0
        self.best_score = None
        self.should_stop = False
