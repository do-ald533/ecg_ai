"""
Funções para cálculo de métricas de avaliação.
"""
from typing import Dict, Optional
import numpy as np
import torch
from sklearn.metrics import (
    f1_score, 
    precision_score, 
    recall_score, 
    roc_auc_score,
    average_precision_score
)


def compute_metrics(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    threshold: float = 0.5,
    prefix: str = ""
) -> Dict[str, float]:
    """
    Calcula métricas de classificação multi-label.
    
    Args:
        y_true: Ground truth labels [batch, num_classes]
        y_pred: Predições (probabilidades) [batch, num_classes]
        threshold: Threshold para binarização
        prefix: Prefixo para nomes das métricas (ex: "val_")
        
    Returns:
        Dicionário com métricas calculadas
    """

    y_true_np = y_true.cpu().numpy()
    y_pred_np = y_pred.cpu().numpy()
    

    y_pred_bin = (y_pred_np > threshold).astype(int)
    

    metrics = {
        f"{prefix}f1_macro": f1_score(y_true_np, y_pred_bin, average="macro", zero_division=0),
        f"{prefix}f1_micro": f1_score(y_true_np, y_pred_bin, average="micro", zero_division=0),
        f"{prefix}precision_macro": precision_score(y_true_np, y_pred_bin, average="macro", zero_division=0),
        f"{prefix}precision_micro": precision_score(y_true_np, y_pred_bin, average="micro", zero_division=0),
        f"{prefix}recall_macro": recall_score(y_true_np, y_pred_bin, average="macro", zero_division=0),
        f"{prefix}recall_micro": recall_score(y_true_np, y_pred_bin, average="micro", zero_division=0),
    }
    

    try:
        metrics[f"{prefix}auroc_macro"] = roc_auc_score(y_true_np, y_pred_np, average="macro")
        metrics[f"{prefix}auroc_micro"] = roc_auc_score(y_true_np, y_pred_np, average="micro")
    except ValueError:

        metrics[f"{prefix}auroc_macro"] = 0.0
        metrics[f"{prefix}auroc_micro"] = 0.0
    

    try:
        metrics[f"{prefix}ap_macro"] = average_precision_score(y_true_np, y_pred_np, average="macro")
        metrics[f"{prefix}ap_micro"] = average_precision_score(y_true_np, y_pred_np, average="micro")
    except ValueError:
        metrics[f"{prefix}ap_macro"] = 0.0
        metrics[f"{prefix}ap_micro"] = 0.0
    
    return metrics


def compute_per_class_metrics(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    class_names: list,
    threshold: float = 0.5
) -> Dict[str, Dict[str, float]]:
    """
    Calcula métricas por classe.
    
    Args:
        y_true: Ground truth labels [batch, num_classes]
        y_pred: Predições (probabilidades) [batch, num_classes]
        class_names: Lista com nomes das classes
        threshold: Threshold para binarização
        
    Returns:
        Dicionário com métricas por classe
    """
    y_true_np = y_true.cpu().numpy()
    y_pred_np = y_pred.cpu().numpy()
    y_pred_bin = (y_pred_np > threshold).astype(int)
    
    per_class = {}
    
    for i, class_name in enumerate(class_names):
        per_class[class_name] = {
            "f1": f1_score(y_true_np[:, i], y_pred_bin[:, i], zero_division=0),
            "precision": precision_score(y_true_np[:, i], y_pred_bin[:, i], zero_division=0),
            "recall": recall_score(y_true_np[:, i], y_pred_bin[:, i], zero_division=0),
        }
        

        try:
            per_class[class_name]["auroc"] = roc_auc_score(y_true_np[:, i], y_pred_np[:, i])
        except ValueError:
            per_class[class_name]["auroc"] = 0.0
    
    return per_class


class MetricsTracker:
    """
    Rastreador de métricas durante o treinamento.
    """
    
    def __init__(self):
        self.history = []
        self.best_metrics = {}
    
    def update(self, metrics: Dict[str, float], epoch: int) -> None:
        """Adiciona métricas de uma época."""
        metrics_with_epoch = {"epoch": epoch, **metrics}
        self.history.append(metrics_with_epoch)
        

        for key, value in metrics.items():
            if key not in self.best_metrics:
                self.best_metrics[key] = {"value": value, "epoch": epoch}
            elif value > self.best_metrics[key]["value"]:
                self.best_metrics[key] = {"value": value, "epoch": epoch}
    
    def get_best(self, metric_name: str) -> Optional[Dict]:
        """Retorna melhor valor de uma métrica."""
        return self.best_metrics.get(metric_name)
    
    def get_last(self) -> Optional[Dict]:
        """Retorna métricas da última época."""
        return self.history[-1] if self.history else None
    
    def get_history(self) -> list:
        """Retorna histórico completo."""
        return self.history
    
    def summary(self) -> str:
        """Retorna string com resumo das melhores métricas."""
        lines = ["=" * 50]
        lines.append("MELHORES MÉTRICAS")
        lines.append("=" * 50)
        
        for metric_name, info in sorted(self.best_metrics.items()):
            lines.append(f"{metric_name:20s}: {info['value']:.4f} (epoch {info['epoch']})")
        
        lines.append("=" * 50)
        return "\n".join(lines)
