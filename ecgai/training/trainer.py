"""
Classe Trainer para gerenciar o processo de treinamento.
"""
import time
from typing import Optional, Dict, Any
from pathlib import Path

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from ecgai.config import Config
from ecgai.training.metrics import compute_metrics, MetricsTracker
from ecgai.utils.checkpoint import save_checkpoint, save_metrics_to_csv, EarlyStopping
from ecgai.utils.distributed import is_main_process
from ecgai.utils.helpers import format_time


class Trainer:
    """
    Classe para gerenciar treinamento distribuído de modelos ECG.
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader,
        val_loader,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        config: Config,
        device: torch.device,
        rank: int = 0,
        world_size: int = 1,
        logger = None,
        scheduler: Optional[Any] = None
    ):
        """
        Args:
            model: Modelo PyTorch
            train_loader: DataLoader de treino
            val_loader: DataLoader de validação
            optimizer: Otimizador
            criterion: Função de perda
            config: Configuração
            device: Device (CPU/GPU)
            rank: Rank do processo
            world_size: Número total de processos
            logger: Logger
            scheduler: Scheduler de learning rate (opcional)
        """
        self.config = config
        self.device = device
        self.rank = rank
        self.world_size = world_size
        self.logger = logger
        

        if world_size > 1:
            self.model = DDP(
                model.to(device),
                device_ids=None,
                broadcast_buffers=False,
                find_unused_parameters=config.distributed.find_unused_parameters
            )
        else:
            self.model = model.to(device)
        

        if config.training.compile_model and hasattr(torch, 'compile'):
            if self.is_main:
                self.log("🔧 Compilando modelo com torch.compile()...")
            self.model = torch.compile(self.model)
        
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        

        self.use_amp = config.training.use_amp and device.type == 'cuda'
        self.scaler = GradScaler() if self.use_amp else None
        

        self.metrics_tracker = MetricsTracker()
        self.early_stopping = EarlyStopping(
            patience=config.training.patience,
            mode="max"
        )
        
        self.current_epoch = 0
        self.best_metric = 0.0
        

        if self.is_main:
            Path(config.checkpoint.save_dir).mkdir(parents=True, exist_ok=True)
    
    @property
    def is_main(self) -> bool:
        """Verifica se é o processo principal."""
        return is_main_process(self.rank)
    
    def log(self, message: str, level: str = "info") -> None:
        """Log apenas do processo principal."""
        if self.is_main and self.logger:
            getattr(self.logger, level)(message)
    
    def train_epoch(self) -> float:
        """
        Executa uma época de treinamento.
        
        Returns:
            Loss média da época
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        

        if self.world_size > 1:
            self.train_loader.sampler.set_epoch(self.current_epoch)
        

        iterator = tqdm(
            self.train_loader, 
            desc=f"Epoch {self.current_epoch}",
            disable=not self.is_main
        )
        
        for batch_idx, (signals, labels) in enumerate(iterator):

            signals = signals.to(self.device, dtype=torch.float32)
            labels = labels.to(self.device, dtype=torch.float32)
            

            self.optimizer.zero_grad(set_to_none=True)
            
            if self.use_amp:
                with autocast():
                    outputs = self.model(signals)
                    loss = self.criterion(outputs, labels)
                

                self.scaler.scale(loss).backward()
                

                if self.config.training.gradient_clip:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.config.training.gradient_clip
                    )
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(signals)
                loss = self.criterion(outputs, labels)
                loss.backward()
                

                if self.config.training.gradient_clip:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.config.training.gradient_clip
                    )
                
                self.optimizer.step()
            

            total_loss += loss.item()
            num_batches += 1
            

            if self.is_main:
                iterator.set_postfix({"loss": f"{loss.item():.4f}"})
        
        avg_loss = total_loss / max(num_batches, 1)
        return avg_loss
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """
        Executa validação.
        
        Returns:
            Dicionário com métricas
        """
        self.model.eval()
        all_outputs = []
        all_labels = []
        
        iterator = tqdm(
            self.val_loader,
            desc="Validating",
            disable=not self.is_main
        )
        
        for signals, labels in iterator:
            signals = signals.to(self.device, dtype=torch.float32)
            labels = labels.to(self.device, dtype=torch.float32)
            
            if self.use_amp:
                with autocast():
                    outputs = self.model(signals)
            else:
                outputs = self.model(signals)
            
            all_outputs.append(outputs.cpu())
            all_labels.append(labels.cpu())
        

        all_outputs = torch.cat(all_outputs, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        

        metrics = compute_metrics(all_labels, all_outputs, prefix="val_")
        
        return metrics
    
    def train(self) -> None:
        """
        Loop principal de treinamento.
        """
        self.log("🚀 Iniciando treinamento...")
        self.log(f"Device: {self.device}")
        self.log(f"World size: {self.world_size}")
        
        start_time = time.time()
        
        for epoch in range(1, self.config.training.epochs + 1):
            self.current_epoch = epoch
            epoch_start = time.time()
            

            if epoch <= self.config.training.warmup_epochs:
                lr = self.config.training.learning_rate * (epoch / self.config.training.warmup_epochs)
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = lr
            

            train_loss = self.train_epoch()
            epoch_time = time.time() - epoch_start
            

            should_validate = (epoch % self.config.training.eval_every == 0) or \
                             (epoch == self.config.training.epochs)
            
            if should_validate:

                if self.world_size > 1:
                    dist.barrier()
                

                if self.is_main:
                    val_metrics = self.validate()
                    

                    metrics = {
                        "train_loss": train_loss,
                        "epoch_time": epoch_time,
                        **val_metrics
                    }
                    

                    self.metrics_tracker.update(metrics, epoch)
                    

                    current_lr = self.optimizer.param_groups[0]['lr']
                    self.log(
                        f"Epoch {epoch:03d}/{self.config.training.epochs} | "
                        f"Loss: {train_loss:.4f} | "
                        f"Val F1: {val_metrics['val_f1_macro']:.4f} | "
                        f"Val AUROC: {val_metrics['val_auroc_macro']:.4f} | "
                        f"LR: {current_lr:.6f} | "
                        f"Time: {format_time(epoch_time)}"
                    )
                    

                    current_metric = val_metrics['val_f1_macro']
                    
                    if current_metric > self.best_metric:
                        self.best_metric = current_metric
                        self.log(f"✅ Novo melhor modelo! F1={current_metric:.4f}")
                        

                        save_checkpoint(
                            self.model,
                            self.optimizer,
                            self.scheduler,
                            epoch,
                            metrics,
                            self.config,
                            filename=self.config.checkpoint.best_model_name
                        )
                    

                    if self.config.checkpoint.save_last:
                        save_checkpoint(
                            self.model,
                            self.optimizer,
                            self.scheduler,
                            epoch,
                            metrics,
                            self.config,
                            filename=self.config.checkpoint.last_model_name
                        )
                    

                    if self.early_stopping(current_metric):
                        self.log("⏹️ Early stopping ativado!")
                        break
                    

                    if self.scheduler is not None:
                        if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                            self.scheduler.step(current_metric)
                        else:
                            self.scheduler.step()
                

                if self.world_size > 1:
                    dist.barrier()
            else:
                if self.is_main:
                    current_lr = self.optimizer.param_groups[0]['lr']
                    self.log(
                        f"Epoch {epoch:03d}/{self.config.training.epochs} | "
                        f"Loss: {train_loss:.4f} | "
                        f"LR: {current_lr:.6f} | "
                        f"Time: {format_time(epoch_time)} "
                        "(sem validação)"
                    )
        

        total_time = time.time() - start_time
        
        if self.is_main:
            self.log("=" * 70)
            self.log(f"🏁 Treinamento concluído em {format_time(total_time)}")
            self.log(f"Melhor F1: {self.best_metric:.4f}")
            self.log("=" * 70)
            

            save_metrics_to_csv(
                self.metrics_tracker.get_history(),
                self.config
            )
            

            self.log(self.metrics_tracker.summary())
