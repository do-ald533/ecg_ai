"""
Funções para treinamento distribuído.
"""
import os
from datetime import timedelta
import torch
import torch.distributed as dist


def setup_distributed() -> tuple[int, int]:
    """
    Inicializa processo distribuído.
    
    Returns:
        rank, world_size
    """
    dist.init_process_group(
        backend=os.environ.get("BACKEND", "gloo"),
        timeout=timedelta(hours=4)
    )
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    

    torch.manual_seed(42 + rank)
    
    return rank, world_size


def cleanup_distributed() -> None:
    """Limpa processo distribuído."""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(rank: int = 0) -> bool:
    """Verifica se é o processo principal."""
    return rank == 0
