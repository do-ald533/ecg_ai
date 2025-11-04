"""
Sistema de logging.
"""
import logging
from ecgai.config import Config


def setup_logging(config: Config, rank: int = 0) -> logging.Logger:
    """
    Configura sistema de logging.
    
    Args:
        config: Configuração
        rank: Rank do processo (apenas rank 0 faz log)
        
    Returns:
        Logger configurado
    """
    logger = logging.getLogger("ECGAI")
    logger.setLevel(getattr(logging, config.logging.log_level))
    

    logger.handlers.clear()
    

    if rank == 0:

        console_handler = logging.StreamHandler()
        console_handler.setLevel(getattr(logging, config.logging.log_level))
        formatter = logging.Formatter(
            '%(asctime)s [%(levelname)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        

        if config.logging.log_file:
            file_handler = logging.FileHandler(config.logging.log_file)
            file_handler.setLevel(getattr(logging, config.logging.log_level))
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
    
    return logger


def log_config(config: Config, logger: logging.Logger) -> None:
    """Loga configuração completa."""
    logger.info("=" * 70)
    logger.info("CONFIGURAÇÃO DO TREINAMENTO")
    logger.info("=" * 70)
    
    logger.info(f"Dados:")
    logger.info(f"  Data dir: {config.data.data_dir}")
    logger.info(f"  Num classes: {len(config.data.labels)}")
    logger.info(f"  Signal shape: [{config.data.num_leads}, {config.data.signal_length}]")
    
    logger.info(f"\nTreinamento:")
    logger.info(f"  Batch size: {config.training.batch_size}")
    logger.info(f"  Epochs: {config.training.epochs}")
    logger.info(f"  Learning rate: {config.training.learning_rate}")
    logger.info(f"  Patience: {config.training.patience}")
    
    logger.info(f"\nDataLoader:")
    logger.info(f"  Num workers: auto-detect")
    logger.info(f"  Pin memory: {config.dataloader.pin_memory}")
    
    logger.info("=" * 70)
