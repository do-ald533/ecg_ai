#!/usr/bin/env python3
"""
ECGAI - Ponto de Entrada Principal

Script principal para treinamento de modelos de classificação de ECG.

Uso:
    python main.py train --config config.yaml
    torchrun --standalone --nproc_per_node=4 main.py train --config config.yaml
    python main.py preprocess --data-dir dataset/unzipped --csv-path dataset/exams.csv
    python main.py evaluate --checkpoint checkpoints/best_model.pt
"""
import argparse
import sys
import os
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))

from ecgai.config import load_config, Config
from ecgai.data import create_dataloaders
from ecgai.data.dataset import compute_class_weights
from ecgai.models import ECG_CNN1D
from ecgai.training import Trainer
from ecgai.utils.logging import setup_logging, log_config
from ecgai.utils.distributed import setup_distributed, cleanup_distributed
from ecgai.utils.helpers import get_device, count_parameters


def parse_args():
    parser = argparse.ArgumentParser(
        description="ECGAI - Classificação de ECG com Deep Learning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Comandos disponíveis')
    
    train_parser = subparsers.add_parser('train', help='Treinar modelo')
    train_parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Caminho para arquivo de configuração YAML'
    )
    train_parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Caminho para checkpoint para retomar treinamento'
    )
    
    eval_parser = subparsers.add_parser('evaluate', help='Avaliar modelo')
    eval_parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Caminho para checkpoint do modelo'
    )
    eval_parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Caminho para arquivo de configuração YAML'
    )
    
    preprocess_parser = subparsers.add_parser('preprocess', help='Pré-processar dados')
    preprocess_parser.add_argument(
        '--data-dir',
        type=str,
        required=True,
        help='Diretório com arquivos HDF5'
    )
    preprocess_parser.add_argument(
        '--csv-path',
        type=str,
        required=True,
        help='Caminho para arquivo CSV com metadados'
    )
    preprocess_parser.add_argument(
        '--output-dir',
        type=str,
        default='processed_npz',
        help='Diretório de saída para arquivos processados'
    )
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        sys.exit(1)
    
    return args


def setup_model_and_optimizer(config: Config, device: torch.device):
    model = ECG_CNN1D(
        n_leads=config.data.num_leads,
        n_classes=len(config.data.labels),
        dropout_rate=config.model.dropout_rate
    )
    
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay
    )
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,
        patience=3,
        verbose=False
    )
    
    weights_path = Path(config.checkpoint.save_dir) / config.checkpoint.weights_name
    
    if weights_path.exists():
        class_weights = torch.load(weights_path)
    else:
        class_weights = compute_class_weights(
            config.data.data_dir,
            config.data.labels
        )
        weights_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(class_weights, weights_path)
    
    class_weights = class_weights.to(device)
    criterion = torch.nn.BCELoss(weight=class_weights)
    
    return model, optimizer, scheduler, criterion


def train_command(args):
    is_distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    
    if is_distributed:
        rank, world_size = setup_distributed()
    else:
        rank, world_size = 0, 1
    
    try:
        config = load_config(args.config)
        logger = setup_logging(config, rank)
        
        if rank == 0:
            logger.info("=" * 70)
            logger.info("ECGAI - TREINAMENTO DE MODELO ECG")
            logger.info("=" * 70)
            log_config(config, logger)
            logger.info(f"\nConfiguração carregada de: {args.config}")
        
        device = get_device()
        
        if rank == 0:
            logger.info(f"\nDevice: {device}")
            logger.info(f"Distributed: {is_distributed}")
            if is_distributed:
                logger.info(f"World size: {world_size}")
        
        if rank == 0:
            logger.info("\nCriando DataLoaders...")
        
        train_loader, val_loader, test_loader = create_dataloaders(
            config, rank, world_size
        )
        
        if rank == 0:
            logger.info(f"  Train samples: {len(train_loader.dataset)}")
            logger.info(f"  Val samples: {len(val_loader.dataset)}")
            logger.info(f"  Test samples: {len(test_loader.dataset)}")
            logger.info(f"  Batch size: {config.training.batch_size}")
            logger.info(f"  Num workers: {config.dataloader.get_num_workers(world_size)}")
        
        if rank == 0:
            logger.info("\nCriando modelo...")
        
        model, optimizer, scheduler, criterion = setup_model_and_optimizer(
            config, device
        )
        
        if rank == 0:
            num_params = count_parameters(model)
            logger.info(f"  Parâmetros treináveis: {num_params:,}")
        
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            criterion=criterion,
            config=config,
            device=device,
            rank=rank,
            world_size=world_size,
            logger=logger,
            scheduler=scheduler
        )
        
        trainer.train()
        
        if is_distributed:
            cleanup_distributed()
        
        if rank == 0:
            logger.info("\nTreinamento finalizado com sucesso!")
            logger.info(f"Checkpoints salvos em: {config.checkpoint.save_dir}/")
    
    except Exception as e:
        if rank == 0:
            print(f"\nErro durante treinamento: {e}")
            import traceback
            traceback.print_exc()
        
        if is_distributed:
            cleanup_distributed()
        
        raise


def evaluate_command(args):
    print("Avaliação de modelo")
    print(f"Checkpoint: {args.checkpoint}")
    print("\nComando 'evaluate' ainda não implementado.")
    print("Use notebooks/train.ipynb para avaliação manual.")


def preprocess_command(args):
    print("Pré-processamento de dados")
    print(f"Data dir: {args.data_dir}")
    print(f"Output dir: {args.output_dir}")
    
    from ecgai.data.preprocess import preprocess_ecg_data
    
    preprocess_ecg_data(
        data_dir=args.data_dir,
        csv_path=args.csv_path,
        output_dir=args.output_dir
    )


def main():
    torch.backends.mkldnn.enabled = True
    
    try:
        torch.set_float32_matmul_precision("medium")
    except:
        pass
    
    args = parse_args()
    
    if args.command == 'train':
        train_command(args)
    elif args.command == 'evaluate':
        evaluate_command(args)
    elif args.command == 'preprocess':
        preprocess_command(args)
    else:
        print(f"Comando desconhecido: {args.command}")
        sys.exit(1)


if __name__ == "__main__":
    main()
