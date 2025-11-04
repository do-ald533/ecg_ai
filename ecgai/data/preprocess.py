"""
Pré-processamento de dados de ECG.
"""
import h5py
import numpy as np
import pandas as pd
import os
from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import logging

logger = logging.getLogger("ECGAI.Preprocess")


def preprocess_ecg_data(
    data_dir: str,
    csv_path: str,
    output_dir: str = "processed_npz",
    labels: list = None,
    test_size: float = 0.2,
    val_split: float = 0.5,
    random_state: int = 42
):
    """
    Pré-processa dados de ECG de arquivos HDF5.
    
    Args:
        data_dir: Diretório com arquivos HDF5 (exams_part*.hdf5)
        csv_path: Caminho para CSV com metadados
        output_dir: Diretório de saída para arquivos processados
        labels: Lista de labels (None = usa padrão)
        test_size: Proporção do test split (0.2 = 20%)
        val_split: Proporção do validation no test (0.5 = metade do test)
        random_state: Seed para reprodutibilidade
    """
    if labels is None:
        labels = ["1dAVb", "RBBB", "LBBB", "SB", "AF", "ST", "normal_ecg"]
    
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Lendo arquivo CSV...")
    df = pd.read_csv(csv_path)
    df[labels] = df[labels].fillna(0).astype(int)
    

    logger.info("Dividindo dados por paciente...")
    patients = df["patient_id"].unique()
    train_pat, test_pat = train_test_split(
        patients, test_size=test_size, random_state=random_state
    )
    val_pat, test_pat = train_test_split(
        test_pat, test_size=val_split, random_state=random_state
    )
    
    def get_split(pid):
        if pid in train_pat:
            return "train"
        elif pid in val_pat:
            return "val"
        else:
            return "test"
    
    df["split"] = df["patient_id"].apply(get_split)
    

    label_map = df.set_index("exam_id")[labels + ["split"]].to_dict(orient="index")
    

    for split in ["train", "val", "test"]:
        (output_dir / split).mkdir(parents=True, exist_ok=True)
    
    logger.info("Iniciando pré-processamento dos arquivos HDF5...")
    

    hdf5_files = sorted([f for f in data_dir.glob("*.hdf5")])
    
    for hdf5_path in tqdm(hdf5_files, desc="Processando arquivos"):
        logger.info(f"Processando {hdf5_path.name}...")
        
        with h5py.File(hdf5_path, "r") as f:
            exam_ids = f["exam_id"][:]
            tracings = f["tracings"]
            
            for i, exam_id in enumerate(exam_ids):
                exam_id = int(exam_id)
                
                if exam_id not in label_map:
                    continue
                
                info = label_map[exam_id]
                signal = tracings[i]
                

                mean = np.mean(signal, axis=0, keepdims=True)
                std = np.std(signal, axis=0, keepdims=True) + 1e-8
                signal = (signal - mean) / std
                

                label_values = np.array(
                    [info[l] for l in labels], dtype=np.float32
                )
                
                split = info["split"]
                

                out_path = output_dir / split / f"{exam_id}.npz"
                np.savez_compressed(
                    out_path,
                    signal=signal.T,
                    label=label_values
                )
    
    logger.info("Pré-processamento concluído com sucesso!")
    logger.info(f"Arquivos salvos em: {output_dir}/{{train,val,test}}/")
    

    for split in ["train", "val", "test"]:
        n_files = len(list((output_dir / split).glob("*.npz")))
        logger.info(f"  {split}: {n_files} amostras")
