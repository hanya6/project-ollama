"""
Modul Data Loader
Memuat data laporan posyandu dan label tema dari file
"""

import json
import os

import pandas as pd

from config import DATA_DIR, SAMPLE_DATA_PATH, LABELS_PATH


def load_sample_data(file_path: str = None) -> pd.DataFrame:
    """
    Memuat data sampel laporan posyandu dari file CSV.
    
    Args:
        file_path: Path ke file CSV (opsional, default dari config)
        
    Returns:
        DataFrame berisi data laporan posyandu
    """
    path = file_path or SAMPLE_DATA_PATH
    
    if not os.path.exists(path):
        raise FileNotFoundError(f"File data tidak ditemukan: {path}")
    
    df = pd.read_csv(path)
    
    # Validasi kolom yang dibutuhkan
    required_columns = ["id", "teks_laporan", "tema_aktual"]
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Kolom '{col}' tidak ditemukan dalam data")
    
    # Bersihkan teks
    df["teks_laporan"] = df["teks_laporan"].str.strip()
    df["tema_aktual"] = df["tema_aktual"].str.strip()
    
    print(f"[INFO] Data berhasil dimuat: {len(df)} baris")
    print(f"[INFO] Distribusi tema:")
    for tema, count in df["tema_aktual"].value_counts().items():
        print(f"       - {tema}: {count}")
    
    return df


def load_labels(file_path: str = None) -> dict:
    """
    Memuat konfigurasi label/tema dari file JSON.
    
    Args:
        file_path: Path ke file JSON (opsional, default dari config)
        
    Returns:
        Dictionary berisi informasi label tema
    """
    path = file_path or LABELS_PATH
    
    if not os.path.exists(path):
        raise FileNotFoundError(f"File label tidak ditemukan: {path}")
    
    with open(path, "r", encoding="utf-8") as f:
        labels_data = json.load(f)
    
    print(f"[INFO] Label dimuat: {labels_data['metadata']['jumlah_tema']} tema")
    
    return labels_data


def get_tema_names(labels_data: dict) -> list:
    """
    Mengambil daftar nama tema dari data label.
    
    Args:
        labels_data: Dictionary data label dari load_labels()
        
    Returns:
        List nama tema
    """
    return [tema["nama"] for tema in labels_data["tema_labels"]]


def get_tema_descriptions_from_labels(labels_data: dict) -> dict:
    """
    Mengambil deskripsi tema dalam format dictionary.
    
    Args:
        labels_data: Dictionary data label dari load_labels()
        
    Returns:
        Dictionary {nama_tema: deskripsi}
    """
    return {
        tema["nama"]: tema["deskripsi"] 
        for tema in labels_data["tema_labels"]
    }


def load_custom_data(file_path: str) -> pd.DataFrame:
    """
    Memuat data kustom dari file CSV.
    File harus memiliki minimal kolom 'teks_laporan'.
    Kolom 'tema_aktual' opsional (untuk evaluasi).
    
    Args:
        file_path: Path ke file CSV
        
    Returns:
        DataFrame berisi data laporan
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File tidak ditemukan: {file_path}")
    
    df = pd.read_csv(file_path)
    
    if "teks_laporan" not in df.columns:
        raise ValueError("File harus memiliki kolom 'teks_laporan'")
    
    # Tambah kolom id jika tidak ada
    if "id" not in df.columns:
        df["id"] = range(1, len(df) + 1)
    
    df["teks_laporan"] = df["teks_laporan"].str.strip()
    
    has_labels = "tema_aktual" in df.columns
    if has_labels:
        df["tema_aktual"] = df["tema_aktual"].str.strip()
    
    print(f"[INFO] Data kustom dimuat: {len(df)} baris")
    print(f"[INFO] Label tersedia: {'Ya' if has_labels else 'Tidak'}")
    
    return df
