"""
Modul Data Loader
Memuat data CSV posyandu dan data generated untuk klasifikasi.
"""

import os
import pandas as pd

from config import (
    GIZI_BALITA_PATH,
    IMUNISASI_BAYI_PATH,
    KESEHATAN_BALITA_PATH,
    GENERATED_DATA_PATH,
)


def load_generated_data(file_path: str = None) -> pd.DataFrame:
    """Memuat data teks laporan yang sudah di-generate."""
    path = file_path or GENERATED_DATA_PATH
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"File tidak ditemukan: {path}\n"
            "Jalankan 'python main.py --generate' terlebih dahulu."
        )
    df = pd.read_csv(path)
    df["teks_laporan"] = df["teks_laporan"].str.strip()
    df["tema_aktual"] = df["tema_aktual"].str.strip()
    print(f"[INFO] Data generated dimuat: {len(df)} baris")
    for tema, count in df["tema_aktual"].value_counts().items():
        print(f"       - {tema}: {count}")
    return df


def load_gizi_balita() -> pd.DataFrame:
    """Memuat data gizi balita."""
    df = pd.read_csv(GIZI_BALITA_PATH)
    print(f"[INFO] gizi_balita.csv: {len(df)} baris")
    return df


def load_imunisasi_bayi() -> pd.DataFrame:
    """Memuat data imunisasi bayi."""
    df = pd.read_csv(IMUNISASI_BAYI_PATH)
    print(f"[INFO] imunisasi_bayi.csv: {len(df)} baris")
    return df


def load_kesehatan_balita() -> pd.DataFrame:
    """Memuat data kesehatan balita."""
    df = pd.read_csv(KESEHATAN_BALITA_PATH)
    print(f"[INFO] kesehatan_balita.csv: {len(df)} baris")
    return df
