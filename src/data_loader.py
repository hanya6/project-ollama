import os
import pandas as pd

from config import GENERATED_DATA_PATH


def load_generated_data(path=GENERATED_DATA_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"File generated data tidak ditemukan: {path}. "
            "Jalankan terlebih dahulu: python main.py --generate"
        )

    df = pd.read_csv(path)

    required_columns = ["teks_laporan", "tema_aktual"]

    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Kolom wajib tidak ditemukan: {col}")

    return df
