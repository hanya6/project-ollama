"""
Modul Data Loader
Memuat data laporan posyandu dari file CSV (raw dan generated)
serta label tema dari file JSON.
"""

import json
import os

import pandas as pd

from config import (
    DATA_DIR,
    GIZI_BALITA_PATH,
    IMUNISASI_BAYI_PATH,
    KESEHATAN_BALITA_PATH,
    GENERATED_DATA_PATH,
    LABELS_PATH,
)


def load_gizi_balita(file_path: str = None) -> pd.DataFrame:
    """
    Memuat data gizi balita dari CSV.

    Args:
        file_path: Path ke file CSV (opsional)

    Returns:
        DataFrame berisi data gizi balita
    """
    path = file_path or GIZI_BALITA_PATH

    if not os.path.exists(path):
        raise FileNotFoundError(f"File tidak ditemukan: {path}")

    df = pd.read_csv(path)
    print(f"[INFO] Data gizi balita dimuat: {len(df)} baris")
    print(f"[INFO] Kolom: {list(df.columns)}")

    # Statistik singkat
    if "Status_Gizi" in df.columns:
        print(f"[INFO] Distribusi status gizi:")
        for status, count in df["Status_Gizi"].value_counts().items():
            print(f"       - {status}: {count}")

    return df


def load_imunisasi_bayi(file_path: str = None) -> pd.DataFrame:
    """
    Memuat data imunisasi bayi dari CSV.

    Args:
        file_path: Path ke file CSV (opsional)

    Returns:
        DataFrame berisi data imunisasi bayi
    """
    path = file_path or IMUNISASI_BAYI_PATH

    if not os.path.exists(path):
        raise FileNotFoundError(f"File tidak ditemukan: {path}")

    df = pd.read_csv(path)
    print(f"[INFO] Data imunisasi bayi dimuat: {len(df)} baris")
    print(f"[INFO] Kolom: {list(df.columns)}")

    # Statistik singkat
    if "Jenis_Imunisasi" in df.columns:
        print(f"[INFO] Distribusi jenis imunisasi:")
        for jenis, count in df["Jenis_Imunisasi"].value_counts().items():
            print(f"       - {jenis}: {count}")

    return df


def load_kesehatan_balita(file_path: str = None) -> pd.DataFrame:
    """
    Memuat data kesehatan balita dari CSV.

    Args:
        file_path: Path ke file CSV (opsional)

    Returns:
        DataFrame berisi data kesehatan balita
    """
    path = file_path or KESEHATAN_BALITA_PATH

    if not os.path.exists(path):
        raise FileNotFoundError(f"File tidak ditemukan: {path}")

    df = pd.read_csv(path)
    print(f"[INFO] Data kesehatan balita dimuat: {len(df)} baris")
    print(f"[INFO] Kolom: {list(df.columns)}")

    # Statistik singkat
    if "Status_Gizi" in df.columns:
        print(f"[INFO] Distribusi status gizi:")
        for status, count in df["Status_Gizi"].value_counts().items():
            print(f"       - {status}: {count}")

    return df


def load_generated_data(file_path: str = None) -> pd.DataFrame:
    """
    Memuat data teks laporan yang sudah di-generate dari CSV.

    Args:
        file_path: Path ke file CSV (opsional)

    Returns:
        DataFrame berisi teks laporan dan label tema
    """
    path = file_path or GENERATED_DATA_PATH

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"File generated data tidak ditemukan: {path}\n"
            f"Jalankan 'python -m src.text_generator' atau 'python main.py --generate' "
            f"terlebih dahulu untuk men-generate teks laporan."
        )

    df = pd.read_csv(path)

    # Validasi kolom yang dibutuhkan
    required_columns = ["id", "teks_laporan", "tema_aktual"]
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Kolom '{col}' tidak ditemukan dalam data generated")

    # Bersihkan teks
    df["teks_laporan"] = df["teks_laporan"].str.strip()
    df["tema_aktual"] = df["tema_aktual"].str.strip()

    print(f"[INFO] Data generated dimuat: {len(df)} baris")
    print(f"[INFO] Distribusi tema:")
    for tema, count in df["tema_aktual"].value_counts().items():
        print(f"       - {tema}: {count}")

    return df


def load_labels(file_path: str = None) -> dict:
    """
    Memuat konfigurasi label/tema dari file JSON.

    Args:
        file_path: Path ke file JSON (opsional)

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


def get_data_summary() -> dict:
    """
    Mendapatkan ringkasan statistik dari semua data yang tersedia.

    Returns:
        Dictionary berisi ringkasan data
    """
    summary = {}

    # Cek file gizi balita
    if os.path.exists(GIZI_BALITA_PATH):
        df = pd.read_csv(GIZI_BALITA_PATH)
        summary["gizi_balita"] = {
            "jumlah": len(df),
            "kolom": list(df.columns),
            "file": GIZI_BALITA_PATH,
        }

    # Cek file imunisasi bayi
    if os.path.exists(IMUNISASI_BAYI_PATH):
        df = pd.read_csv(IMUNISASI_BAYI_PATH)
        summary["imunisasi_bayi"] = {
            "jumlah": len(df),
            "kolom": list(df.columns),
            "file": IMUNISASI_BAYI_PATH,
        }

    # Cek file kesehatan balita
    if os.path.exists(KESEHATAN_BALITA_PATH):
        df = pd.read_csv(KESEHATAN_BALITA_PATH)
        summary["kesehatan_balita"] = {
            "jumlah": len(df),
            "kolom": list(df.columns),
            "file": KESEHATAN_BALITA_PATH,
        }

    # Cek file generated
    if os.path.exists(GENERATED_DATA_PATH):
        df = pd.read_csv(GENERATED_DATA_PATH)
        summary["generated_laporan"] = {
            "jumlah": len(df),
            "distribusi_tema": df["tema_aktual"].value_counts().to_dict(),
            "file": GENERATED_DATA_PATH,
        }

    return summary
