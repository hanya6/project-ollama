import os

import pandas as pd

from config import GENERATED_DATA_PATH


REQUIRED_COLUMNS = [
    "teks_laporan",
    "tema_aktual",
]


def load_generated_data(
    path: str = GENERATED_DATA_PATH,
) -> pd.DataFrame:
    """
    Membaca dataset teks laporan yang telah dibuat.

    Args:
        path:
            Lokasi file CSV hasil generator.

    Returns:
        DataFrame berisi teks laporan dan label aktual.

    Raises:
        FileNotFoundError:
            Jika file belum tersedia.

        ValueError:
            Jika data kosong atau kolom wajib tidak tersedia.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"File generated data tidak ditemukan: {path}. "
            "Jalankan terlebih dahulu: "
            "python main.py --generate"
        )

    df = pd.read_csv(path)

    if df.empty:
        raise ValueError(
            f"File generated data kosong: {path}"
        )

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Kolom wajib tidak ditemukan: "
            + ", ".join(missing_columns)
        )

    df["teks_laporan"] = (
        df["teks_laporan"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    df["tema_aktual"] = (
        df["tema_aktual"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    empty_text_count = (
        df["teks_laporan"] == ""
    ).sum()

    empty_label_count = (
        df["tema_aktual"] == ""
    ).sum()

    if empty_text_count > 0:
        raise ValueError(
            f"Ditemukan {empty_text_count} teks laporan kosong."
        )

    if empty_label_count > 0:
        raise ValueError(
            f"Ditemukan {empty_label_count} label aktual kosong."
        )

    print(
        f"[INFO] Generated data berhasil dibaca: {path}"
    )

    print(
        f"[INFO] Total data: {len(df)}"
    )

    return df
