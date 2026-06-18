import os
import random

import pandas as pd

from config import (
    GIZI_CSV,
    KESEHATAN_CSV,
    IMUNISASI_CSV,
    GENERATED_DATA_PATH,
)


class TextGenerator:
    def __init__(self, seed=42):
        random.seed(seed)

        self.gizi_df = self.read_csv(GIZI_CSV, "gizi_balita.csv")
        self.kesehatan_df = self.read_csv(KESEHATAN_CSV, "kesehatan_balita.csv")
        self.imunisasi_df = self.read_csv(IMUNISASI_CSV, "imunisasi_bayi.csv")

    def read_csv(self, path, filename):
        if not os.path.exists(path):
            print(f"[WARNING] File tidak ditemukan: {filename}")
            return pd.DataFrame()

        print(f"[INFO] Membaca data {filename}...")
        df = pd.read_csv(path)

        print(f"[INFO] Kolom {filename}:")
        print(df.columns.tolist())

        return df

    def generate_from_gizi(self, row):
        teks = f"""
Laporan posyandu menunjukkan hasil pemantauan gizi balita.

Balita bernama {row['Nama Balita']} dengan ID {row['ID_Balita']}
berusia {row['Umur (Bulan)']} bulan dan berjenis kelamin {row['Jenis Kelamin']}.

Hasil pengukuran menunjukkan berat badan {row['Berat Badan (kg)']} kg
dan tinggi badan {row['Tinggi Badan (cm)']} cm.

Berdasarkan hasil pemeriksaan, status gizi balita tercatat sebagai {row['Status Gizi']}.
Keterangan dari petugas adalah {row['Keterangan']}.

Petugas posyandu memberikan edukasi kepada orang tua mengenai pemenuhan gizi seimbang,
pemantauan berat badan secara berkala, serta pentingnya membawa balita ke posyandu
untuk memantau pertumbuhan anak.
"""

        return " ".join(teks.split())

    def generate_from_kesehatan(self, row):
        teks = f"""
Laporan posyandu menunjukkan hasil pemeriksaan kesehatan balita.

Balita bernama {row['Nama Balita']} dengan ID {row['ID_Balita']}
berusia {row['Umur (bulan)']} bulan dan berjenis kelamin {row['Jenis Kelamin']}.

Status imunisasi balita tercatat {row['Status Imunisasi']}.
Status kesehatan balita adalah {row['Status Kesehatan']}.
Status tumbuh kembang balita tercatat {row['Status Tumbuh Kembang']}.

Riwayat penyakit yang pernah dialami balita adalah {row['Riwayat Penyakit']}.

Petugas posyandu melakukan pemantauan kesehatan balita, memberikan edukasi kepada orang tua,
serta menyarankan pemeriksaan lanjutan apabila terdapat gangguan kesehatan atau tumbuh kembang.
"""

        return " ".join(teks.split())

    def generate_from_imunisasi(self, row):
        teks = f"""
Laporan posyandu menunjukkan pelayanan imunisasi bayi.

Bayi bernama {row['Nama Bayi']} dengan ID {row['ID_Balita']}
berusia {row['Umur (Bulan)']} bulan dan berjenis kelamin {row['Jenis Kelamin']}.

Jenis imunisasi yang diberikan adalah {row['Jenis Imunisasi']}.
Status imunisasi bayi tercatat {row['Status Imunisasi']}.

Petugas posyandu memberikan informasi kepada orang tua mengenai manfaat imunisasi,
jadwal imunisasi lanjutan, serta pentingnya melengkapi imunisasi dasar bayi
untuk mencegah penyakit yang dapat dicegah dengan imunisasi.
"""

        return " ".join(teks.split())

    def sample_dataframe(self, df, n):
        if df.empty:
            return pd.DataFrame()

        if n is None:
            return df

        if len(df) >= n:
            return df.sample(n=n, random_state=42)

        return df.sample(n=n, replace=True, random_state=42)

    def generate_all(self, n_per_tema=None, shuffle=True):
        data = []

        if not self.gizi_df.empty:
            gizi_sample = self.sample_dataframe(self.gizi_df, n_per_tema)

            for _, row in gizi_sample.iterrows():
                data.append({
                    "teks_laporan": self.generate_from_gizi(row),
                    "tema_aktual": "Gizi Balita"
                })

        if not self.kesehatan_df.empty:
            kesehatan_sample = self.sample_dataframe(self.kesehatan_df, n_per_tema)

            for _, row in kesehatan_sample.iterrows():
                data.append({
                    "teks_laporan": self.generate_from_kesehatan(row),
                    "tema_aktual": "Kesehatan Balita"
                })

        if not self.imunisasi_df.empty:
            imunisasi_sample = self.sample_dataframe(self.imunisasi_df, n_per_tema)

            for _, row in imunisasi_sample.iterrows():
                data.append({
                    "teks_laporan": self.generate_from_imunisasi(row),
                    "tema_aktual": "Imunisasi Bayi"
                })

        df = pd.DataFrame(data)

        if df.empty:
            raise ValueError(
                "Data kosong. Pastikan file CSV tersedia di folder data."
            )

        if shuffle:
            df = df.sample(frac=1, random_state=42).reset_index(drop=True)

        return df

    def generate_and_save(self, n_per_tema=None, shuffle=True):
        print("[INFO] Generating teks laporan...")

        df = self.generate_all(
            n_per_tema=n_per_tema,
            shuffle=shuffle
        )

        os.makedirs(os.path.dirname(GENERATED_DATA_PATH), exist_ok=True)

        df.to_csv(GENERATED_DATA_PATH, index=False)

        print(f"[INFO] Total teks laporan dibuat: {len(df)}")
        print(f"[INFO] File disimpan ke: {GENERATED_DATA_PATH}")

        return df
