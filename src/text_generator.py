import os
import random
import pandas as pd

from config import (
    DATA_DIR,
    GENERATED_DATA_PATH,
    TEMA_LABELS,
)


def get_value(row, possible_columns, default="-"):
    """
    Mengambil nilai dari baris CSV berdasarkan beberapa kemungkinan nama kolom.
    Fungsi ini dibuat agar tidak terjadi KeyError jika nama kolom berbeda.
    """
    for col in possible_columns:
        if col in row.index:
            value = row[col]
            if pd.notna(value):
                return value
    return default


class TextGenerator:
    def __init__(self, seed=42):
        random.seed(seed)

        self.gizi_path = os.path.join(DATA_DIR, "gizi_balita.csv")
        self.imunisasi_path = os.path.join(DATA_DIR, "imunisasi.csv")
        self.ibu_hamil_path = os.path.join(DATA_DIR, "ibu_hamil.csv")
        self.kb_path = os.path.join(DATA_DIR, "kb.csv")
        self.keluhan_path = os.path.join(DATA_DIR, "keluhan.csv")

        self.gizi_df = self._read_csv_safe(self.gizi_path)
        self.imunisasi_df = self._read_csv_safe(self.imunisasi_path)
        self.ibu_hamil_df = self._read_csv_safe(self.ibu_hamil_path)
        self.kb_df = self._read_csv_safe(self.kb_path)
        self.keluhan_df = self._read_csv_safe(self.keluhan_path)

    def _read_csv_safe(self, path):
        if os.path.exists(path):
            print(f"[INFO] Membaca data {os.path.basename(path)}...")
            return pd.read_csv(path)
        else:
            print(f"[WARNING] File tidak ditemukan: {path}")
            return pd.DataFrame()

    def generate_from_gizi(self, row):
        tanggal = get_value(row, [
            "Tanggal_Pemeriksaan",
            "tanggal_pemeriksaan",
            "Tanggal Pemeriksaan",
            "tanggal",
            "Tanggal",
            "tgl_pemeriksaan",
            "Tgl_Pemeriksaan"
        ])

        nama = get_value(row, [
            "Nama_Balita",
            "nama_balita",
            "Nama Balita",
            "nama",
            "Nama"
        ])

        umur = get_value(row, [
            "Umur",
            "umur",
            "Usia",
            "usia",
            "umur_bulan",
            "Usia_Bulan"
        ])

        berat = get_value(row, [
            "Berat_Badan",
            "berat_badan",
            "Berat Badan",
            "bb",
            "BB"
        ])

        tinggi = get_value(row, [
            "Tinggi_Badan",
            "tinggi_badan",
            "Tinggi Badan",
            "tb",
            "TB"
        ])

        status = get_value(row, [
            "Status_Gizi",
            "status_gizi",
            "Status Gizi",
            "status",
            "Status"
        ])

        teks = f"""
Pada tanggal {tanggal}, dilakukan pemeriksaan pertumbuhan balita atas nama {nama}.
Balita berusia {umur} bulan dengan berat badan {berat} kg dan tinggi badan {tinggi} cm.
Berdasarkan hasil pemeriksaan di posyandu, status gizi balita termasuk kategori {status}.
Petugas memberikan edukasi kepada orang tua mengenai pemenuhan nutrisi seimbang,
pemantauan berat badan, serta pentingnya kunjungan rutin ke posyandu.
"""

        return teks.strip()

    def generate_from_imunisasi(self, row):
        tanggal = get_value(row, [
            "Tanggal_Imunisasi",
            "tanggal_imunisasi",
            "Tanggal Imunisasi",
            "tanggal",
            "Tanggal"
        ])

        nama = get_value(row, [
            "Nama_Balita",
            "nama_balita",
            "Nama Balita",
            "nama",
            "Nama"
        ])

        umur = get_value(row, [
            "Umur",
            "umur",
            "Usia",
            "usia",
            "umur_bulan"
        ])

        jenis = get_value(row, [
            "Jenis_Imunisasi",
            "jenis_imunisasi",
            "Jenis Imunisasi",
            "imunisasi",
            "Jenis"
        ])

        status = get_value(row, [
            "Status_Imunisasi",
            "status_imunisasi",
            "Status Imunisasi",
            "status",
            "Status"
        ])

        teks = f"""
Pada tanggal {tanggal}, balita atas nama {nama} yang berusia {umur} bulan
mendapatkan pelayanan imunisasi {jenis} di posyandu.
Status pemberian imunisasi tercatat {status}.
Petugas kesehatan memberikan penjelasan kepada orang tua mengenai manfaat imunisasi,
jadwal imunisasi lanjutan, serta pentingnya melengkapi imunisasi dasar anak.
"""

        return teks.strip()

    def generate_from_ibu_hamil(self, row):
        tanggal = get_value(row, [
            "Tanggal_Pemeriksaan",
            "tanggal_pemeriksaan",
            "Tanggal Pemeriksaan",
            "tanggal",
            "Tanggal"
        ])

        nama = get_value(row, [
            "Nama_Ibu",
            "nama_ibu",
            "Nama Ibu",
            "nama",
            "Nama"
        ])

        usia_kehamilan = get_value(row, [
            "Usia_Kehamilan",
            "usia_kehamilan",
            "Usia Kehamilan",
            "umur_kehamilan",
            "Umur_Kehamilan"
        ])

        tekanan_darah = get_value(row, [
            "Tekanan_Darah",
            "tekanan_darah",
            "Tekanan Darah",
            "tensi",
            "Tensi"
        ])

        keluhan = get_value(row, [
            "Keluhan",
            "keluhan",
            "Keluhan_Ibu",
            "keluhan_ibu"
        ])

        teks = f"""
Pada tanggal {tanggal}, dilakukan pemeriksaan ibu hamil atas nama {nama}.
Usia kehamilan ibu tercatat {usia_kehamilan} minggu dengan tekanan darah {tekanan_darah}.
Ibu menyampaikan keluhan berupa {keluhan}.
Petugas posyandu memberikan pemantauan kesehatan kehamilan, edukasi tanda bahaya kehamilan,
anjuran konsumsi tablet tambah darah, serta menyarankan pemeriksaan lanjutan bila diperlukan.
"""

        return teks.strip()

    def generate_from_kb(self, row):
        tanggal = get_value(row, [
            "Tanggal_Pelayanan",
            "tanggal_pelayanan",
            "Tanggal Pelayanan",
            "tanggal",
            "Tanggal"
        ])

        nama = get_value(row, [
            "Nama_Ibu",
            "nama_ibu",
            "Nama Ibu",
            "nama",
            "Nama"
        ])

        metode = get_value(row, [
            "Metode_KB",
            "metode_kb",
            "Metode KB",
            "jenis_kb",
            "Jenis_KB",
            "KB"
        ])

        status = get_value(row, [
            "Status_KB",
            "status_kb",
            "Status KB",
            "status",
            "Status"
        ])

        teks = f"""
Pada tanggal {tanggal}, ibu atas nama {nama} mendapatkan pelayanan keluarga berencana di posyandu.
Metode kontrasepsi yang digunakan adalah {metode} dengan status pelayanan {status}.
Petugas memberikan konseling mengenai manfaat KB, jadwal kontrol ulang,
kemungkinan efek samping, serta pentingnya pemilihan metode kontrasepsi yang sesuai.
"""

        return teks.strip()

    def generate_from_keluhan(self, row):
        tanggal = get_value(row, [
            "Tanggal_Keluhan",
            "tanggal_keluhan",
            "Tanggal Keluhan",
            "Tanggal_Pemeriksaan",
            "tanggal",
            "Tanggal"
        ])

        nama = get_value(row, [
            "Nama",
            "nama",
            "Nama_Pasien",
            "nama_pasien",
            "Nama_Ibu",
            "Nama_Balita"
        ])

        keluhan = get_value(row, [
            "Keluhan",
            "keluhan",
            "Jenis_Keluhan",
            "jenis_keluhan"
        ])

        tindakan = get_value(row, [
            "Tindakan",
            "tindakan",
            "Penanganan",
            "penanganan",
            "Saran"
        ])

        teks = f"""
Pada tanggal {tanggal}, warga atas nama {nama} datang ke posyandu dengan keluhan {keluhan}.
Petugas melakukan pencatatan keluhan dan memberikan penanganan berupa {tindakan}.
Selain itu, petugas memberikan edukasi kesehatan, anjuran pemantauan kondisi,
serta menyarankan pemeriksaan ke fasilitas kesehatan apabila keluhan berlanjut.
"""

        return teks.strip()

    def _sample_dataframe(self, df, n):
        if df.empty:
            return pd.DataFrame()

        if n is None:
            return df

        if len(df) >= n:
            return df.sample(n=n, random_state=42)
        else:
            return df.sample(n=n, replace=True, random_state=42)

    def generate_all(self, n_per_tema=None, shuffle=True):
        all_data = []

        if not self.gizi_df.empty:
            df_sample = self._sample_dataframe(self.gizi_df, n_per_tema)
            for _, row in df_sample.iterrows():
                all_data.append({
                    "teks_laporan": self.generate_from_gizi(row),
                    "tema_aktual": "Gizi Balita"
                })

        if not self.imunisasi_df.empty:
            df_sample = self._sample_dataframe(self.imunisasi_df, n_per_tema)
            for _, row in df_sample.iterrows():
                all_data.append({
                    "teks_laporan": self.generate_from_imunisasi(row),
                    "tema_aktual": "Imunisasi"
                })

        if not self.ibu_hamil_df.empty:
            df_sample = self._sample_dataframe(self.ibu_hamil_df, n_per_tema)
            for _, row in df_sample.iterrows():
                all_data.append({
                    "teks_laporan": self.generate_from_ibu_hamil(row),
                    "tema_aktual": "Ibu Hamil"
                })

        if not self.kb_df.empty:
            df_sample = self._sample_dataframe(self.kb_df, n_per_tema)
            for _, row in df_sample.iterrows():
                all_data.append({
                    "teks_laporan": self.generate_from_kb(row),
                    "tema_aktual": "Keluarga Berencana"
                })

        if not self.keluhan_df.empty:
            df_sample = self._sample_dataframe(self.keluhan_df, n_per_tema)
            for _, row in df_sample.iterrows():
                all_data.append({
                    "teks_laporan": self.generate_from_keluhan(row),
                    "tema_aktual": "Keluhan Umum"
                })

        df = pd.DataFrame(all_data)

        if shuffle and not df.empty:
            df = df.sample(frac=1, random_state=42).reset_index(drop=True)

        return df

    def generate_and_save(self, n_per_tema=None, shuffle=True):
        print("[INFO] Generating teks laporan...")

        df = self.generate_all(
            n_per_tema=n_per_tema,
            shuffle=shuffle
        )

        if df.empty:
            raise ValueError(
                "Tidak ada data yang berhasil digenerate. "
                "Pastikan file CSV tersedia di folder data."
            )

        os.makedirs(os.path.dirname(GENERATED_DATA_PATH), exist_ok=True)

        df.to_csv(GENERATED_DATA_PATH, index=False)

        print(f"[INFO] Total teks berhasil dibuat: {len(df)}")
        print(f"[INFO] File disimpan ke: {GENERATED_DATA_PATH}")

        return df
