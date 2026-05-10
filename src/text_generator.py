"""
Modul Text Generator
Menghasilkan teks laporan naratif dari data tabular CSV Posyandu.
Teks yang dihasilkan mensimulasikan laporan kegiatan posyandu
yang ditulis oleh kader atau bidan.
"""

import random
import pandas as pd
from typing import List, Tuple

from config import (
    GIZI_BALITA_PATH,
    IMUNISASI_BAYI_PATH,
    KESEHATAN_BALITA_PATH,
    GENERATED_DATA_PATH,
    FILE_TEMA_MAPPING,
)


# ============================================================
# TEMPLATE TEKS LAPORAN - PEMANTAUAN GIZI BALITA
# ============================================================

TEMPLATES_GIZI = [
    (
        "Pada tanggal {tanggal}, telah dilakukan pemantauan gizi balita di posyandu. "
        "Balita {nama} ({jk}, usia {umur} bulan) ditimbang dengan hasil berat badan {bb} kg, "
        "tinggi badan {tb} cm, dan lingkar kepala {lk} cm. "
        "Status gizi balita tercatat {status_gizi}. "
        "{asi_text} "
        "Pemeriksaan dilakukan oleh {kader}. "
        "Keterangan: {keterangan}."
    ),
    (
        "Laporan kegiatan penimbangan balita tanggal {tanggal}. "
        "Balita atas nama {nama}, jenis kelamin {jk}, berusia {umur} bulan "
        "telah menjalani pemantauan pertumbuhan. "
        "Hasil pengukuran: BB = {bb} kg, TB = {tb} cm, LK = {lk} cm. "
        "Berdasarkan pengukuran tersebut, status gizi balita adalah {status_gizi}. "
        "{asi_text} "
        "Petugas yang melayani: {kader}. Catatan: {keterangan}."
    ),
    (
        "Kegiatan posyandu tanggal {tanggal} mencatat hasil pemantauan gizi "
        "untuk balita {nama} ({jk}, {umur} bulan). "
        "Pengukuran antropometri menunjukkan berat badan {bb} kg, "
        "tinggi badan {tb} cm, dan lingkar kepala {lk} cm. "
        "Balita ini memiliki status gizi {status_gizi}. "
        "{asi_text} "
        "Pelayanan oleh {kader} dengan catatan: {keterangan}."
    ),
    (
        "Hasil pemantauan gizi balita di posyandu pada {tanggal}: "
        "Nama balita: {nama}, jenis kelamin: {jk}, usia: {umur} bulan. "
        "Data antropometri - Berat badan: {bb} kg, Tinggi badan: {tb} cm, "
        "Lingkar kepala: {lk} cm. Status gizi yang tercatat: {status_gizi}. "
        "{asi_text} "
        "Diperiksa oleh {kader}. Keterangan tambahan: {keterangan}."
    ),
    (
        "Pada pelaksanaan posyandu tanggal {tanggal}, dilakukan penimbangan "
        "dan pengukuran terhadap balita {nama} yang berjenis kelamin {jk} "
        "dan berusia {umur} bulan. Berat badan tercatat {bb} kg, "
        "tinggi badan {tb} cm, serta lingkar kepala {lk} cm. "
        "Hasil evaluasi menunjukkan status gizi {status_gizi}. "
        "{asi_text} "
        "Penanggung jawab pemeriksaan: {kader}. Catatan: {keterangan}."
    ),
]


# ============================================================
# TEMPLATE TEKS LAPORAN - IMUNISASI BAYI
# ============================================================

TEMPLATES_IMUNISASI = [
    (
        "Pada tanggal {tgl_imunisasi}, telah dilaksanakan pemberian imunisasi "
        "di {posyandu}. Bayi {nama} ({jk}, usia {umur} bulan) "
        "mendapatkan imunisasi {jenis_imunisasi}. "
        "Status imunisasi: {status_imunisasi}. "
        "Petugas pelaksana: {petugas}. "
        "Keterangan pasca imunisasi: {keterangan}."
    ),
    (
        "Laporan pelayanan imunisasi bayi di {posyandu} tanggal {tgl_imunisasi}. "
        "Bayi atas nama {nama}, jenis kelamin {jk}, berusia {umur} bulan "
        "telah menerima vaksin {jenis_imunisasi}. "
        "Kelengkapan imunisasi saat ini: {status_imunisasi}. "
        "Pelayanan dilakukan oleh {petugas}. "
        "Kondisi setelah imunisasi: {keterangan}."
    ),
    (
        "Kegiatan imunisasi di {posyandu} pada tanggal {tgl_imunisasi} "
        "mencatat pemberian vaksin {jenis_imunisasi} kepada bayi {nama} "
        "({jk}, {umur} bulan). "
        "Status kelengkapan imunisasi: {status_imunisasi}. "
        "Imunisasi diberikan oleh {petugas}. "
        "Pemantauan KIPI: {keterangan}."
    ),
    (
        "Hasil pencatatan imunisasi bayi tanggal {tgl_imunisasi} di {posyandu}: "
        "Nama bayi: {nama}, jenis kelamin: {jk}, usia: {umur} bulan. "
        "Jenis imunisasi yang diberikan: {jenis_imunisasi}. "
        "Status imunisasi tercatat: {status_imunisasi}. "
        "Petugas yang memberikan: {petugas}. "
        "Kejadian Ikutan Pasca Imunisasi (KIPI): {keterangan}."
    ),
    (
        "Pelaksanaan program imunisasi di {posyandu} tanggal {tgl_imunisasi}. "
        "Bayi bernama {nama} yang berjenis kelamin {jk} dan berusia {umur} bulan "
        "telah divaksinasi dengan {jenis_imunisasi}. "
        "Catatan status imunisasi: {status_imunisasi}. "
        "Pemberian vaksin oleh {petugas}. "
        "Observasi setelah pemberian: {keterangan}."
    ),
]


# ============================================================
# TEMPLATE TEKS LAPORAN - PEMERIKSAAN KESEHATAN BALITA
# ============================================================

TEMPLATES_KESEHATAN = [
    (
        "Pada tanggal {tanggal}, telah dilakukan pemeriksaan kesehatan balita "
        "di posyandu. Balita {nama} ({jk}, usia {umur} bulan) diperiksa secara menyeluruh. "
        "Hasil pemeriksaan: BB = {bb} kg, TB = {tb} cm, LK = {lk} cm. "
        "Status gizi: {status_gizi}. "
        "{asi_text} "
        "{imunisasi_text} "
        "Pemeriksaan oleh {kader}. Catatan: {catatan}."
    ),
    (
        "Laporan pemeriksaan kesehatan terpadu balita tanggal {tanggal}. "
        "Balita atas nama {nama}, {jk}, usia {umur} bulan "
        "mendapatkan pelayanan kesehatan lengkap di posyandu. "
        "Pengukuran: berat badan {bb} kg, tinggi badan {tb} cm, lingkar kepala {lk} cm. "
        "Status gizi tercatat {status_gizi}. "
        "{asi_text} "
        "{imunisasi_text} "
        "Petugas: {kader}. Keterangan: {catatan}."
    ),
    (
        "Kegiatan pemeriksaan kesehatan balita di posyandu tanggal {tanggal} "
        "untuk balita {nama} ({jk}, {umur} bulan). "
        "Data pertumbuhan: BB {bb} kg, TB {tb} cm, LK {lk} cm. "
        "Evaluasi status gizi menunjukkan hasil {status_gizi}. "
        "{asi_text} "
        "{imunisasi_text} "
        "Dilayani oleh {kader}. Catatan perkembangan: {catatan}."
    ),
    (
        "Hasil pemeriksaan kesehatan balita secara komprehensif pada {tanggal}: "
        "Nama: {nama}, Jenis kelamin: {jk}, Usia: {umur} bulan. "
        "Antropometri - BB: {bb} kg, TB: {tb} cm, LK: {lk} cm. "
        "Status gizi: {status_gizi}. "
        "{asi_text} "
        "{imunisasi_text} "
        "Pemeriksa: {kader}. Keterangan: {catatan}."
    ),
    (
        "Pelayanan kesehatan balita di posyandu pada tanggal {tanggal} "
        "mencatat pemeriksaan untuk balita {nama} yang berjenis kelamin {jk} "
        "dan berusia {umur} bulan. "
        "Hasil pengukuran menunjukkan BB {bb} kg, TB {tb} cm, dan LK {lk} cm "
        "dengan status gizi {status_gizi}. "
        "{asi_text} "
        "{imunisasi_text} "
        "Penanggung jawab: {kader}. Catatan: {catatan}."
    ),
]


class TextGenerator:
    """
    Generator teks laporan posyandu dari data tabular.
    Mengubah data CSV menjadi teks naratif yang realistis
    untuk keperluan klasifikasi Zero-Shot Learning.
    """

    def __init__(self, seed: int = 42):
        """
        Inisialisasi generator.

        Args:
            seed: Random seed untuk reproduktifitas
        """
        random.seed(seed)
        self.templates_gizi = TEMPLATES_GIZI
        self.templates_imunisasi = TEMPLATES_IMUNISASI
        self.templates_kesehatan = TEMPLATES_KESEHATAN

    def _generate_asi_text(self, asi_value: str) -> str:
        """Generate teks tentang ASI eksklusif."""
        if asi_value == "Ya":
            options = [
                "Balita mendapatkan ASI eksklusif.",
                "Tercatat mendapat ASI eksklusif selama 6 bulan.",
                "Riwayat pemberian ASI eksklusif: Ya.",
                "Ibu memberikan ASI eksklusif.",
            ]
        else:
            options = [
                "Balita tidak mendapatkan ASI eksklusif.",
                "Riwayat ASI eksklusif: Tidak.",
                "Tidak diberikan ASI eksklusif.",
                "Pemberian ASI eksklusif tidak tercapai.",
            ]
        return random.choice(options)

    def _generate_imunisasi_text(self, imunisasi_value: str) -> str:
        """Generate teks tentang kelengkapan imunisasi."""
        if imunisasi_value == "Ya":
            options = [
                "Status imunisasi dasar: lengkap.",
                "Imunisasi dasar telah lengkap diberikan.",
                "Kelengkapan imunisasi: sudah lengkap.",
            ]
        else:
            options = [
                "Status imunisasi dasar: belum lengkap.",
                "Imunisasi dasar belum lengkap.",
                "Kelengkapan imunisasi: belum lengkap, perlu dijadwalkan.",
            ]
        return random.choice(options)

    def generate_from_gizi(self, row: pd.Series) -> str:
        """
        Generate teks laporan dari satu baris data gizi_balita.csv.

        Args:
            row: Satu baris DataFrame gizi balita

        Returns:
            Teks laporan naratif
        """
        template = random.choice(self.templates_gizi)
        asi_text = self._generate_asi_text(row["ASI_Eksklusif"])

        text = template.format(
            tanggal=row["Tanggal_Pemeriksaan"],
            nama=row["Nama_Balita"],
            jk=row["Jenis_Kelamin"],
            umur=row["Umur_Bulan"],
            bb=row["Berat_Badan"],
            tb=row["Tinggi_Badan"],
            lk=row["Lingkar_Kepala"],
            status_gizi=row["Status_Gizi"],
            asi_text=asi_text,
            kader=row["Kader"],
            keterangan=row["Keterangan"],
        )
        return text

    def generate_from_imunisasi(self, row: pd.Series) -> str:
        """
        Generate teks laporan dari satu baris data imunisasi_bayi.csv.

        Args:
            row: Satu baris DataFrame imunisasi bayi

        Returns:
            Teks laporan naratif
        """
        template = random.choice(self.templates_imunisasi)

        text = template.format(
            tgl_imunisasi=row["Tanggal_Imunisasi"],
            posyandu=row["Posyandu"],
            nama=row["Nama_Bayi"],
            jk=row["Jenis_Kelamin"],
            umur=row["Umur_Bulan"],
            jenis_imunisasi=row["Jenis_Imunisasi"],
            status_imunisasi=row["Status_Imunisasi"],
            petugas=row["Petugas"],
            keterangan=row["Keterangan"],
        )
        return text

    def generate_from_kesehatan(self, row: pd.Series) -> str:
        """
        Generate teks laporan dari satu baris data kesehatan_balita.csv.

        Args:
            row: Satu baris DataFrame kesehatan balita

        Returns:
            Teks laporan naratif
        """
        template = random.choice(self.templates_kesehatan)
        asi_text = self._generate_asi_text(row["ASI_Eksklusif"])
        imunisasi_text = self._generate_imunisasi_text(row["Imunisasi_Lengkap"])

        text = template.format(
            tanggal=row["Tanggal_Pemeriksaan"],
            nama=row["Nama_Balita"],
            jk=row["Jenis_Kelamin"],
            umur=row["Umur_Bulan"],
            bb=row["Berat_Badan"],
            tb=row["Tinggi_Badan"],
            lk=row["Lingkar_Kepala"],
            status_gizi=row["Status_Gizi"],
            asi_text=asi_text,
            imunisasi_text=imunisasi_text,
            kader=row["Kader"],
            catatan=row["Catatan"],
        )
        return text

    def generate_all(
        self,
        n_per_tema: int = None,
        shuffle: bool = True,
    ) -> pd.DataFrame:
        """
        Generate seluruh teks laporan dari ketiga file CSV.

        Args:
            n_per_tema: Jumlah sampel per tema (None = semua data)
            shuffle: Acak urutan data

        Returns:
            DataFrame berisi teks laporan dan label tema
        """
        results = []

        # --- 1. Gizi Balita ---
        print("[INFO] Membaca data gizi_balita.csv...")
        df_gizi = pd.read_csv(GIZI_BALITA_PATH)
        if n_per_tema:
            df_gizi = df_gizi.head(n_per_tema)

        print(f"[INFO] Generating {len(df_gizi)} teks laporan gizi balita...")
        for idx, row in df_gizi.iterrows():
            text = self.generate_from_gizi(row)
            results.append({
                "id": row["ID_Balita"],
                "teks_laporan": text,
                "tema_aktual": "Pemantauan Gizi Balita",
                "sumber_file": "gizi_balita.csv",
            })

        # --- 2. Imunisasi Bayi ---
        print("[INFO] Membaca data imunisasi_bayi.csv...")
        df_imunisasi = pd.read_csv(IMUNISASI_BAYI_PATH)
        if n_per_tema:
            df_imunisasi = df_imunisasi.head(n_per_tema)

        print(f"[INFO] Generating {len(df_imunisasi)} teks laporan imunisasi...")
        for idx, row in df_imunisasi.iterrows():
            text = self.generate_from_imunisasi(row)
            results.append({
                "id": row["ID_Bayi"],
                "teks_laporan": text,
                "tema_aktual": "Imunisasi Bayi",
                "sumber_file": "imunisasi_bayi.csv",
            })

        # --- 3. Kesehatan Balita ---
        print("[INFO] Membaca data kesehatan_balita.csv...")
        df_kesehatan = pd.read_csv(KESEHATAN_BALITA_PATH)
        if n_per_tema:
            df_kesehatan = df_kesehatan.head(n_per_tema)

        print(f"[INFO] Generating {len(df_kesehatan)} teks laporan kesehatan balita...")
        for idx, row in df_kesehatan.iterrows():
            text = self.generate_from_kesehatan(row)
            results.append({
                "id": row["ID_Balita"],
                "teks_laporan": text,
                "tema_aktual": "Pemeriksaan Kesehatan Balita",
                "sumber_file": "kesehatan_balita.csv",
            })

        # Buat DataFrame
        df_result = pd.DataFrame(results)

        # Acak urutan
        if shuffle:
            df_result = df_result.sample(frac=1, random_state=42).reset_index(drop=True)

        print(f"\n[INFO] Total teks laporan yang dihasilkan: {len(df_result)}")
        print("[INFO] Distribusi tema:")
        for tema, count in df_result["tema_aktual"].value_counts().items():
            print(f"       - {tema}: {count}")

        return df_result

    def generate_and_save(
        self,
        output_path: str = None,
        n_per_tema: int = None,
        shuffle: bool = True,
    ) -> pd.DataFrame:
        """
        Generate teks laporan dan simpan ke file CSV.

        Args:
            output_path: Path file output (default dari config)
            n_per_tema: Jumlah sampel per tema
            shuffle: Acak urutan data

        Returns:
            DataFrame berisi teks laporan
        """
        df = self.generate_all(n_per_tema=n_per_tema, shuffle=shuffle)

        save_path = output_path or GENERATED_DATA_PATH
        df.to_csv(save_path, index=False)
        print(f"[INFO] Data disimpan ke: {save_path}")

        return df


def generate_sample_texts(n: int = 5) -> List[Tuple[str, str]]:
    """
    Generate beberapa contoh teks untuk preview.

    Args:
        n: Jumlah contoh per tema

    Returns:
        List tuple (teks, tema)
    """
    generator = TextGenerator()
    df = generator.generate_all(n_per_tema=n, shuffle=False)

    samples = []
    for _, row in df.iterrows():
        samples.append((row["teks_laporan"], row["tema_aktual"]))

    return samples


if __name__ == "__main__":
    """Jalankan secara standalone untuk generate data."""
    print("=" * 60)
    print("GENERATOR TEKS LAPORAN POSYANDU")
    print("=" * 60)

    generator = TextGenerator()
    df = generator.generate_and_save()

    # Tampilkan contoh
    print("\n" + "=" * 60)
    print("CONTOH TEKS YANG DIHASILKAN")
    print("=" * 60)

    for tema in df["tema_aktual"].unique():
        print(f"\n--- Tema: {tema} ---")
        sample = df[df["tema_aktual"] == tema].iloc[0]
        print(f"Teks: {sample['teks_laporan'][:200]}...")
        print()
