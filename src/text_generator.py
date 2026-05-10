"""
Modul Text Generator
Mengubah data tabular CSV posyandu menjadi teks laporan naratif
untuk klasifikasi Zero-Shot Learning.
"""

import random
import pandas as pd

from config import (
    GIZI_BALITA_PATH,
    IMUNISASI_BAYI_PATH,
    KESEHATAN_BALITA_PATH,
    GENERATED_DATA_PATH,
)

# Template teks - PEMANTAUAN GIZI BALITA
TEMPLATES_GIZI = [
    (
        "Pada tanggal {tanggal}, telah dilakukan pemantauan gizi balita di posyandu. "
        "Balita {nama} ({jk}, usia {umur} bulan) ditimbang dengan hasil berat badan {bb} kg, "
        "tinggi badan {tb} cm, dan lingkar kepala {lk} cm. "
        "Status gizi balita tercatat {status_gizi}. {asi_text} "
        "Pemeriksaan dilakukan oleh {kader}. Keterangan: {ket}."
    ),
    (
        "Laporan penimbangan balita tanggal {tanggal}. "
        "Balita atas nama {nama}, {jk}, usia {umur} bulan menjalani pemantauan pertumbuhan. "
        "Hasil pengukuran: BB={bb} kg, TB={tb} cm, LK={lk} cm. "
        "Status gizi: {status_gizi}. {asi_text} "
        "Petugas: {kader}. Catatan: {ket}."
    ),
    (
        "Kegiatan posyandu tanggal {tanggal} mencatat pemantauan gizi balita {nama} "
        "({jk}, {umur} bulan). Antropometri: BB {bb} kg, TB {tb} cm, LK {lk} cm. "
        "Status gizi {status_gizi}. {asi_text} "
        "Pelayanan oleh {kader}. Keterangan: {ket}."
    ),
    (
        "Hasil pemantauan gizi di posyandu {tanggal}: balita {nama} ({jk}, {umur} bulan). "
        "Berat badan: {bb} kg, tinggi badan: {tb} cm, lingkar kepala: {lk} cm. "
        "Evaluasi status gizi menunjukkan {status_gizi}. {asi_text} "
        "Diperiksa oleh {kader}. Catatan: {ket}."
    ),
    (
        "Pelaksanaan posyandu {tanggal}: penimbangan balita {nama}, {jk}, {umur} bulan. "
        "BB tercatat {bb} kg, TB {tb} cm, LK {lk} cm dengan status gizi {status_gizi}. "
        "{asi_text} Penanggung jawab: {kader}. Keterangan: {ket}."
    ),
]

# Template teks - IMUNISASI BAYI
TEMPLATES_IMUNISASI = [
    (
        "Pada tanggal {tgl_imunisasi}, dilaksanakan pemberian imunisasi di {posyandu}. "
        "Bayi {nama} ({jk}, usia {umur} bulan) mendapatkan imunisasi {jenis}. "
        "Status imunisasi: {status}. Petugas: {petugas}. "
        "Keterangan pasca imunisasi: {ket}."
    ),
    (
        "Laporan imunisasi bayi di {posyandu} tanggal {tgl_imunisasi}. "
        "Bayi {nama}, {jk}, {umur} bulan menerima vaksin {jenis}. "
        "Kelengkapan imunisasi: {status}. Pelayanan oleh {petugas}. "
        "Kondisi setelah imunisasi: {ket}."
    ),
    (
        "Kegiatan imunisasi di {posyandu} pada {tgl_imunisasi}: "
        "pemberian vaksin {jenis} kepada bayi {nama} ({jk}, {umur} bulan). "
        "Status kelengkapan: {status}. Diberikan oleh {petugas}. "
        "Pemantauan KIPI: {ket}."
    ),
    (
        "Pencatatan imunisasi {tgl_imunisasi} di {posyandu}: "
        "Bayi {nama} ({jk}, {umur} bulan), jenis imunisasi: {jenis}. "
        "Status: {status}. Petugas pelaksana: {petugas}. KIPI: {ket}."
    ),
    (
        "Program imunisasi {posyandu} tanggal {tgl_imunisasi}: "
        "bayi {nama} ({jk}, {umur} bulan) divaksinasi {jenis}. "
        "Catatan status: {status}. Pemberian oleh {petugas}. "
        "Observasi: {ket}."
    ),
]

# Template teks - PEMERIKSAAN KESEHATAN BALITA
# Strategi: Menekankan bahwa ini pemeriksaan TERPADU (gizi + imunisasi)
TEMPLATES_KESEHATAN = [
    (
        "Pada tanggal {tanggal}, dilakukan pemeriksaan kesehatan terpadu balita di posyandu. "
        "Balita {nama} ({jk}, {umur} bulan) menjalani pemeriksaan menyeluruh. "
        "Hasil pengukuran pertumbuhan: BB={bb} kg, TB={tb} cm, LK={lk} cm. Status gizi: {status_gizi}. "
        "{asi_text} Selain itu, dilakukan pengecekan kelengkapan imunisasi: {imun_text} "
        "Pemeriksa: {kader}. Catatan perkembangan: {catatan}."
    ),
    (
        "Laporan pemeriksaan kesehatan balita secara komprehensif tanggal {tanggal}. "
        "Balita {nama}, {jk}, {umur} bulan mendapat pelayanan kesehatan terpadu di posyandu. "
        "Pemantauan pertumbuhan: BB {bb} kg, TB {tb} cm, LK {lk} cm dengan status gizi {status_gizi}. "
        "{asi_text} Evaluasi kelengkapan imunisasi: {imun_text} "
        "Petugas: {kader}. Keterangan: {catatan}."
    ),
    (
        "Pemeriksaan kesehatan balita terpadu di posyandu {tanggal} untuk {nama} ({jk}, {umur} bulan). "
        "Aspek pertumbuhan - BB: {bb} kg, TB: {tb} cm, LK: {lk} cm, status gizi: {status_gizi}. "
        "{asi_text} Aspek imunisasi - {imun_text} "
        "Dilayani oleh {kader}. Catatan kesehatan: {catatan}."
    ),
    (
        "Hasil pemeriksaan kesehatan komprehensif balita {tanggal}: "
        "{nama} ({jk}, {umur} bulan). "
        "Pertumbuhan: BB {bb} kg, TB {tb} cm, LK {lk} cm. Status gizi: {status_gizi}. "
        "{asi_text} Riwayat imunisasi diperiksa: {imun_text} "
        "Pemeriksa: {kader}. Catatan perkembangan: {catatan}."
    ),
    (
        "Pelayanan kesehatan terpadu balita di posyandu {tanggal}: {nama}, {jk}, {umur} bulan. "
        "Pemantauan gizi: BB {bb} kg, TB {tb} cm, LK {lk} cm, status gizi {status_gizi}. "
        "{asi_text} Pemantauan imunisasi: {imun_text} "
        "Petugas pelayanan: {kader}. Catatan: {catatan}."
    ),
]


class TextGenerator:
    """Generator teks laporan posyandu dari data tabular CSV."""

    def __init__(self, seed: int = 42):
        random.seed(seed)

    def _asi_text(self, value: str) -> str:
        if value == "Ya":
            return random.choice([
                "Balita mendapatkan ASI eksklusif.",
                "Riwayat ASI eksklusif: Ya.",
                "Tercatat mendapat ASI eksklusif.",
            ])
        return random.choice([
            "Balita tidak mendapatkan ASI eksklusif.",
            "Riwayat ASI eksklusif: Tidak.",
            "ASI eksklusif tidak tercapai.",
        ])

    def _imun_text(self, value: str) -> str:
        if value == "Ya":
            return random.choice([
                "Status kelengkapan imunisasi dasar: LENGKAP.",
                "Imunisasi dasar telah lengkap diberikan.",
                "Kelengkapan imunisasi: sudah lengkap semua.",
            ])
        return random.choice([
            "Status kelengkapan imunisasi dasar: BELUM LENGKAP.",
            "Imunisasi dasar belum lengkap, perlu dijadwalkan.",
            "Kelengkapan imunisasi: belum lengkap.",
        ])

    def generate_from_gizi(self, row: pd.Series) -> str:
        tpl = random.choice(TEMPLATES_GIZI)
        return tpl.format(
            tanggal=row["Tanggal_Pemeriksaan"],
            nama=row["Nama_Balita"],
            jk=row["Jenis_Kelamin"],
            umur=row["Umur_Bulan"],
            bb=row["Berat_Badan"],
            tb=row["Tinggi_Badan"],
            lk=row["Lingkar_Kepala"],
            status_gizi=row["Status_Gizi"],
            asi_text=self._asi_text(row["ASI_Eksklusif"]),
            kader=row["Kader"],
            ket=row["Keterangan"],
        )

    def generate_from_imunisasi(self, row: pd.Series) -> str:
        tpl = random.choice(TEMPLATES_IMUNISASI)
        return tpl.format(
            tgl_imunisasi=row["Tanggal_Imunisasi"],
            posyandu=row["Posyandu"],
            nama=row["Nama_Bayi"],
            jk=row["Jenis_Kelamin"],
            umur=row["Umur_Bulan"],
            jenis=row["Jenis_Imunisasi"],
            status=row["Status_Imunisasi"],
            petugas=row["Petugas"],
            ket=row["Keterangan"],
        )

    def generate_from_kesehatan(self, row: pd.Series) -> str:
        tpl = random.choice(TEMPLATES_KESEHATAN)
        return tpl.format(
            tanggal=row["Tanggal_Pemeriksaan"],
            nama=row["Nama_Balita"],
            jk=row["Jenis_Kelamin"],
            umur=row["Umur_Bulan"],
            bb=row["Berat_Badan"],
            tb=row["Tinggi_Badan"],
            lk=row["Lingkar_Kepala"],
            status_gizi=row["Status_Gizi"],
            asi_text=self._asi_text(row["ASI_Eksklusif"]),
            imun_text=self._imun_text(row["Imunisasi_Lengkap"]),
            kader=row["Kader"],
            catatan=row["Catatan"],
        )

    def generate_all(self, n_per_tema: int = None, shuffle: bool = True) -> pd.DataFrame:
        """Generate seluruh teks laporan dari ketiga file CSV."""
        results = []

        # 1. Gizi Balita
        print("[INFO] Membaca data gizi_balita.csv...")
        df_gizi = pd.read_csv(GIZI_BALITA_PATH)
        if n_per_tema:
            df_gizi = df_gizi.head(n_per_tema)
        print(f"[INFO] Generating {len(df_gizi)} teks laporan gizi...")
        for _, row in df_gizi.iterrows():
            results.append({
                "id": row["ID_Balita"],
                "teks_laporan": self.generate_from_gizi(row),
                "tema_aktual": "Pemantauan Gizi Balita",
                "sumber": "gizi_balita.csv",
            })

        # 2. Imunisasi Bayi
        print("[INFO] Membaca data imunisasi_bayi.csv...")
        df_imun = pd.read_csv(IMUNISASI_BAYI_PATH)
        if n_per_tema:
            df_imun = df_imun.head(n_per_tema)
        print(f"[INFO] Generating {len(df_imun)} teks laporan imunisasi...")
        for _, row in df_imun.iterrows():
            results.append({
                "id": row["ID_Bayi"],
                "teks_laporan": self.generate_from_imunisasi(row),
                "tema_aktual": "Imunisasi Bayi",
                "sumber": "imunisasi_bayi.csv",
            })

        # 3. Kesehatan Balita
        print("[INFO] Membaca data kesehatan_balita.csv...")
        df_kes = pd.read_csv(KESEHATAN_BALITA_PATH)
        if n_per_tema:
            df_kes = df_kes.head(n_per_tema)
        print(f"[INFO] Generating {len(df_kes)} teks laporan kesehatan...")
        for _, row in df_kes.iterrows():
            results.append({
                "id": row["ID_Balita"],
                "teks_laporan": self.generate_from_kesehatan(row),
                "tema_aktual": "Pemeriksaan Kesehatan Balita",
                "sumber": "kesehatan_balita.csv",
            })

        df = pd.DataFrame(results)
        if shuffle:
            df = df.sample(frac=1, random_state=42).reset_index(drop=True)

        print(f"\n[INFO] Total teks: {len(df)}")
        for tema, count in df["tema_aktual"].value_counts().items():
            print(f"       - {tema}: {count}")

        return df

    def generate_and_save(self, n_per_tema: int = None, shuffle: bool = True) -> pd.DataFrame:
        """Generate dan simpan ke CSV."""
        df = self.generate_all(n_per_tema=n_per_tema, shuffle=shuffle)
        df.to_csv(GENERATED_DATA_PATH, index=False)
        print(f"[INFO] Disimpan ke: {GENERATED_DATA_PATH}")
        return df


if __name__ == "__main__":
    gen = TextGenerator()
    df = gen.generate_and_save()
    print("\nContoh teks:")
    for tema in df["tema_aktual"].unique():
        sample = df[df["tema_aktual"] == tema].iloc[0]
        print(f"\n[{tema}]")
        print(f"  {sample['teks_laporan'][:200]}...")
