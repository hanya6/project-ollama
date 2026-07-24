import os
import random
from typing import Optional

import pandas as pd

from config import (
    GIZI_CSV,
    KESEHATAN_CSV,
    IMUNISASI_CSV,
    GENERATED_DATA_PATH,
)


class TextGenerator:
    """
    Generator teks laporan Posyandu dengan beberapa tingkat kesulitan.

    Tujuan:
    - Menghindari kebocoran label secara langsung.
    - Membuat teks lebih natural dan bervariasi.
    - Menambahkan sebagian kasus ambigu.
    - Mempertahankan tema_aktual sebagai ground truth.

    Tingkat kesulitan:
    - easy: informasi utama masih jelas.
    - medium: menggunakan ungkapan tidak langsung.
    - hard: informasi lebih singkat dan ambigu.
    """

    LABEL_GIZI = "Gizi Balita"
    LABEL_KESEHATAN = "Kesehatan Balita"
    LABEL_IMUNISASI = "Imunisasi Bayi"

    def __init__(
        self,
        seed: int = 42,
        easy_ratio: float = 0.35,
        medium_ratio: float = 0.40,
        hard_ratio: float = 0.25,
    ):
        self.seed = seed
        self.random = random.Random(seed)

        total_ratio = easy_ratio + medium_ratio + hard_ratio

        if abs(total_ratio - 1.0) > 1e-9:
            raise ValueError(
                "Jumlah easy_ratio, medium_ratio, dan hard_ratio harus 1.0"
            )

        self.easy_ratio = easy_ratio
        self.medium_ratio = medium_ratio
        self.hard_ratio = hard_ratio

        self.gizi_df = self.read_csv(
            GIZI_CSV,
            "gizi_balita.csv"
        )

        self.kesehatan_df = self.read_csv(
            KESEHATAN_CSV,
            "kesehatan_balita.csv"
        )

        self.imunisasi_df = self.read_csv(
            IMUNISASI_CSV,
            "imunisasi_bayi.csv"
        )

    # =====================================================
    # UTILITAS
    # =====================================================

    def read_csv(self, path: str, filename: str) -> pd.DataFrame:
        if not os.path.exists(path):
            print(f"[WARNING] File tidak ditemukan: {path}")
            return pd.DataFrame()

        print(f"[INFO] Membaca data {filename}...")

        df = pd.read_csv(path)

        print(f"[INFO] Jumlah data {filename}: {len(df)}")
        print(f"[INFO] Kolom {filename}: {df.columns.tolist()}")

        return df

    @staticmethod
    def clean_value(value, default: str = "tidak tersedia") -> str:
        if pd.isna(value):
            return default

        value = str(value).strip()

        if not value:
            return default

        return value

    def choose(self, options: list[str]) -> str:
        return self.random.choice(options)

    def determine_difficulty(self) -> str:
        value = self.random.random()

        if value < self.easy_ratio:
            return "easy"

        if value < self.easy_ratio + self.medium_ratio:
            return "medium"

        return "hard"

    @staticmethod
    def normalize_text(text: str) -> str:
        return " ".join(text.split())

    def sample_dataframe(
        self,
        df: pd.DataFrame,
        n: Optional[int],
    ) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame()

        if n is None:
            return df.copy()

        if n <= 0:
            raise ValueError("n_per_tema harus lebih besar dari 0.")

        if len(df) >= n:
            return df.sample(
                n=n,
                random_state=self.seed
            ).copy()

        return df.sample(
            n=n,
            replace=True,
            random_state=self.seed
        ).copy()

    # =====================================================
    # GENERATOR GIZI
    # =====================================================

    def generate_from_gizi(
        self,
        row: pd.Series,
        difficulty: Optional[str] = None,
    ) -> str:
        difficulty = difficulty or self.determine_difficulty()

        nama = self.clean_value(row.get("Nama Balita"))
        identitas = self.clean_value(row.get("ID_Balita"))
        umur = self.clean_value(row.get("Umur (Bulan)"))
        jenis_kelamin = self.clean_value(row.get("Jenis Kelamin"))
        berat = self.clean_value(row.get("Berat Badan (kg)"))
        tinggi = self.clean_value(row.get("Tinggi Badan (cm)"))
        status = self.clean_value(row.get("Status Gizi"))
        keterangan = self.clean_value(row.get("Keterangan"))

        pembuka = self.choose([
            "Balita datang mengikuti kegiatan rutin di posyandu.",
            "Petugas mencatat hasil kunjungan bulanan seorang balita.",
            "Kunjungan dilakukan untuk melihat perubahan kondisi anak.",
            "Orang tua membawa anak untuk mengikuti pelayanan rutin.",
        ])

        if difficulty == "easy":
            teks = f"""
            {pembuka}

            Anak bernama {nama}, ID {identitas}, berusia {umur} bulan
            dan berjenis kelamin {jenis_kelamin}.

            Hasil penimbangan menunjukkan berat badan {berat} kg,
            sedangkan hasil pengukuran tinggi badan adalah {tinggi} cm.

            Berdasarkan hasil pengukuran tersebut, kondisi anak tercatat
            sebagai {status}. Catatan petugas: {keterangan}.

            Orang tua dianjurkan memperhatikan pola makan anak dan
            melakukan pengukuran kembali pada kunjungan berikutnya.
            """

        elif difficulty == "medium":
            teks = f"""
            {pembuka}

            {nama}, usia {umur} bulan, menjalani pengukuran fisik rutin.
            Angka pada timbangan tercatat {berat} kg dan panjang atau
            tinggi tubuh tercatat {tinggi} cm.

            Hasil pencatatan petugas menunjukkan kondisi {status}.
            Terdapat keterangan tambahan berupa {keterangan}.

            Keluarga diminta memperhatikan asupan harian dan mengamati
            perubahan kondisi anak sampai jadwal kunjungan berikutnya.
            """

        else:
            teks = self.choose([
                f"""
                Pada kunjungan bulan ini, {nama} memiliki hasil
                pengukuran {berat} kg dan {tinggi} cm.

                Petugas mencatat kondisi {status} serta memberikan
                arahan kepada keluarga untuk melakukan pemantauan ulang.
                """,

                f"""
                Catatan kunjungan {nama}, usia {umur} bulan:
                hasil pengukuran tubuh adalah {berat} kg dan {tinggi} cm.
                Kondisi yang tercatat yaitu {status}.

                Orang tua diminta memperhatikan kebutuhan harian anak.
                """,

                f"""
                Perubahan kondisi fisik {nama} kembali diamati pada
                kunjungan rutin. Angka yang diperoleh adalah
                {berat} kg dan {tinggi} cm.

                Petugas mencatat {keterangan} dan meminta keluarga
                datang kembali bulan depan.
                """,
            ])

        return self.normalize_text(teks)

    # =====================================================
    # GENERATOR IMUNISASI
    # =====================================================

    def generate_from_imunisasi(
        self,
        row: pd.Series,
        difficulty: Optional[str] = None,
    ) -> str:
        difficulty = difficulty or self.determine_difficulty()

        nama = self.clean_value(row.get("Nama Bayi"))
        identitas = self.clean_value(row.get("ID_Balita"))
        umur = self.clean_value(row.get("Umur (Bulan)"))
        jenis_kelamin = self.clean_value(row.get("Jenis Kelamin"))
        jenis = self.clean_value(row.get("Jenis Imunisasi"))
        status = self.clean_value(row.get("Status Imunisasi"))

        pembuka = self.choose([
            "Bayi datang mengikuti jadwal pelayanan sesuai usianya.",
            "Petugas menerima kunjungan bayi untuk pelayanan lanjutan.",
            "Orang tua membawa bayi sesuai jadwal pada buku KIA.",
            "Kunjungan dilakukan untuk melanjutkan pelayanan pencegahan.",
        ])

        if difficulty == "easy":
            teks = f"""
            {pembuka}

            Bayi bernama {nama}, ID {identitas}, berusia {umur} bulan
            dan berjenis kelamin {jenis_kelamin}.

            Pada kunjungan ini diberikan {jenis}.
            Catatan kelengkapan pelayanan anak adalah {status}.

            Petugas menjelaskan jadwal pemberian berikutnya dan meminta
            orang tua mengamati kondisi bayi setelah pelayanan.
            """

        elif difficulty == "medium":
            teks = f"""
            {pembuka}

            {nama}, usia {umur} bulan, memperoleh tindakan {jenis}
            sesuai catatan jadwal pelayanan anak.

            Riwayat pemberian sebelumnya tercatat {status}.
            Orang tua diminta kembali sesuai jadwal pada buku KIA
            serta mengamati kemungkinan reaksi setelah tindakan.
            """

        else:
            jenis_tidak_langsung = self.make_indirect_immunization_name(jenis)

            teks = self.choose([
                f"""
                {nama} datang sesuai jadwal usia {umur} bulan.
                Petugas memberikan {jenis_tidak_langsung} dan
                memperbarui catatan pelayanan pada buku KIA.

                Orang tua diminta kembali pada jadwal berikutnya.
                """,

                f"""
                Pelayanan lanjutan untuk {nama} dilakukan berdasarkan
                riwayat yang tercatat {status}.

                Pada kunjungan ini anak menerima {jenis_tidak_langsung}.
                Keluarga diberi penjelasan mengenai reaksi ringan
                yang mungkin muncul setelah tindakan.
                """,

                f"""
                Buku KIA milik {nama} diperiksa pada kunjungan hari ini.
                Petugas kemudian memberikan tindakan sesuai usia anak
                dan memperbarui catatan kelengkapannya.

                Jadwal berikutnya telah disampaikan kepada keluarga.
                """,
            ])

        return self.normalize_text(teks)

    @staticmethod
    def make_indirect_immunization_name(jenis: str) -> str:
        jenis_lower = jenis.lower()

        if "bcg" in jenis_lower:
            return "pelayanan untuk perlindungan awal terhadap tuberkulosis"

        if "dpt" in jenis_lower:
            return "pelayanan lanjutan untuk perlindungan terhadap beberapa penyakit"

        if "polio" in jenis_lower:
            return "tetes perlindungan yang diberikan sesuai jadwal usia"

        if "campak" in jenis_lower or "mr" in jenis_lower:
            return "suntikan perlindungan sesuai jadwal anak"

        return "tindakan pencegahan sesuai jadwal usia"

    # =====================================================
    # GENERATOR KESEHATAN
    # =====================================================

    def generate_from_kesehatan(
        self,
        row: pd.Series,
        difficulty: Optional[str] = None,
    ) -> str:
        difficulty = difficulty or self.determine_difficulty()

        nama = self.clean_value(row.get("Nama Balita"))
        identitas = self.clean_value(row.get("ID_Balita"))
        umur = self.clean_value(
            row.get(
                "Umur (bulan)",
                row.get("Umur (Bulan)")
            )
        )
        jenis_kelamin = self.clean_value(row.get("Jenis Kelamin"))
        status_imunisasi = self.clean_value(row.get("Status Imunisasi"))
        status_kesehatan = self.clean_value(row.get("Status Kesehatan"))
        tumbuh_kembang = self.clean_value(
            row.get("Status Tumbuh Kembang")
        )
        riwayat = self.clean_value(row.get("Riwayat Penyakit"))

        pembuka = self.choose([
            "Balita mengikuti kunjungan rutin untuk penilaian menyeluruh.",
            "Petugas melakukan pencatatan beberapa aspek kondisi anak.",
            "Kunjungan hari ini mencakup peninjauan kondisi dan perkembangan anak.",
            "Orang tua membawa anak untuk menjalani pelayanan rutin terpadu.",
        ])

        if difficulty == "easy":
            teks = f"""
            {pembuka}

            Anak bernama {nama}, ID {identitas}, berusia {umur} bulan
            dan berjenis kelamin {jenis_kelamin}.

            Kondisi umum anak tercatat {status_kesehatan}.
            Perkembangan anak tercatat {tumbuh_kembang}.
            Riwayat pelayanan pencegahan tercatat {status_imunisasi}.
            Riwayat penyakit yang dilaporkan adalah {riwayat}.

            Petugas memberikan saran berdasarkan seluruh hasil
            pemeriksaan dan pencatatan tersebut.
            """

        elif difficulty == "medium":
            teks = f"""
            {pembuka}

            {nama}, usia {umur} bulan, menjalani penilaian kondisi umum,
            perkembangan sesuai usia, dan riwayat pelayanan sebelumnya.

            Hasil penilaian kondisi umum adalah {status_kesehatan}.
            Perkembangannya tercatat {tumbuh_kembang}, sedangkan catatan
            pelayanan sebelumnya adalah {status_imunisasi}.

            Keluarga juga menyampaikan riwayat berupa {riwayat}.
            Petugas memberikan arahan tindak lanjut sesuai temuan.
            """

        else:
            teks = self.choose([
                f"""
                Pada kunjungan {nama}, petugas meninjau kondisi umum,
                perkembangan anak, catatan pada buku KIA, dan riwayat
                penyakit sebelumnya.

                Hasilnya mencatat kondisi {status_kesehatan},
                perkembangan {tumbuh_kembang}, serta riwayat {riwayat}.
                """,

                f"""
                Beberapa aspek kondisi {nama} diperiksa dalam satu
                kunjungan. Catatan menunjukkan kondisi umum
                {status_kesehatan} dan perkembangan {tumbuh_kembang}.

                Buku KIA juga ditinjau sebelum petugas memberikan
                rekomendasi kepada keluarga.
                """,

                f"""
                {nama}, usia {umur} bulan, datang untuk evaluasi rutin.
                Petugas meninjau perkembangan, kondisi umum, serta
                riwayat pelayanan anak.

                Berdasarkan hasil tersebut, keluarga diberi arahan
                untuk kunjungan berikutnya.
                """,
            ])

        return self.normalize_text(teks)

    # =====================================================
    # DATA AMBIGU TAMBAHAN
    # =====================================================

    def create_ambiguous_variation(
        self,
        text: str,
        label: str,
    ) -> str:
        """
        Menambahkan sedikit konteks lintas tema pada sebagian data.

        Konteks tambahan tidak mengubah ground truth, karena fokus utama
        laporan tetap ditentukan oleh sumber datanya.
        """

        if label == self.LABEL_GIZI:
            additions = [
                "Buku KIA juga diperiksa secara singkat oleh petugas.",
                "Orang tua diingatkan mengikuti jadwal pelayanan lainnya.",
                "Catatan pelayanan anak sebelumnya turut diperiksa.",
            ]

        elif label == self.LABEL_IMUNISASI:
            additions = [
                "Petugas juga menanyakan pola makan anak kepada orang tua.",
                "Kondisi fisik anak diamati sebelum tindakan diberikan.",
                "Orang tua turut berkonsultasi mengenai pertumbuhan anak.",
            ]

        else:
            additions = [
                "Petugas memberi perhatian khusus pada perkembangan anak.",
                "Keluarga diminta menjaga pola makan dan jadwal kunjungan.",
                "Catatan pelayanan sebelumnya menjadi bagian dari penilaian.",
            ]

        return self.normalize_text(
            f"{text} {self.choose(additions)}"
        )

    # =====================================================
    # PEMBENTUKAN DATASET
    # =====================================================

    def append_generated_rows(
        self,
        target: list[dict],
        dataframe: pd.DataFrame,
        generator_function,
        label: str,
        n_per_tema: Optional[int],
        ambiguity_ratio: float,
    ) -> None:
        sampled = self.sample_dataframe(
            dataframe,
            n_per_tema
        )

        for _, row in sampled.iterrows():
            difficulty = self.determine_difficulty()

            text = generator_function(
                row,
                difficulty=difficulty
            )

            is_ambiguous = self.random.random() < ambiguity_ratio

            if is_ambiguous:
                text = self.create_ambiguous_variation(
                    text,
                    label
                )

            target.append({
                "teks_laporan": text,
                "tema_aktual": label,
                "tingkat_kesulitan": difficulty,
                "ambigu": is_ambiguous,
            })

    def generate_all(
        self,
        n_per_tema: Optional[int] = None,
        shuffle: bool = True,
        ambiguity_ratio: float = 0.20,
    ) -> pd.DataFrame:
        if not 0 <= ambiguity_ratio <= 1:
            raise ValueError(
                "ambiguity_ratio harus berada antara 0 dan 1."
            )

        data: list[dict] = []

        if not self.gizi_df.empty:
            self.append_generated_rows(
                target=data,
                dataframe=self.gizi_df,
                generator_function=self.generate_from_gizi,
                label=self.LABEL_GIZI,
                n_per_tema=n_per_tema,
                ambiguity_ratio=ambiguity_ratio,
            )

        if not self.kesehatan_df.empty:
            self.append_generated_rows(
                target=data,
                dataframe=self.kesehatan_df,
                generator_function=self.generate_from_kesehatan,
                label=self.LABEL_KESEHATAN,
                n_per_tema=n_per_tema,
                ambiguity_ratio=ambiguity_ratio,
            )

        if not self.imunisasi_df.empty:
            self.append_generated_rows(
                target=data,
                dataframe=self.imunisasi_df,
                generator_function=self.generate_from_imunisasi,
                label=self.LABEL_IMUNISASI,
                n_per_tema=n_per_tema,
                ambiguity_ratio=ambiguity_ratio,
            )

        df = pd.DataFrame(data)

        if df.empty:
            raise ValueError(
                "Data kosong. Pastikan seluruh file CSV tersedia "
                "dan memiliki struktur kolom yang benar."
            )

        if shuffle:
            df = df.sample(
                frac=1,
                random_state=self.seed
            ).reset_index(drop=True)

        return df

    def generate_and_save(
        self,
        n_per_tema: Optional[int] = None,
        shuffle: bool = True,
        ambiguity_ratio: float = 0.20,
    ) -> pd.DataFrame:
        print("[INFO] Membuat teks laporan...")

        df = self.generate_all(
            n_per_tema=n_per_tema,
            shuffle=shuffle,
            ambiguity_ratio=ambiguity_ratio,
        )

        output_dir = os.path.dirname(GENERATED_DATA_PATH)

        if output_dir:
            os.makedirs(
                output_dir,
                exist_ok=True
            )

        df.to_csv(
            GENERATED_DATA_PATH,
            index=False
        )

        print(f"[INFO] Total teks dibuat: {len(df)}")
        print(
            "[INFO] Distribusi tingkat kesulitan:\n"
            f"{df['tingkat_kesulitan'].value_counts()}"
        )
        print(
            "[INFO] Jumlah data ambigu: "
            f"{int(df['ambigu'].sum())}/{len(df)}"
        )
        print(f"[INFO] File disimpan: {GENERATED_DATA_PATH}")

        return df
